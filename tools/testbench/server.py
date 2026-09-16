"""Local testbench: pick an assignment and a grader test, record a run, step through it.

    python tools/testbench/server.py            then open http://127.0.0.1:8765

Listens on 127.0.0.1 only. Each run happens in a fresh subprocess (recorder.py), so a
crash or infinite loop in student code cannot take the server down.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STATIC = os.path.join(HERE, "static")
RECORDER = os.path.join(HERE, "recorder.py")
ASSIGNMENT_DIR = re.compile(r"^A\d+$")
RUN_TIMEOUT = 180


def discover_assignments():
    found = []
    for name in sorted(os.listdir(REPO), key=lambda n: (len(n), n)):
        src = os.path.join(REPO, name, "src")
        if ASSIGNMENT_DIR.match(name) and os.path.isfile(os.path.join(src, "grader.py")):
            found.append(name)
    return found


def src_dir(assignment):
    if assignment not in discover_assignments():
        raise ValueError(f"Unknown assignment: {assignment!r}")
    return os.path.join(REPO, assignment, "src")


def list_tests(assignment):
    """Read the grader with ast rather than importing it, so the list still loads
    when the submission is broken."""
    path = os.path.join(src_dir(assignment), "grader.py")
    with open(path, encoding="utf-8", errors="replace") as handle:
        tree = ast.parse(handle.read())
    tests = []
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name.startswith("Test_")):
            continue
        question = node.name[len("Test_"):]
        for fn in node.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_")):
                continue
            doc = (ast.get_docstring(fn) or "").strip()
            first = doc.splitlines()[0] if doc else ""
            if ":" in first and "-" in first.split(":")[0]:
                test_id, description = first.split(":", 1)
                test_id, description = test_id.strip(), description.strip()
            else:
                test_id, description = f"{question}-{fn.name[len('test_'):]}-basic", first
            decorators = " ".join(ast.unparse(d) for d in fn.decorator_list)
            hidden = "hidden" in test_id or "is_hidden=True" in decorators
            parts = test_id.split("-")
            kind = "hidden" if hidden else (parts[2] if len(parts) >= 3 else "basic")
            timeout = re.search(r"timeout\s*=\s*(\d+)", decorators)
            tests.append({
                "id": test_id,
                "kind": kind,
                "question": question,
                "description": description,
                "hidden": hidden,
                "timeout": int(timeout.group(1)) if timeout else 5,
                "start": fn.lineno,
                "end": fn.end_lineno,
            })
    return tests


def python_files(assignment):
    src = src_dir(assignment)
    return sorted(n for n in os.listdir(src) if n.endswith(".py"))


def run_test(assignment, test_id, trace, max_steps):
    src = src_dir(assignment)
    known = {t["id"] for t in list_tests(assignment)}
    if test_id not in known:
        raise ValueError(f"Unknown test: {test_id!r}")
    allowed = set(python_files(assignment))
    trace = [name for name in trace if name in allowed] or ["submission.py", "grader.py"]
    max_steps = max(100, min(int(max_steps), 200000))

    handle, out_path = tempfile.mkstemp(prefix="testbench_", suffix=".json")
    os.close(handle)
    try:
        completed = subprocess.run(
            [sys.executable, RECORDER, "--src", src, "--test", test_id, "--out", out_path,
             "--max-steps", str(max_steps), "--trace", ",".join(trace)],
            cwd=src, capture_output=True, text=True, timeout=RUN_TIMEOUT,
        )
        with open(out_path, encoding="utf-8") as result_file:
            content = result_file.read()
        if not content.strip():
            return {"status": "error", "message": "The recorder produced no output.",
                    "traceback": completed.stderr, "steps": [], "files": [], "test": test_id}
        return json.loads(content)
    except subprocess.TimeoutExpired:
        return {"status": "error", "test": test_id, "steps": [], "files": [],
                "message": f"Run exceeded {RUN_TIMEOUT} s and was stopped (possible infinite loop).",
                "traceback": ""}
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


_INSPECT_CACHE = {}
INSPECT_CACHE_MAX = 200


def inspect_value(assignment, test_id, trace, max_steps, step, name, path, offset, expected):
    """Re-run the test, stop at `step`, and describe `name` (optionally walking into it).

    The recording only keeps a short text of each value, so opening one up means running
    the test again and looking at the real object. `expected` is the text that was
    recorded, so a re-run that produces something different can be reported as stale.
    """
    key = json.dumps([assignment, test_id, sorted(trace), max_steps, step, name, path, offset], sort_keys=True)
    if key in _INSPECT_CACHE:
        return _INSPECT_CACHE[key]

    src = src_dir(assignment)
    known = {t["id"] for t in list_tests(assignment)}
    if test_id not in known:
        raise ValueError(f"Unknown test: {test_id!r}")
    allowed = set(python_files(assignment))
    trace = [n for n in trace if n in allowed] or ["submission.py", "grader.py"]
    spec = {"step": int(step), "name": name, "path": path or [], "offset": int(offset), "depth": 2, "limit": 25}

    spec_handle, spec_path = tempfile.mkstemp(prefix="testbench_spec_", suffix=".json")
    out_handle, out_path = tempfile.mkstemp(prefix="testbench_insp_", suffix=".json")
    os.close(spec_handle)
    os.close(out_handle)
    try:
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle)
        completed = subprocess.run(
            [sys.executable, RECORDER, "--src", src, "--test", test_id, "--out", out_path,
             "--max-steps", str(int(max_steps)), "--trace", ",".join(trace), "--inspect", spec_path],
            cwd=src, capture_output=True, text=True, timeout=RUN_TIMEOUT,
        )
        with open(out_path, encoding="utf-8") as handle:
            content = handle.read()
        if not content.strip():
            return {"error": "The re-run produced no output.", "detail": completed.stderr[-400:]}
        result = json.loads(content)
    except subprocess.TimeoutExpired:
        return {"error": f"The re-run exceeded {RUN_TIMEOUT} s."}
    finally:
        for path_to_remove in (spec_path, out_path):
            try:
                os.remove(path_to_remove)
            except OSError:
                pass

    if expected and result.get("rootPreview"):
        # object reprs carry a memory address that differs on every run, so compare
        # with addresses blanked out; anything left over is a real difference
        without_address = lambda text: re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
        if without_address(result["rootPreview"]) != without_address(expected):
            result["stale"] = True
    if len(_INSPECT_CACHE) > INSPECT_CACHE_MAX:
        _INSPECT_CACHE.clear()
    _INSPECT_CACHE[key] = result
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[testbench] " + (fmt % args) + "\n")

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, content_type):
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
            if url.path == "/api/assignments":
                return self._json({"assignments": discover_assignments()})
            if url.path == "/api/tests":
                assignment = query.get("assignment", "")
                return self._json({"tests": list_tests(assignment), "files": python_files(assignment)})
            if url.path == "/api/source":
                assignment, name = query.get("assignment", ""), query.get("name", "")
                if name not in python_files(assignment):
                    raise ValueError(f"Unknown file: {name!r}")
                with open(os.path.join(src_dir(assignment), name), encoding="utf-8", errors="replace") as h:
                    return self._json({"name": name, "source": h.read()})
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if url.path == "/api/run":
                result = run_test(payload.get("assignment", ""), payload.get("test", ""),
                                  payload.get("trace", ["submission.py", "grader.py"]),
                                  payload.get("max_steps", 20000))
                return self._json(result)
            if url.path == "/api/inspect":
                return self._json(inspect_value(
                    payload.get("assignment", ""), payload.get("test", ""),
                    payload.get("trace", ["submission.py", "grader.py"]), payload.get("max_steps", 20000),
                    payload.get("step", 0), payload.get("name", ""), payload.get("path", []),
                    payload.get("offset", 0), payload.get("expected"),
                ))
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


class Server(ThreadingHTTPServer):
    # Windows lets a second process bind the same port when SO_REUSEADDR is on, which
    # silently leaves an older instance answering some requests. Refuse instead.
    allow_reuse_address = False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        print(f"Could not listen on port {args.port}: {exc}")
        print("Another testbench is probably already running. Stop it, or pass --port.")
        raise SystemExit(1)
    print(f"Testbench running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    print(f"Assignments found: {', '.join(discover_assignments()) or 'none'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
