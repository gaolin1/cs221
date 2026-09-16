"""Run a single grader test under a line tracer and write the recording as JSON.

Invoked by server.py in a fresh subprocess, with the assignment's src directory as
the working directory, so that each run starts from a clean interpreter and a
crash in student code cannot take the server down.

Only frames whose source file is in the requested trace list are recorded. Frames
from the standard library, numpy, torch and so on are never traced, which keeps
the recording fast and small.
"""
import argparse
import ast
import contextlib
import io
import json
import os
import reprlib
import signal
import sys
import time
import traceback
import unittest


REPR_LIMIT = 300  # hard cap on any single recorded value


class SafeRepr(reprlib.Repr):
    """reprlib with dict and sequence subclasses handled, so that a large defaultdict
    or similar is summarised instead of being fully formatted on every step."""

    def __init__(self):
        super().__init__()
        self.maxlevel = 3
        self.maxdict = 12
        self.maxlist = 12
        self.maxtuple = 12
        self.maxset = 12
        self.maxstring = 160
        self.maxother = 160

    def repr1(self, obj, level):
        if isinstance(obj, dict) and type(obj) is not dict:
            return type(obj).__name__ + self.repr_dict(obj, level)
        if isinstance(obj, list) and type(obj) is not list:
            return type(obj).__name__ + self.repr_list(obj, level)
        return super().repr1(obj, level)


_repr = SafeRepr()


def safe_repr(obj):
    try:
        text = _repr.repr(obj)
    except Exception as exc:  # a broken __repr__ must not stop the recording
        text = f"<repr failed: {type(exc).__name__}>"
    if len(text) > REPR_LIMIT:
        text = text[:REPR_LIMIT] + "..."
    return text


# ---------------------------------------------------------------------------------
# Static analysis: for each statement, which names it reads, rebinds and mutates.
# Combined with the recording, this gives the page a dependency graph of variables.
# ---------------------------------------------------------------------------------

# Method calls that change their receiver in place, so `xs.append(v)` counts as a write to xs.
MUTATING_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "update", "add", "discard",
    "sort", "reverse", "setdefault", "popitem", "appendleft", "extendleft", "popleft",
    "add_", "sub_", "mul_", "div_", "zero_", "fill_", "copy_", "clamp_", "normal_", "uniform_",
}


class _NameCollector(ast.NodeVisitor):
    """Collect names read and written in an expression, skipping names bound inside a
    comprehension or lambda, which belong to that inner scope."""

    def __init__(self):
        self.loads = []
        self.stores = []
        self.nodes = []  # (Name node, "r" or "w") for every name kept, with its position
        self._scopes = []

    def _bound(self, name):
        return any(name in scope for scope in self._scopes)

    def visit_Name(self, node):
        if self._bound(node.id):
            return
        if isinstance(node.ctx, ast.Load):
            self.loads.append(node.id)
            self.nodes.append((node, "r"))
        else:
            self.stores.append(node.id)
            self.nodes.append((node, "w"))

    def _comprehension(self, node, results):
        bound = {n.id for gen in node.generators for n in ast.walk(gen.target) if isinstance(n, ast.Name)}
        self.visit(node.generators[0].iter)  # evaluated in the enclosing scope
        self._scopes.append(bound)
        for index, gen in enumerate(node.generators):
            if index:
                self.visit(gen.iter)
            for condition in gen.ifs:
                self.visit(condition)
        for expr in results:
            self.visit(expr)
        self._scopes.pop()

    def visit_ListComp(self, node):
        self._comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._comprehension(node, [node.key, node.value])

    def visit_Lambda(self, node):
        args = node.args
        bound = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        bound.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
        for default in args.defaults + [d for d in args.kw_defaults if d is not None]:
            self.visit(default)
        self._scopes.append(bound)
        self.visit(node.body)
        self._scopes.pop()


def _expr_names(node, loads, mutates, marks):
    collector = _NameCollector()
    collector.visit(node)
    loads.extend(collector.loads)
    marks.extend(collector.nodes)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in MUTATING_METHODS:
            base = sub.func.value
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if isinstance(base, ast.Name):
                mutates.append(base.id)
                marks.append((base, "m"))


def _target_names(target, stores, mutates, loads, marks):
    """Split an assignment target into names that are rebound (x = ...) and objects
    that are changed in place (x.attr = ..., x[i] = ...)."""
    if isinstance(target, ast.Name):
        stores.append(target.id)
        marks.append((target, "w"))
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _target_names(element, stores, mutates, loads, marks)
    elif isinstance(target, ast.Starred):
        _target_names(target.value, stores, mutates, loads, marks)
    elif isinstance(target, (ast.Attribute, ast.Subscript)):
        base = target
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            if isinstance(base, ast.Subscript):
                _expr_names(base.slice, loads, mutates, marks)
            base = base.value
        if isinstance(base, ast.Name):
            mutates.append(base.id)
            marks.append((base, "m"))
        else:
            _expr_names(base, loads, mutates, marks)


def _dedupe(items):
    return list(dict.fromkeys(items))


def analyze_source(source):
    """Map line number -> {"s": rebound, "m": mutated, "r": read, "k": statement kind}.
    Only the statement's own header counts, not the body of an if or for."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    info = {}
    rank = {"r": 0, "w": 1, "m": 2}  # when one occurrence has two roles, the stronger wins
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        stores, mutates, loads, marks = [], [], [], []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _target_names(target, stores, mutates, loads, marks)
            _expr_names(node.value, loads, mutates, marks)
        elif isinstance(node, ast.AugAssign):
            _target_names(node.target, stores, mutates, loads, marks)
            if isinstance(node.target, ast.Name):
                loads.append(node.target.id)
            _expr_names(node.value, loads, mutates, marks)
        elif isinstance(node, ast.AnnAssign):
            _target_names(node.target, stores, mutates, loads, marks)
            if node.value is not None:
                _expr_names(node.value, loads, mutates, marks)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _target_names(node.target, stores, mutates, loads, marks)
            _expr_names(node.iter, loads, mutates, marks)
        elif isinstance(node, (ast.While, ast.If)):
            _expr_names(node.test, loads, mutates, marks)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                _expr_names(item.context_expr, loads, mutates, marks)
                if item.optional_vars is not None:
                    _target_names(item.optional_vars, stores, mutates, loads, marks)
        elif isinstance(node, (ast.Return, ast.Expr)):
            if node.value is not None:
                _expr_names(node.value, loads, mutates, marks)
        elif isinstance(node, ast.Assert):
            _expr_names(node.test, loads, mutates, marks)
            if node.msg is not None:
                _expr_names(node.msg, loads, mutates, marks)
        elif isinstance(node, ast.Raise):
            for part in (node.exc, node.cause):
                if part is not None:
                    _expr_names(part, loads, mutates, marks)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                _target_names(target, stores, mutates, loads, marks)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                stores.append((alias.asname or alias.name).split(".")[0])
        else:
            continue
        entry = info.setdefault(node.lineno, {"s": [], "m": [], "r": [], "k": type(node).__name__, "t": []})
        entry["s"] = _dedupe(entry["s"] + stores)
        entry["m"] = _dedupe(entry["m"] + mutates)
        entry["r"] = _dedupe(entry["r"] + loads)
        # every occurrence of a name on the statement's own lines, with its position and role,
        # so the page can colour the exact token: [line, column, role, name]
        roles = {(t[0], t[1]): t for t in entry["t"]}
        for name_node, role in marks:
            key = (name_node.lineno, name_node.col_offset)
            existing = roles.get(key)
            if existing is None or rank[role] > rank[existing[2]]:
                roles[key] = [name_node.lineno, name_node.col_offset, role, name_node.id]
        entry["t"] = sorted(roles.values())
    return info


class Recorder:
    def __init__(self, src, trace_names, max_steps):
        self.max_steps = max_steps
        self.steps = []
        self.recording = True
        self.truncated = False
        self.depth = 0
        self.files = []
        self.file_ids = {}
        for name in trace_names:
            path = os.path.normcase(os.path.abspath(os.path.join(src, name)))
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as handle:
                    source = handle.read()
                self.file_ids[path] = len(self.files)
                self.files.append({
                    "id": len(self.files), "name": name, "source": source,
                    "lines": analyze_source(source),
                })

    def _file_id(self, frame):
        return self.file_ids.get(os.path.normcase(os.path.abspath(frame.f_code.co_filename)))

    def _locals(self, frame):
        values = {}
        for name, value in frame.f_locals.items():
            if name.startswith("__") and name.endswith("__"):
                continue
            values[name] = safe_repr(value)
        return values

    def _add(self, event, frame, fid, extra=None):
        if len(self.steps) >= self.max_steps:
            self.recording = False
            self.truncated = True
            return False
        step = {
            "e": event,
            "f": fid,
            "l": frame.f_lineno,
            "fn": frame.f_code.co_name,
            "d": self.depth,
            "v": self._locals(frame),
        }
        if extra:
            step.update(extra)
        self.steps.append(step)
        return True

    def global_trace(self, frame, event, arg):
        if not self.recording or event != "call":
            return None
        fid = self._file_id(frame)
        if fid is None:
            return None
        self.depth += 1
        if not self._add("call", frame, fid):
            return None
        return self.local_trace

    def local_trace(self, frame, event, arg):
        if not self.recording:
            return None
        fid = self._file_id(frame)
        if event == "line":
            if not self._add("line", frame, fid):
                return None
        elif event == "return":
            self._add("return", frame, fid, {"r": safe_repr(arg)})
            self.depth -= 1
        elif event == "exception":
            exc_type, exc_value, _ = arg
            try:
                message = str(exc_value)
            except Exception:
                message = "<unprintable>"
            if len(message) > REPR_LIMIT:
                message = message[:REPR_LIMIT] + "..."
            self._add("exception", frame, fid, {"x": f"{exc_type.__name__}: {message}"})
        return self.local_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--trace", default="submission.py,grader.py")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    os.chdir(src)
    sys.path.insert(0, src)

    # The graders use SIGALRM for timeouts on Linux and macOS. Tracing slows code
    # down, so a real alarm would fail tests that pass normally; disable it here.
    if hasattr(signal, "alarm"):
        signal.alarm = lambda *_: 0

    trace_names = [n.strip() for n in args.trace.split(",") if n.strip()]
    recorder = Recorder(src, trace_names, args.max_steps)
    result = {
        "test": args.test,
        "status": None,
        "message": "",
        "traceback": "",
        "stdout": "",
        "elapsed": None,
        "timeout": None,
        "hidden": "hidden" in args.test,
        "files": recorder.files,
        "steps": recorder.steps,
        "truncated": False,
        "max_steps": args.max_steps,
        "failure": None,
    }

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            import grader  # imports the student's submission as a side effect
            test = grader.getTestCaseForTestID(args.test)
        if test is None:
            raise LookupError(f"No test named {args.test} in grader.py")
        method = getattr(test, test._testMethodName, None)
        result["timeout"] = getattr(method, "__timeout__", None)
    except BaseException:
        result["status"] = "error"
        result["message"] = "The grader or submission failed to import."
        result["traceback"] = traceback.format_exc()
        result["stdout"] = captured.getvalue()
        write(args.out, result)
        return

    outcome = unittest.TestResult()
    start = time.perf_counter()
    sys.settrace(recorder.global_trace)
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            test.run(outcome)
    finally:
        sys.settrace(None)
    result["elapsed"] = time.perf_counter() - start
    result["stdout"] = captured.getvalue()
    result["truncated"] = recorder.truncated

    problems = outcome.failures + outcome.errors
    if problems:
        kind = "fail" if outcome.failures else "error"
        _, formatted = problems[0]
        result["status"] = kind
        result["traceback"] = formatted
        result["message"] = formatted.strip().splitlines()[-1] if formatted.strip() else kind
        # unittest keeps only the formatted text, so recover the location from it
        result["failure"] = locate_from_text(formatted, recorder)
    elif outcome.skipped:
        result["status"] = "skipped"
        result["message"] = outcome.skipped[0][1]
    else:
        result["status"] = "pass"

    if result["failure"] is not None:
        fid, line = result["failure"]["f"], result["failure"]["l"]
        # Prefer the exception event at the failing line, which shows what was raised;
        # Python emits a return event on the same line right after it.
        match = fallback = None
        for index in range(len(recorder.steps) - 1, -1, -1):
            step = recorder.steps[index]
            if step["f"] == fid and step["l"] == line:
                if fallback is None:
                    fallback = index
                if step["e"] == "exception":
                    match = index
                    break
        result["failure"]["step"] = match if match is not None else fallback

    write(args.out, result)


def locate_from_text(formatted, recorder):
    """Find the innermost 'File "...", line N' in a formatted traceback that points
    at a recorded file."""
    found = None
    for raw in formatted.splitlines():
        raw = raw.strip()
        if not raw.startswith('File "'):
            continue
        try:
            path = raw.split('"')[1]
            line = int(raw.split("line ")[1].split(",")[0])
        except (IndexError, ValueError):
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key in recorder.file_ids:
            found = {"f": recorder.file_ids[key], "l": line}
    return found


def write(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    main()
