# Testbench

Pick an assignment and a grader test, record a run of it, then step through the
recording in the browser like a debugger: forwards, backwards, over, out, and straight
to the line that failed.

```
python tools/testbench/server.py
```

Then open http://127.0.0.1:8765. Use the same Python you run the graders with, since
the tests import that interpreter's packages (for A3, that means `osmium` installed).

## Using it

1. Choose the assignment in the header. Every folder like `A1`, `A2`, `A3` with a
   `src/grader.py` is found automatically.
2. Pick a test on the left. Before running, the grader source opens with that test's
   body shaded, so you can read exactly what it sets up and asserts.
3. Tick which files to record (`submission.py` and `grader.py` by default; add
   `util.py` to watch the search algorithm call your methods).
4. Press **Run test** (or double-click the test, or Ctrl+Enter).

After a run the view jumps to the failing line if there is one, otherwise to the
first step. The right-hand pane shows the result, the current event, every local
variable (values that changed since the previous step in the same function are
highlighted), and the call stack, where clicking a frame jumps to it.

### What each line works on

As you step, the current line shows exactly which data it operates on:

- **In the code**, each variable name on the line is boxed by its role: **blue** if the
  line reads it, **green** if it assigns it, **amber** if it changes it in place (for
  example `xs.append(v)`, `d[k] = v` or `self.x = v`). In `x = x + 1` the left `x` is
  green and the right one blue.
- **In the Variables panel**, the variables that line touches move to the top under
  *This line works on*, tagged READS, WRITES or CHANGES. Anything it assigns or changes
  shows **before &rarr; after**, because a step captures the moment just before its line
  runs; a variable the line creates shows **new &rarr; value**.
- Globals and imported names (like `State` or `np`) are not highlighted, only the
  function's own variables.

### Breakpoints

Click a line number to set a breakpoint (click again to remove), or press **B** on the
current line. They are saved per assignment and kept between runs, since they belong
to a file and line rather than to one recording.

- **bp ▶** (F8) jumps to the next time the recording reaches a breakpoint line, and
  **◀ bp** (Shift+F8) jumps to the previous one. Because it's a replay, you can
  continue backwards, which a live debugger can't.
- After a run, the view stops at the first breakpoint hit if any breakpoint is reached;
  otherwise it opens on the failure.
- The Breakpoints list shows how many times each one was reached. Clicking an entry
  jumps to its next hit. A hollow marker means that line never ran in this recording.
- A breakpoint on a line stops once per time the line runs, not again when the same
  line returns.

### Variable flow

Press **Variable flow** above the code (or **V**) to see how the variables of the
function you're stepping in are created and used. Clicking a variable name in the
Variables panel opens it with that variable already highlighted.

- One lane per variable, one column per moment something happened to a variable. The
  numbers along the top are source line numbers, so a loop shows up as a repeating run.
- Markers show each variable arriving as a **parameter**, being **created**,
  **reassigned**, **mutated** in place (for example `xs.append(v)` or `self.x = v`),
  **read**, and the value **returned** or exception **raised**. A dashed outline on a
  mutation means something changed the object without that line writing to it, such as
  a function it called.
- Arrows run from the values a line read to the value it produced.
- Click a lane name to highlight everything that fed that variable and everything it
  went on to feed; click it again or press Esc to clear. Click a marker to jump to that
  step, and double-click to jump there in the code.
- Hover a marker for the value, the step, and the source line.
- The dashed line marks the current step, and moves as you step.

How it knows: when a run is recorded, each file is parsed to find which names every
line reads, rebinds and mutates, and that is matched against the recorded values. It
shows the function containing the current step; use the Call stack to move up to the
caller's flow. Very long functions show 600 events at a time around the current step.

| Key | Action |
| --- | --- |
| V | switch between code and variable flow |
| Esc | clear the variable flow highlight |
| B | toggle a breakpoint on the current line |
| F8 / Shift+F8 | continue forward / back to a breakpoint hit |
| Right / Left | next / previous step |
| Shift+Right / Shift+Left | step over / back over calls |
| Up | step out of the current function |
| Home / End | first / last step |
| N | next line in `submission.py` |
| F | jump to the failure |
| Ctrl+Enter | run the selected test |

## How it works

- `server.py` serves the page and a small JSON API on 127.0.0.1 only. It lists tests
  by parsing `grader.py` with `ast`, so the list still loads when your submission
  crashes on import.
- `recorder.py` runs one test in a fresh subprocess under `sys.settrace`, recording
  each line, call, return and exception in the chosen files together with a
  summarised copy of the local variables. Frames from other files are never traced,
  which keeps runs fast.
- The page replays that recording; nothing runs while you step.

## Limits to know

- Recording stops at **Max steps** (20,000 by default) but the test still runs to the
  end, so the pass or fail result is always real. Searches over the Stanford map can
  hit the cap; raise it, or untick `util.py`.
- Variable values are summaries capped at 300 characters, and large containers show
  only their first dozen entries.
- Tracing makes code slower, so the recorded time is not the graded time. The
  SIGALRM timeouts the graders use on Linux and macOS are disabled while recording.
- Hidden tests run your code but report "RAN (hidden)", because they are only scored
  by the real autograder.
- A run is stopped after 180 seconds, which catches infinite loops.
