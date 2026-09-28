# XCS221 Assignment 4 — MDPs and Reinforcement Learning (Mountain Car)

Due **Sunday, October 4, 11:59pm PT**. 50 points: **31 coding + 19 written**.

Code goes in `src/submission.py` (between the `START/END CODE HERE` markers only). Written answers go in
`tex/submission.tex`, built with `make` from `A4/`.

---

## Progress tracker

| Part | Type | Pts | What it is | Status |
|---|---|---|---|---|
| 1a  VI by hand, 2 iterations       | Written | 2   | 10 values: $V_1, V_2$ for all 5 states | ☐ |
| 1b  Policy from $V_2$              | Written | 1   | $\pi^\star$ for states −1, 0, +1 | ☐ |
| 2a  Discount → $\lambda'=1$ reduction | Written | 3 | define $T'$, $R'$ with the new state $o$ | ☐ |
| 3a  `valueIteration`               | Coding  | 5   | 0-basic 1 · 1-basic 2 · **2-hidden 2 (speed, n=500, 14 s)** | ☐ |
| 3b  `ModelBasedMonteCarlo`         | Coding  | 7   | 0-basic 1 · 1-basic 1 · **2-hidden 2 · 3-hidden 3** | ☐ |
| 3c  MBVI plot + discussion         | Written | 2   | run `train.py --agent value-iteration` | ☐ |
| 4a  `TabularQLearning`             | Coding  | 9   | 0-basic 4 · 1-basic 3 · **2-hidden 2** | ☐ |
| 4b  `fourierFeatureExtractor`      | Coding  | 4   | 0-basic 4 | ☐ |
| 4c  `FunctionApproxQLearning`      | Coding  | 6   | 0-basic 1 · 1-basic 2 · 2-basic 3 | ☐ |
| 4d  Tabular vs FA plots            | Written | 2   | two training runs + 2–3 sentences | ☐ |
| 4e  When FA beats tabular          | Written | 2   | 2–3 sentences | ☐ |
| 5a  How `max_speed` is used        | Written | 1   | one sentence | ☐ |
| 5b  FA with `max_speed=100000`     | Written | 1   | one training run + one sentence | ☐ |
| 5c  `ConstrainedQLearning` + why policies differ | Coding + Written | 2 | no graded code test; `grader.py 5c-0-helper` prints the comparison | ☐ |
| 5d  Two AV MDPs (harmful / mitigated) | Written | 0.5 | states, actions, reward, exploration policy, ethics note, ×2 | ☐ |
| 6a–6e  Product ethics: bias benchmarking | Written | 2.5 | 0.5 each, same product as A1/A2 (Claude Code) | ☐ |

**Hidden tests are 9 of the 31 coding points** (3a-2, 3b-2, 3b-3, 4a-2). Their assertions are stripped from the
local `grader.py` (empty `BEGIN_HIDE`/`END_HIDE` blocks), so a clean local run only shows 22/31.

---

## Where the points are

- **3b + 4a = 16 points**, over a third of the assignment. Both are "epsilon-greedy `getAction` + an
  `incorporateFeedback` update". Get the epsilon-greedy pattern right once and reuse it in 4a, 4c and 5c.
- **1a / 1b / 3a are the same MDP.** `run_VI_over_numberLine` with the default `NumberLineMDP()` is the Problem 1
  chain, so a finished 3a checks the policy you get by hand in 1b. (It runs to convergence, not 2 iterations, so it
  checks 1b's policy, not 1a's numbers.)
- **The Q-learning update is one line of maths used three times** (4a tabular, 4c weights, 5c inherits 4c):
  $$\hat Q(s,a) \leftarrow \hat Q(s,a) - \eta\big[\hat Q(s,a) - (r + \gamma \max_{a'} \hat Q(s',a'))\big]$$
  with the $\max_{a'}$ term dropped when $s'$ is terminal. In 4c the same residual multiplies $\phi(s)$ and
  updates column $a$ of `W`.

---

## Things spotted in the starter code and grader

### 3a `valueIteration`
- The provided loop is `while True:` with no exit. Until the convergence check is written, **any test that calls
  it hangs until the grader timeout**. Don't run the full grader before 3a is in.
- Terminal states never appear as keys in `succAndRewardProb`, so `V` being a `defaultdict(float)` is what gives
  them value 0. Only states in `stateActions` need updating.
- 3a-2-hidden runs `n=500` (≈1000 states) within 14 s. A plain dict-based loop is fine as long as nothing inside
  an iteration is quadratic in the number of states.

### 3b `ModelBasedMonteCarlo`
- `getAction` docstring: return a **random** action when `state` is not in `self.pi`. 3b-3-hidden calls
  `getAction` on a fresh agent over all five states, *including the terminal ones*, and counts how often action 1
  comes back. That is this branch.
- The starter comment says to avoid querying for random numbers when not exploring: when `explore=False`, go
  straight to the policy without calling `random.random()`.
- 3b-0-basic expects the greedy action between 8800 and 9200 times out of 10000 at `explorationProb=0.2`. That
  window is 0.8 + 0.2 × ½, i.e. the random branch picks uniformly from **all** actions (including the greedy one).
- `incorporateFeedback`: the estimates are $\hat T(s'|s,a) = \text{count}/\sum\text{counts}$ and
  $\hat R(s,a,s') = \text{rTotal}/\text{count}$. 3b-1-basic compares `rl.pi` to `{1: 1, -1: 2}` with `==`, so
  the policy must contain only the states that have been observed.

### 4a `TabularQLearning`
- 4a-0-basic spells out the update: from all-zero Q, `incorporateFeedback(0, 1, -5, 1, False)` gives
  `Q[(0,1)] = -0.5`: step size 0.1 times the residual (−5 + 0 − 0).
- `self.Q` is a `defaultdict`, so reading `Q[(s', a')]` inserts keys. Harmless here, but worth knowing when you
  inspect `rl.Q`.

### 4b `fourierFeatureExtractor`
- Same shape of code as `util.polynomialFeatureExtractor` with **outer sum instead of outer product**, then
  `cos(π · ...)` at the end. Apply `scale` before the outer sum. Output length is `(maxCoeff+1)**len(state)`.
- The grader sorts both arrays before comparing, so feature order doesn't matter.

### 4c `FunctionApproxQLearning`
- `self.W` has shape `(featureDim, len(actions))`, one weight column per action. The feature extractor ignores
  the action.
- 4c-2-basic compares `W` row-by-row up to permutation (again, feature order is free).

### 5c `ConstrainedQLearning`
- Uses the velocity formula from the handout with `self.force` and `self.gravity`. Keep only actions whose
  predicted next velocity is below `self.max_speed`, and **handle the case where none qualify**.
- No graded code test. `python grader.py 5c-0-helper` (60 s timeout) prints action counts for
  `max_speed=10000` vs `0.065`, which is what the written part asks about.

### 5a
- The question asks about `custom_mountain_car.py`: `max_speed` sets the observation-space bounds (lines 113–114)
  and is where velocity gets clipped (line 135).

---

## Running things

```bash
cd A4/src
python grader.py 3a-0-basic          # one test
python grader.py                     # everything (only once 3a terminates)
python train.py --agent value-iteration      # 3c
python train.py --agent tabular              # 4d
python train.py --agent function-approximation                   # 4d
python train.py --agent function-approximation --max_speed=100000  # 5b
```

Each `train.py` run is 3500 training episodes + 500 eval episodes and saves a reward plot. The written parts
need those plots, so **3c, 4d and 5b wait on the coding**.

`mountaincar.py --agent ...` opens a pygame window, so it needs a display and won't run in a headless session.

The testbench (`python tools/testbench/server.py`) picks up `A4` automatically for stepping through a test.

Environment note: `numpy 2.4.6` + `gymnasium 1.3.0` installed via pip in this container. The pinned versions in
`src/requirements.txt` are what the course expects.

---

## Suggested order

1. **1a → 1b by hand** (short, and builds the intuition 3a needs).
2. **3a**, then check it reproduces your 1b policy.
3. **3b**, **4a**, then **4b → 4c → 5c** (each builds on the previous one).
4. The training runs for **3c / 4d / 5b**, then those written parts.
5. **2a** whenever: it is standalone (hint: compare the two Bellman recurrences).
6. **5d**, **6a–6e**: no dependencies. 6b needs you to actually run your contrastive inputs through the product,
   so leave time for that.
