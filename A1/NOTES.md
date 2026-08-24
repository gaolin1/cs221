# XCS221 Assignment 1 — Foundations

Working notes and progress log. Due Sunday, Aug 30, 11:59pm PT.

**Written answers** go in a PDF submission. **Coding answers** go in `src/submission.py`.

---

## Progress tracker

| Part | Type | Pts | Status |
|---|---|---|---|
| 1a  Numpy AI-tutor session      | Written | 1 | ☐ not started |
| 1b  Matmul complexity           | Written | 2 | ☑ **done** — `O(mnp)` |
| 1c  einsum AI-tutor session     | Written | 1 | ☐ not started |
| 1d  Einstein summation strings  | Written | 2 | ◐ **in progress** |
| 1e  `linear_project`            | Coding  | 3 | ☐ |
| 1f  `split_last_dim_pattern`    | Coding  | 3 | ☐ |
| 1g  `normalized_inner_products` | Coding  | 3 | ☐ |
| 1h  `mask_strictly_upper`       | Coding  | 3 | ☐ |
| 1i  `prob_weighted_sum_einsum`  | Coding  | 3 | ☐ |
| 2a  Gradient warmup             | Written | 2 | ☐ |
| 2b  `gradient_warmup`           | Coding  | 3 | ☐ |
| 2c  Matmul gradient by hand     | Written | 3 | ☐ |
| 2d  `matrix_grad`               | Coding  | 3 | ☐ |
| 2e  `lsq_grad` / `lsq_finite_diff_grad` | Coding | 3 | ☐ |
| 3a  Optimization warmup         | Written | 3 | ☐ |
| 3b  Gradient descent tutor session | Written | 1 | ☐ |
| 3c  `gradient_descent_quadratic`| Coding  | 3 | ☐ |
| 4a–4d  Ethical issue spotting   | Written | 4 | ☐ |
| 5a–5b  Product ethics basics    | Written | 2 | ☐ |

### Blockers
- [ ] **Starter code missing.** `A1/submission.py` in this repo is 7 bytes containing `{\rtf1}`. Need the real
  handout: `src/submission.py`, `src/grader.py`, `src/environment.yml`. Autograder is run with
  `python grader.py` from inside `src/`.

---

## 1b — Matrix multiplication complexity ✅

**Answer:** `O(mnp)`

**Justification:** For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB$ has
$m \cdot p$ entries. Each entry is an inner product of length $n$:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

so each costs $n$ multiply-adds. Total: $mp \cdot n = O(mnp)$.

Caveats worth stating:
- This is the **standard / naive** algorithm. Strassen-style methods beat $O(n^3)$ on square matrices.
- Sanity check: square case $m = n = p$ gives $O(n^3)$. ✓

---

## Notation reference

Cheat sheet for the symbols this assignment keeps using.

### $\mathbb{R}$ — the real numbers

The set of all real numbers: $-3$, $0$, $\tfrac{1}{2}$, $\pi$, $1.7$. Not complex, not integers-only.

### $\in$ — "is an element of" / "lives in"

`x ∈ S` reads "x is in the set S." So $w \in \mathbb{R}$ means "$w$ is a real number."

### $\mathbb{R}^{n \times d}$ — the shape of a matrix

**The superscript is NOT exponentiation.** It is a *shape label*.

$$X \in \mathbb{R}^{n \times d} \quad\Longleftrightarrow\quad X \text{ is a table of real numbers with } n \text{ rows and } d \text{ columns}$$

In NumPy this is exactly `X.shape == (n, d)`.

```
        d columns
      ┌───────────────┐
      │ x11 x12 ... x1d │
 n    │ x21 x22 ... x2d │
rows  │  :   :       :  │
      │ xn1 xn2 ... xnd │
      └───────────────┘
```

Read it as *"the set of all n-by-d real matrices."* $X$ is one particular member of that set.

Why the `×`? There *is* a real connection to multiplication — an $n \times d$ matrix holds $n \cdot d$ individual
numbers, so it has $nd$ degrees of freedom. But the notation is telling you the **layout** (2 axes, sized $n$ and
$d$), not a single number $n^d$.

### Related forms

| Notation | Means | NumPy shape |
|---|---|---|
| $w \in \mathbb{R}$ | a scalar | `()` |
| $w \in \mathbb{R}^{d}$ | a vector of length $d$ | `(d,)` |
| $X \in \mathbb{R}^{n \times d}$ | matrix, $n$ rows × $d$ cols | `(n, d)` |
| $A \in \mathbb{R}^{B \times M \times D}$ | 3-axis tensor (a *batch* of $B$ matrices) | `(B, M, D)` |

A single superscript ($\mathbb{R}^d$) = 1 axis = vector. Two ($\mathbb{R}^{n\times d}$) = 2 axes = matrix. Three = 3 axes.
The count of numbers separated by $\times$ is the **number of axes**; each number is the **length of that axis**.

### $X^\top$ — transpose

Flip rows and columns. If $X \in \mathbb{R}^{n \times d}$ then $X^\top \in \mathbb{R}^{d \times n}$.
`X.T` in NumPy.

### $\langle a, b \rangle$ — inner (dot) product

$\langle a, b \rangle = \sum_k a_k b_k$ for two vectors of the same length. Returns a **scalar**.

### $\operatorname{diag}(M)$

For a square $M$, the vector of its diagonal entries $[M_{11}, M_{22}, \dots]$. `np.diag(M)` in NumPy.

### Shape-checking habit

Before writing any einsum, write down the input shapes and the target output shape. Most einsum bugs are shape
bugs, and the notation above *is* the shape spec — it is being handed to you on purpose.

```
1d(i):   X:(n,d)  ·  w:(d,)     ->  (n,)
1d(ii):  X:(n,d)  ·  X:(n,d)    ->  (n,n)
1d(iii): X:(n,d)  ·  X:(n,d)    ->  (d,)
```

---

## 1c — einsum tutor session (in progress)

### The one rule

Label every axis with a letter, then write `inputs -> output`:

- A letter appearing in the inputs but **not** in the output is **summed over**.
- A letter appearing in both is **kept** (matched elementwise / carried through).

That's it. Matrix multiply:

```python
from einops import einsum
C = einsum(A, B, 'm n, n p -> m p')   # 'n' is absent from output => summed over
```

The three distinct letters `m`, `n`, `p` are exactly the three nested loops from 1b — so the string literally
displays the $O(mnp)$ cost.

---

## 1d — Einstein summation strings

**Task:** given $X \in \mathbb{R}^{n \times d}$ and $w \in \mathbb{R}^d$, write einsum strings for
(i) $Xw$, (ii) $XX^\top$, (iii) $\operatorname{diag}(X^\top X)$, with brief justification for each.

### (i) $Xw$ — `(n,d), (d,) -> (n,)` ✅

```python
einsum(X, w, 'n d, d -> n')
```

**Justification:** $w$'s single axis is labeled `d` so it aligns with $X$'s column axis. `d` is absent from the
output, so it is summed over — giving $(Xw)_i = \sum_d X_{id} w_d$. `n` is carried through, so the result has
one entry per row of $X$.

### (ii) $XX^\top$ — `(n,d), (n,d) -> (n,n)` ✅

```python
einsum(X, X, 'i d, j d -> i j')
```

**Justification:** Entry $[i,j]$ is the dot product of row $i$ with row $j$:
$(XX^\top)_{ij} = \sum_d X_{id} X_{jd}$. The two row axes are *different* rows, so they get **different**
letters `i` and `j`, both kept in the output. The feature axis `d` is shared by both operands and dropped from
the output, so it is contracted. No explicit transpose is needed — the transpose is expressed by *which letters
line up*, not by flipping an array.

### (iii) $\operatorname{diag}(X^\top X)$ — `(n,d), (n,d) -> (d,)`

- [ ] TODO

---

## einsum: the letter rules (worked out from mistakes)

**Mental model: each letter is a `for` loop variable.**

```python
# 'i d, j d -> i j'
for i in range(n):
    for j in range(n):
        total = 0
        for d in range(D):        # d not in output => it is the summed loop
            total += X[i, d] * X[j, d]
        out[i, j] = total
```

Three distinct letters ⇒ three distinct loop variables. Everything else follows.

### Rule 1 — a letter is an *index*, not an axis name

Same letter = same index value at the same moment. `n` does not mean "the row axis," it means "row number `n`."

### Rule 2 — labels describe the array you actually pass in

`einsum(X, X, 'n d, d n -> ...')` with `X` of shape `(n,d)` **asserts the second operand is `(d,n)`**. It isn't.

```
operands could not be broadcast together with remapped shapes
(4,3)->(4,3)  (4,3)->(3,4)
```

Dangerous: if $X$ were square this would not error, it would silently compute the wrong thing.

### Rule 3 — repeated letter *within one* index list = diagonal

Each comma-separated group (and the output) is **one tensor's index list**. Repeating a letter inside a single
list means both indices are forced equal, so you only visit the diagonal:

```python
Y = [[0,1,2],[3,4,5],[6,7,8]]
np.einsum('nn->n', Y)   # [0, 4, 8]   the diagonal
np.einsum('nn->',  Y)   # 12          the trace
```

So `-> n n` is not a request for an $n \times n$ matrix. With only one loop variable `n` you can only write
`out[n, n]` — the diagonal. NumPy rejects it outright in the output position:

```
ValueError: einstein sum subscripts string includes output subscript 'n' multiple times
```

**Two different axes need two different letters, even when they have the same length.**

### Rule 4 — repeated letter *across* index lists is the whole point

That is how axes get matched up. `d` in `'i d, j d'` appears **once per operand**, in two different lists — that
is alignment, not a diagonal. What happens next depends on the output:

| Letter appears in | Meaning |
|---|---|
| inputs only (not output) | **contracted** — summed over |
| inputs **and** output | **kept** — aligned elementwise / batched |
| one input and output | kept — carried straight through |

### The contrast that caused the confusion

```
'i d, j d -> i j'
     ^     ^          d: once in list 1, once in list 2   -> ALIGN, then sum   OK
'n d, d n -> n n'
               ^ ^    n: twice in the SAME list (output)  -> DIAGONAL          ILLEGAL
```

### Checklist before submitting any einsum string

1. Does each operand's label have exactly as many letters as that array has axes?
2. Do the letters match the array's **real** shape, in order? (No mental transposes.)
3. Within any single list, is every letter distinct? (Unless you *want* a diagonal.)
4. Are the letters you want summed absent from the output?
