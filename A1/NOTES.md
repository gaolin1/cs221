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
| 1d  Einstein summation strings  | Written | 2 | ☑ **done** — verified vs NumPy |
| 1e  `linear_project`            | Coding  | 3 | ☑ **done** — 1e-0-basic passes |
| 1f  `split_last_dim_pattern`    | Coding  | 3 | ☑ **done** — 1f-0-basic passes |
| 1g  `normalized_inner_products` | Coding  | 3 | ☑ **done** — 1g-0-basic passes |
| 1h  `mask_strictly_upper`       | Coding  | 3 | ☑ **done** — 1h-0-basic passes |
| 1i  `prob_weighted_sum_einsum`  | Coding  | 3 | ☑ **done** — 1i-0-basic passes |
| 2a  Gradient warmup             | Written | 2 | ☐ |
| 2b  `gradient_warmup`           | Coding  | 3 | ☐ |
| 2c  Matmul gradient by hand     | Written | 3 | ☐ |
| 2d  `matrix_grad`               | Coding  | 3 | ☐ |
| 2e  `lsq_grad` / `lsq_finite_diff_grad` | Coding | 3 | ☐ |
| 3a  Optimization warmup         | Written | 3 | ☐ |
| 3b  Gradient descent tutor session | Written | 1 | ☐ |
| 3c  `gradient_descent_quadratic`| Coding  | 3 | ☐ |
| 4a–4d  Ethical issue spotting   | Written | 4 | ☐ |
| 5a–5d  Product ethics basics    | Written | 4 | ☐ |

**Total 50 pts = 27 coding + 23 written.** Every coding part is two tests: `X-0-basic` (1.5, local) + `X-1-hidden` (1.5, remote only — stripped from your `grader.py`, see the empty `BEGIN_HIDE`/`END_HIDE` blocks). So a local **13.5/13.5 is only half** the coding marks; the hidden half rewards code that generalises past the one visible case.

### Blockers
- ~~Starter code missing.~~ **Resolved.** Real handout is cloned at `XCS221-A1/`. Work in
  `XCS221-A1/src/submission.py`; run the autograder from inside `XCS221-A1/src/`:

  ```
  python grader.py              # all nine coding tests
  python grader.py 1e-0-basic   # one test — needs the FULL three-part id, not `1e-0`
  ```

  The `(medora_chat_build)` env already has numpy 2.1.3 + einops 0.8.1, so no conda setup needed.
  (`A1/submission.py` is a leftover empty stub — delete it so there is only one submission file.)

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

### (iii) $\operatorname{diag}(X^\top X)$ — `(n,d), (n,d) -> (d,)` ✅

```python
einsum(X, X, 'n d, n d -> d')
```

**Justification:** The $k$-th diagonal entry of $X^\top X$ is the squared norm of column $k$:
$\operatorname{diag}(X^\top X)_d = \sum_n X_{nd}X_{nd} = \sum_n X_{nd}^2$. Both operands are the plain array
$X$, so both are labeled `n d`. The row axis `n` is absent from the output, so it is summed; the feature axis `d`
appears in both inputs **and** the output, so it is kept and aligned elementwise. Net effect: square $X$
elementwise, then sum down the rows.

**Watch the `diag` carefully.** $X^\top X$ itself is $(d \times d)$ — for $X$ of shape $(3,2)$ that is
$(2,3)@(3,2) = (2,2)$. It is `diag` that collapses it to a length-$d$ **vector**, shape `(2,)`:

```
X = [[1,2],      X.T @ X  = [[35, 44],   (2,2)
     [3,4],                  [44, 56]]
     [5,6]]      np.diag(.) = [35, 56]   (2,)   <- what (iii) asks for
                 col0=(1,3,5) -> 1+9+25 = 35
```

**Why einsum is the better route here (ties back to 1b):** `np.diag(X.T @ X)` builds the whole $d\times d$
matrix and then throws the off-diagonal away — $O(nd^2)$ work for $d$ useful numbers. The einsum string never
forms the matrix; it visits only the pairs it needs.

| Route | Cost |
|---|---|
| `np.diag(X.T @ X)` | $O(nd^2)$ |
| `einsum('nd,nd->d', X, X)` | $O(nd)$ |

Read it off the letters exactly like 1b: `'n d, d p -> n p'` has 3 distinct letters -> 3 loops -> $O(ndp)$.
`'n d, n d -> d'` has only **2** distinct letters -> 2 loops -> $O(nd)$. For $d=768$ that is ~768x less work.
**The number of distinct letters in an einsum string is its complexity.**

Note the contrast with (ii) — same two arrays, opposite axis roles:

```
'i d, j d -> i j'    sum over d, keep rows     -> (n,n)  pairwise row dot products
'n d, n d -> d'      sum over n, keep columns  -> (d,)   column squared norms
```

### Verification

```python
X = rng.normal(size=(5,3)); w = rng.normal(size=3)
np.allclose(np.einsum('nd,d->n',   X, w), X @ w)              # True
np.allclose(np.einsum('id,jd->ij', X, X), X @ X.T)            # True
np.allclose(np.einsum('nd,nd->d',  X, X), np.diag(X.T @ X))   # True
```

---

## einsum in one paragraph

> Every axis of every input must be labelled, and each label's size is pinned by the array it sits on. **The way
> you label is what builds the intermediate ("big") array**: a fresh name adds an axis, a reused name merges two
> axes into one so they line up. Then **omitting a letter from the output is the signal to sum over it.**

The lever is **name reuse** — it is how you shrink the big array, and the big array's size is the cost. That is
the whole difference between `'n d, n e -> d e'` at $O(nd^2)$ and `'n d, n d -> d'` at $O(nd)$: a smaller block,
chosen by reusing a name.

| | Forced | Your choice |
|---|---|---|
| labels | every axis gets one; its size comes from the array | **which names you reuse** -> builds the big array |
| output | — | **what survives** -> the rest is summed; **the order** -> the layout |

## einsum: the one mechanism

Everything else in this file is a **consequence** of this. There are no cases and no special modes.

### The machine

Picture a control panel:

- **A row of dials** — one per distinct letter in the string. Each counts `0,1,2,...` up to that letter's size.
- **A grid of boxes** — the output. The **output letters** are wired to the box selector: they decide which box
  you are pointing at right now.
- **The inputs** are lookup tables. Each reads itself using *its own* letters' current dial settings.

Then run this, and only this:

```
turn every dial through every possible combination:
    1. read one number from each input, at its letters' current dial settings
    2. multiply those numbers together
    3. drop the result in the box addressed by the output dials, READ IN THE ORDER WRITTEN
```

**The one underlying rule: the same letter is the same dial.**

### The same thing stated as a formula (the "big array" view)

Equivalent to the dial machine, and often easier to reason about:

1. **Build the big array** `P`, indexed by **every distinct letter** in the string. Each entry is the product of
   one element from each input, read at its own letters.
2. **Sum `P` over every letter NOT in the output.**
3. **Arrange the surviving axes** in the order written after the arrow.

Careful with the wording: the output names the axes that **survive**; you collapse everything else.

```
X = [[1,2],[3,4]]

'i k, k j -> i j'   letters i,k,j   P is (2,2,2)   collapse k -> [[7,10],[15,22]]
      out[0,0] <- k-slice [1, 6]  = 7        the untotaled products ARE the big array
      out[0,1] <- k-slice [2, 8]  = 10
      out[1,0] <- k-slice [3,12]  = 15
      out[1,1] <- k-slice [6,16]  = 22

'i d, j d -> i j'   letters i,j,d   P is (2,2,2)   collapse d -> [[5,11],[11,25]]
'n d, n d -> d'     letters n,d     P is (2,2)     collapse n -> [10,20]
```

**The big array has exactly one entry per dial combination** — the dials are its indices. Same model, two views.

**Its size is the complexity.** Counting distinct letters was always counting the big array's dimensions; its
entry count is the product of all letter sizes, one multiply each. 3 letters -> `2*2*2 = 8`; 2 letters ->
`2*2 = 4`.

### Careful: "diagonal" happens on the BIG ARRAY, not on the output

Writing `d` in both operands does **not** extract a diagonal from the answer. It restricts **which points exist
in the big array**, before any summing.

```
'n d, n e -> d e'   letters n,d,e   8 points   of which 4 have d==e:  5, 12, 21, 32
'n d, n d -> d'     letters n,d     4 points   exactly those:         5, 12, 21, 32
```

The 2-letter big array *is* the `d==e` slice of the 3-letter one.

**Two different letters, two independent facts** — they do not compete:

| Letter | Placement | Effect |
|---|---|---|
| `d` | in both inputs **and** the output | keep only points where both inputs' `d` agree -> diagonal slice |
| `n` | absent from the output | those points pile up -> summed |

So the result is *the diagonal slice, then summed along n*. "It sums over the rows" and "it is a diagonal" are
both true, about different letters.

**Why it equals `diag(X^T C)`** — slicing on `(d,e)` and summing on `n` touch different axes, so they commute:

```
sum over n THEN diagonal:  M = [[26,30],[38,44]] -> diag = [26,44]
diagonal THEN sum over n:  [5,12,21,32]          -> sums = [26,44]
```

Same answer, but route 2 visits 4 points instead of 8. **That is the whole saving** — off-diagonal points are
excluded before being paid for, not computed and discarded.

### Two distinct diagonal mechanisms — do not conflate

| String | Mechanism | Diagonal of what |
|---|---|---|
| `'i i -> i'` | one letter twice in **one** operand; reads `Y[i][i]` via a stride sum | **that array** |
| `'n d, n d -> d'` | one letter shared across **two** operands, kept in the output | the **joint big array** — neither `X` nor `C` has a diagonal taken |

In part (iii) it is the second. Nothing diagonal happens to `X` itself. The output is a length-`d` vector that
*equals* the diagonal of a matrix never built, because the off-diagonal points were excluded up front.

### Two different things get called "intermediate"

| | What it is | Materialized? |
|---|---|---|
| the big array `P` | conceptual product-of-everything, indexed by all letters | **never** — NumPy accumulates into the output as it goes |
| `X.T @ X` in `np.diag(X.T @ X)` | a real `(d,d)` array in memory | **yes** — 18 MB at n=2000, d=1500 |

This explains the 31x benchmark in one line — compare the *big arrays*:

```
einsum('n d, n d -> d')          letters n,d     -> n*d   entries
np.diag(X.T @ X)
   step 1 is 'n d, n e -> d e'   letters n,d,e   -> n*d^2 entries   <- an extra letter
   step 2 discards all but d of them
```

Same three steps both routes; one just has a bigger big array.

### One-sentence summary

> Multiply everything against everything into one array indexed by all the letters. Sum away the letters you
> did not name. Lay out what is left in the order you wrote it.

### Everything else falls out of step 3

| What it looks like | What is actually happening |
|---|---|
| "the missing letter gets summed" | that dial is not wired to the box, so turning it does not move the box — every product it makes lands in the **same box** and piles up. Piling up in one box *is* addition. Nothing decided to sum. |
| "shared letters move in lockstep" | they **are** one dial. It cannot point at two values. |
| "different letters give all combinations" | separate dials, turned independently |
| "a repeated letter in one input = diagonal" | that input reads position `[dial][dial]` — same dial twice, so only the diagonal is reachable |
| the cost | = how many dial combinations exist = product of the letter sizes |

### Proof: one machine, two strings, X = [[1,2],[3,4]]

```
'i d, j d -> i j'   3 dials -> 8 turns, boxes addressed by [i,j]
   d=0 i=0 j=0   X[0,0]=1 x Y[0,0]=1  ->  1  -> box [0,0]
   d=1 i=0 j=0   X[0,1]=2 x Y[0,1]=2  ->  4  -> box [0,0]     box[0,0] = 1+4 = 5
   ...                                                        d not in address -> pile up

'n d, n d -> d'     2 dials -> 4 turns, boxes addressed by [d]
   d=0 n=0       X[0,0]=1 x Y[0,0]=1  ->  1  -> box [0]
   d=0 n=1       X[1,0]=3 x Y[1,0]=3  ->  9  -> box [0]       box[0]   = 1+9 = 10
                                                              n not in address -> pile up
```

Same procedure both times. The **only** difference is how many dials exist and which are wired to the box.

Note what gets *read*: run 1 reads `X[0,0] x Y[1,0]` — different cells, because `i` and `j` are separate dials.
Run 2 reads `X[0,0] x Y[0,0]` — the same cell, because both inputs carry the same letters. That is all
"lockstep" ever meant.

And the cost is literally the number of rows in that trace: 8 vs 4. Three dials vs two.

### Why `-> i j` and `-> j i` differ (addressing vs execution)

The output letters are a **filing rule**, not a computation instruction. They say how to assemble the address
tuple from the current dial readings:

```
-> i j   address = (dial_i, dial_j)
-> j i   address = (dial_j, dial_i)
```

Every product and every pile-up is identical either way — only the slot changes, which is why the two results
are transposes rather than different numbers:

```
i=0 j=1 d=0   X[0,0]*C[1,0]= 7    -> i j files at [0][1]    -> j i files at [1][0]
i=0 j=1 d=1   X[0,1]*C[1,1]=16      (both accumulate to 23, in different slots)
```

Pure relabeling, no arithmetic at all, is visible with one operand: `einsum('i j -> j i', X) == X.T`.

**Spreadsheet analogy.** The sequence you fill cells in (execution order) does not change the finished sheet.
Deciding *stores are rows, currencies are columns* (addressing order) does. Both are "orders"; only the second
is a decision.

### What NumPy actually does

```
'i j -> j i'   strides (24,8) -> (8,24)   shares memory with X: True
'i i -> i'     stride 32 = 24 + 8         shares memory with Y: True
```

- A transpose is a **stride swap** — a view of the same buffer, no numbers moved.
- A diagonal is a **stride sum** — add the two axes' strides and you step down the diagonal. "Same letter twice
  = diagonal" is not a rule being checked; it is what happens when one dial advances two axes at once.

Pipeline: parse and validate repeated-letter sizes -> build a strided view per operand (repeated letter
collapses two axes by adding strides) -> broadcast over the union of letters -> sum-of-products over axes
absent from the output -> lay out the result in the order the output letters were written. With
`optimize=True` and 3+ operands it first picks a pairwise contraction path and often dispatches to BLAS.

### The three "orders" — only one is a choice

**1. The order the dials are turned — not a thing you control.** Addition does not care what order products
land in a box, so any loop order gives the same answer. NumPy picks one for memory speed.

```
dial order (i,j,d) -> {(0,0):17, (0,1):23, (1,0):39, (1,1):53}
dial order (d,j,i) -> identical
```

**2. The order of letters AFTER the arrow — your choice, sets the layout.**

```
'i d, j d -> i j' -> [[17,23],      'i d, j d -> j i' -> [[17,39],
                      [39,53]]                            [23,53]]     transposes
```

**3. The order of letters INSIDE each input label — dictated by the array.** Two halves:

*Names are arbitrary* — renaming changes nothing:

```
'n d, n d -> d' -> [17,29,45]
'a b, a b -> b' -> [17,29,45]    identical
```

*Positions bind to axes* — position 1 -> axis 0, position 2 -> axis 1. Swapping them changes the answer:

```
Y = [[1,2,3],    'n d, n d -> d' -> [17,29,45]   kept position 2 -> COLUMN norms
     [4,5,6]]    'd n, d n -> d' -> [14,77]      kept position 1 -> ROW norms
```

**Slots are fixed by the array; names are free.** In `'i d, j d'` you did not choose to put `d` second in
both — X's shape forced it. The only decision was to *reuse* the letter `d` there instead of inventing a new
one, and that reuse is what makes the two arrays meet on their column axis.

So writing a string is exactly two decisions:

1. **Which letters do I reuse?** -> what lines up, and what gets totaled away
2. **What order do I write the survivors?** -> the layout of the answer

Everything else is forced by the data or irrelevant.

### Reading any string in 10 seconds

1. **Count the distinct letters** -> number of dials -> the complexity.
2. **Output letters** -> the shape of the box grid.
3. **Letters missing from the output** -> their products pile into one box, so they are totaled away.

### Writing one — the same three questions, inverted

1. *What do I want one number for?* -> those are the output letters
2. *What do I want totaled away?* -> leave those letters out of the output
3. *What must line up between the inputs?* -> give those the same letter

### Plain-English version (the spreadsheet story)

Two ordinary spreadsheets: units sold by **store x product**, and price by **product x currency**. Want money
by **store x currency**.

```
store product, product currency -> store currency
```

- `product` is written in **both** sheets -> apple-units must meet apple-price. You cannot pair apple-units
  with banana-price.
- `product` is **missing** from the answer -> you walk every product and total it up. It gets used up.
- `store` and `currency` are **in** the answer -> a number for each combination.

Want the per-product breakdown instead? Keep the letter: `store product, product currency -> store product
currency` — now nothing is totaled. **The letters you keep are the level of detail you want; anything you leave
out gets totaled away.** It is a pivot table: output letters = group by, dropped letters = sum.

### Checklist before submitting any einsum string

1. Does each operand's label have exactly as many letters as that array has axes?
2. Do the letters match the array's **real** shape, in order? (No mental transposes.)
3. Within any single list, is every letter distinct? (Unless you *want* a diagonal.)
4. Are the letters you want summed absent from the output?

---

## 1e — Batch linear projection ✅

**Task:** `x:(B,D_in)`, `W:(D_in,D_out)`, `b:(D_out,)` → `y:(B,D_out)` with `y[i] = x[i] W + b`.
einops `einsum` for the matmul, broadcasting for the bias, no loops, no `@`.

```python
x_with_weights = einsum(x, W, 'batch d_in, d_in d_out -> batch d_out')
return x_with_weights + b
```

**Deriving the pattern.** Same read-off-the-letters procedure as 1d. For each axis ask only one question:
does it survive to the output?

| Axis | Appears in | In output? | Role |
|---|---|---|---|
| `batch` | `x` only | yes | carried — one output row per input row |
| `d_in` | `x` **and** `W` | no | **contracted** — summed over |
| `d_out` | `W` only | yes | carried — one output column per weight column |

Shared by both operands **and** absent from the output ⇒ summed. That gives
$\sum_{d_{in}} x[b,d_{in}]\,W[d_{in},d_{out}]$, the definition of matmul. The table *is* the string.

Complexity check (ties back to 1b): 3 distinct letters → 3 loops → $O(B \cdot D_{in} \cdot D_{out})$. ✓

**Use full words for axis names.** einops accepts any valid identifier, so
`'batch d_in, d_in d_out -> batch d_out'` reads like the docstring. `'b i, i o -> b o'` is the same
computation but has to be decoded.

### Two gotchas

**1. einops argument order is the reverse of NumPy's.**

```python
np.einsum('n d, d -> n', X, w)      # string FIRST
einsum(X, w, 'n d, d -> n')         # einops: tensors first, string LAST
```

**2. The bias needs no reshape.** NumPy right-aligns shapes when broadcasting:

```
y      (B, D_out)
b         (D_out,)   -> read as (1, D_out), stretched down all B rows
           ^^^^^^^  trailing axes match, so it is legal
```

Plain `+ b` works — no `b[None, :]`, no `reshape`. It broadcasts precisely *because* the axis that fails to
line up (`batch`) is the one `b` is missing entirely, so NumPy fills it in.

**Do not reach for `np.array` here.** It is a *constructor* (build an array from a list, or copy an existing
one), not an addition operator. `einsum` already returns an ndarray, and `+` on two ndarrays is already
elementwise addition with broadcasting. General rule: in NumPy you rarely build arrays in order to combine
arrays — if you are reaching for `np.array` mid-computation, the operation you want probably already exists as
an operator or ufunc. `np.array` is for *getting into* NumPy from Python lists; once there you stay there with
`+`, `*`, `@`, `einsum`.

**Verified** beyond the single grader case: shapes `(B,Din,Dout)` = `(7,5,3)`, `(1,4,4)`, `(20,1,6)`, `(3,8,1)`
all match `x @ W + b`, including the degenerate `Din=1` and `Dout=1` edges.

---

## 1f — Split last dimension ✅

**Task:** return the `einops.rearrange` **pattern string** taking `(B, D)` → `(B, G, D/G)`. Return the string
only; the grader applies it with `g=num_groups`. Assume `D % G == 0`.

```python
return 'b (g d) -> b g d'
```

**The handout's `'b d -> b g (d/g)'` is not valid einops.** It is written as an "e.g." and only gestures at the
shape — einops patterns have **no division operator**. Do not type it.

### Parentheses = one composite axis

A parenthesized group is several logical axes packed into one physical axis. Which side of the `->` it sits on
sets the direction:

| Parens on the… | Meaning |
|---|---|
| **left** (input) | that one axis *is* `a*b` — **split** it apart |
| **right** (output) | **merge** those axes into one |

Here two axes become three, so the composite is on the **input** side: `D` secretly is `G × (D/G)`.
einops cannot infer both factors from `D` alone — hence the grader passing `g=num_groups` to pin one; `D/g`
follows automatically.

### The odometer rule — the whole problem

**Inside a parenthesis, the leftmost axis ticks slowest** (like odometer digits, like C-order `reshape`).

| `(g d)` ✅ | `(d g)` ❌ |
|---|---|
| flat index = `g*(D/G) + d` | flat index = `d*G + g` |
| `d` is the fast digit | `g` is the fast digit |
| group 0 owns the first **contiguous chunk** | group 0 takes every `G`-th element |

Same `x = arange(12).reshape(2,6)`, row 0 = `[0 1 2 3 4 5]`:

```
g=2:   (g d) -> [[0,1,2],[3,4,5]]      (d g) -> [[0,2,4],[1,3,5]]      both shape (2,2,3)
g=3:   (g d) -> [[0,1],[2,3],[4,5]]    (d g) -> [[0,3],[1,4],[2,5]]    both shape (2,3,2)
```

### ⚠ The trap I fell into

I first wrote `'b (d g) -> b g d'`. It produced **exactly the right shape** and silently wrong contents.

The mistake was reading the output side `b g d` as "g groups of size d" and assuming the left side just had to
*mention* both names. But **the right side only sets axis order; the parenthesis on the left sets memory
layout.** The two sides are independent, which is why a wrong left side still yields a plausible right-hand shape.

> **A matching shape is not evidence of a correct rearrange. Always check a value.**

`(d g)` is not nonsense in general — it is what you want for genuinely *interleaved* layouts. It is just not
this problem.

### Output shape, read left to right

```
(2,   2,   3)  <- x:(2,6) with g=2
 ^    ^    ^
 B    G   D/G
```

Confusing in the g=2 test only because `B` and `G` are both 2. The middle slot always tracks `g` exactly:

| `x.shape` | `g` | result |
|---|---|---|
| `(2,6)` | 2 | `(2, 2, 3)` |
| `(2,6)` | 3 | `(2, 3, 2)` |
| `(2,6)` | 6 | `(2, 6, 1)` |
| `(2,6)` | 1 | `(2, 1, 6)` |

The pattern string never changes across those rows — that is the payoff of returning a *pattern* instead of
doing the reshape.

### Duplicate axis names are a hard error

`'d (g d) -> d g d'` (reusing `d` for batch and for the inner axis) fails loudly:

```
EinopsError: Indexing expression contains duplicate dimension "d"
```

Names are how einops matches positions across the `->`, so a repeat is genuinely ambiguous. Note the contrast:
this one **errors**, while `(d g)` vs `(g d)` stays **silent** — the silent class is the dangerous one.

---

## 1g — Normalized inner products ✅

**Task:** `A:(B,M,D)`, `C:(B,N,D)` → `S:(B,M,N)` with `S[b,i,j] = <A[b,i,:], C[b,j,:]>`; divide by `sqrt(D)`
when `normalize=True`. This is 1d(ii) with a batch axis added — it is literally an attention score matrix.

```python
a_inner_c = einsum(A, C, '... m d, ... n d -> ... m n')
if normalize:
    a_inner_c = a_inner_c / np.sqrt(A.shape[-1])
return a_inner_c
```

**`...` (ellipsis) beats naming the batch axis.** It matches *any* number of leading axes, so the same string
handles `(B,M,D)` and `(B1,B2,M,D)`. An explicit `b` only handles one. Worth preferring for the hidden test.

### The third axis role — the one 1e did not have

| In both operands? | In output? | einsum does |
|---|---|---|
| no (one operand only) | yes | carry through |
| **yes** | **no** | **sum over it** (1e's `d_in`, here `d`) |
| **yes** | **yes** | **align elementwise, no sum** ← the batch case |

That last row *is* the batch loop. Naming an axis identically in both operands **and** the output means "match
these up, do not contract" — no `for b in range(B)` needed. Already used in 1d(iii)'s `'n d, n d -> d'`.

**`m` and `n` must get different names.** Same lesson as 1d(ii): they are different rows, you want *all pairs*,
so both indices range independently and both survive. Reusing one name asks for a diagonal instead.

4 distinct letters, 3 in the output ⇒ exactly one contracted (`d`). Cost O(B·M·N·D).

### ⚠ The trap I fell into: `1/np.sqrt(S)`

I wrote `1/np.sqrt(a_inner_c)` — sqrt of the **scores**, then reciprocal. Wrong on three counts:

1. The sqrt applies to **`D`, an integer** (`A.shape[-1]`), not to the array.
2. `S` belongs in the **numerator**: `S / sqrt(D)`, not `1 / sqrt(S)`.
3. Dot products **go negative** ⇒ `np.sqrt` of them yields `nan` (and `sqrt(0)` → `1/0` → `inf`).

```
raw S               = [[-0.961, 0.458, -0.147], ...]
1/np.sqrt(S)  WRONG = [[nan,    1.477, nan   ], ...]
S/np.sqrt(D)  RIGHT = [[-0.392, 0.187, -0.060], ...]     D=6, sqrt(D)=2.4495
```

**Tell:** `D` appeared nowhere in my expression, though the problem statement names it explicitly. If a quantity
the problem mentions never shows up in the code, something is wrong.

**Two different things both called "normalize":**

| | divisor depends on | per-element? |
|---|---|---|
| **this problem** | only the **shape** (`D`) | no — one constant for the whole matrix |
| `x/norm(x)`, z-scoring | the **values** | yes — differs per row |

Rescaling by one constant preserves the relative pattern (biggest stays biggest, signs intact). A per-element
`sqrt` is a nonlinear distortion — a completely different operation.

**Why sqrt(D) and not D:** a dot product sums `D` random-ish terms, so its *standard deviation* grows like
sqrt(D). Dividing it out keeps the spread of `S` roughly constant as `D` changes. This is exactly attention's
1/sqrt(d_k).

**Style:** write `S = S / ...`, not a bare `S / ...` — the latter computes and discards.

---

## 1h — Mask strictly upper triangle ✅

**Task:** set entries with column > row to `-inf` in `scores:(B,L,L)`. Broadcasting, no loops, float output.

```python
positions = np.arange(scores.shape[-1])
masked_scores = np.where(positions[:, None] >= positions, scores, -np.inf)
return masked_scores
```

### Why `tril`/`triu` cannot do it

They have **no fill-value parameter** — the signature is `np.triu(m, k=0)`, where `k` is the *diagonal offset*.
Zero is baked into what they are. Not a doc-reading failure; the feature does not exist. (`-np.inf * 0` is
`nan`, so no arithmetic trick either.)

Two distinct uses, only one of which is a dead end:

| | verdict |
|---|---|
| `np.tril(scores)` — on the **data** | ✗ masked cells become `0`, indistinguishable from *real* zeros already in the data. Information destroyed. |
| `np.tril(np.ones((L,L), bool))` — on a **ones array** | ✓ genuinely works, gives a real mask (`np.tri(L, dtype=bool)` is the one-call version) |

The second **would pass the grader**. Index grids are worth writing anyway: the problem asks for them, and they
generalise where triangles do not — `np.abs(i-j) <= 2` (band), `i-j > k` (sliding-window attention) have no
triangle function at all. The real lesson is *build indices, compare them, `np.where` the result*.

### The rule that unlocked it

> **A positional mask depends only on *where* a cell is, never on *what is in it*.**
> ⇒ `scores` appears **only** as a value argument to `np.where` — **never inside the condition**.

I broke this twice: `np.where(scores < positions, ...)` then `np.where(scores[None,:] < positions, ...)`. Both
compare **data against indices**, which is meaningless. The first produced an all-`False` mask, so *everything*
became `-inf` — that was the giveaway.

### Only ONE `None` is needed

Automatic broadcast padding adds axes **on the left only**. So `(L,)` auto-pads to `(1,L)` — the *column*
orientation, free. There is **no** automatic route to `(L,1)`; that one must be explicit.

| | shape | role |
|---|---|---|
| `positions` | `(L,)` → auto `(1,L)` | column index `j` |
| `positions[:, None]` | `(L,1)` | row index `i` |

**A scalar has no orientation to get wrong; a vector does.** That is why 1g's `np.sqrt(D)` needed nothing —
shape `()` pads to all-1s and stretches every direction at once.

### Choosing the comparison

`np.where(cond, value_if_true, value_if_false)` — condition and argument order must agree:

```
np.where(<keep condition>, scores, -np.inf)     # true on survivors   <- what I used
np.where(<mask condition>, -np.inf, scores)     # true on victims
```

*Strictly* upper ⇒ the diagonal **survives**. Keep-set is the lower triangle **including** the diagonal ⇒
`i >= j`. Print the four candidates on `L=4` and eyeball which has the diagonal on the right side:

```
i >  j         i >= j  ✅      j >  i         j >= i
[0 0 0 0]     [1 0 0 0]     [0 1 1 1]      [1 1 1 1]
[1 0 0 0]     [1 1 0 0]     [0 0 1 1]      [0 1 1 1]
[1 1 0 0]     [1 1 1 0]     [0 0 0 1]      [0 0 1 1]
[1 1 1 0]     [1 1 1 1]     [0 0 0 0]      [0 0 0 1]
```

### The batch axis handles itself

```
scores  (B, L, L)
mask       (L, L)   -> pads to (1, L, L)
result  (B, L, L)   same mask, every batch element
```

No slicing, no loop over `b`.

### `np.where` broadcasts all THREE arguments

`cond (L,L)`, `scores (B,L,L)`, `-np.inf ()` → common shape `(B,L,L)`, then elementwise **selection**.

**It selects; it does not apply a function — and it does NOT short-circuit.** Both branches are fully evaluated
before the selection happens:

```python
np.where(x != 0, 1/x, 0)   # 1/x computed for ALL x including zeros -> divide-by-zero warning anyway
```

### `-inf` spellings, and why it must be `-inf`

`-np.inf` == `float('-inf')` == `-math.inf`. `np.where` **auto-promotes** an int array to float when a branch is
`-np.inf`, satisfying the docstring float requirement for free (in-place `scores[mask] = -np.inf` on an int
array would *not*). Why not a big negative sentinel: `np.exp(-inf)` is **exactly** `0.0`, so masked entries get
exactly zero softmax probability; `-1e9` is only close, and misbehaves in float32.

### `np.arange` refresher

NumPy's `range` ("array range"). Stop is exclusive, so `np.arange(L)` = `0..L-1` — exactly the valid indices of
a length-`L` axis. Unlike `range` it is a **real array**: has `.shape`, does arithmetic, accepts `[:, None]`,
and allows floats.

### `[:, None]` refresher — axis, not data

The comma separates **axes**; one slot per axis. `:` = keep this whole axis; `None` (= `np.newaxis`) = insert a
new length-1 axis here.

**Adding an axis is not adding a row.** `.size` is the tell:

| | shape | size | data |
|---|---|---|---|
| `s[None, :]` — new **axis** | `(2,3)`→`(1,2,3)` | 6 → **6** | unchanged, *shares memory* |
| `np.vstack` — new **row** | `(2,3)`→`(3,3)` | 6 → **9** | 3 genuinely new numbers |

Same numbers, one extra `[` wrapped around the outside. Also spelled `p.reshape(-1, 1)`.

---

## 1i — Probability-weighted sum ✅

**Task:** return the einsum **string** for `P:(B,N)`, `V:(B,N,D)` → `out:(B,D)` with
`out[b,:] = sum_j P[b,j] * V[b,j,:]`.

```python
return 'batch weights, batch weights data -> batch data'
```

Same three-way classification as 1g:

| Axis | In `P` | In `V` | In output | Role |
|---|---|---|---|---|
| `batch` | ✓ | ✓ | ✓ | aligned — the batch |
| `weights` (`N`) | ✓ | ✓ | ✗ | **contracted** — this *is* the sum over j |
| `data` (`D`) | ✗ | ✓ | ✓ | carried |

The `weights` row is the whole problem: the `sum_j` in the formula *is* "shared by both operands, absent from
the output." Nothing else to decide. Descriptive names make the contraction self-evident — `weights` appears
twice on the left and never on the right, so it is visibly the axis being summed away.

### ⚠ Handout error: it says "numpy.einsum", the grader uses einops

The LaTeX says *"provide only the `numpy.einsum` string"*, but `grader.py` calls `einsum(P, V, pattern)` from
**einops**. The `submission.py` docstring is the correct one. The dialects are **not** interchangeable:

| | separator | `'bn'` means |
|---|---|---|
| **einops** | whitespace-separated names | ONE axis literally named `bn` |
| **numpy** | one char per axis, no spaces | axis `b` then axis `n` |

```
einsum(P, V, 'bn,bnd->bd')     -> EinopsError: Unknown axis bd on right side
np.einsum('bn,bnd->bd', P, V)  -> works
```

Always write these einops-style with spaces, as in 1e and 1g.

### The arc

1g gives attention **scores**, a softmax over them gives `P`, and 1i is the **weighted sum of values** by `P`.
Problems 1g + 1h + 1i are the whole attention mechanism.

---

## Local test harness — `src/my_tests.py`

Written because **the hidden half of every coding part is stripped from the local grader** (nine empty
`BEGIN_HIDE`/`END_HIDE` pairs; the `Makefile` shows the `sed` that deletes them). There is nothing to recover
locally, and hunting for leaked copies is an honor-code problem — so the substitute is to write the
generalisation tests myself.

```
python my_tests.py          # everything
python my_tests.py 1g       # one part
```

**Principle: compare against an *independent reference* on randomised shapes.** Every reference is computed a
*different way* than the submission computes it — `@`, `np.einsum`, `np.triu_indices`, numerical
differentiation, `np.linalg.lstsq` — so a shared misconception cannot make both sides agree.

Coverage the shipped grader does not have:

| Part | Extra |
|---|---|
| 1e | six shape combos incl. `Din=1`, `Dout=1`, `1x1x1`; bias isolated via `W=0` |
| 1f | `g` in {1,2,3,6}, `D` in {6,8,12}; explicit **contiguous-not-strided** value check |
| 1g | `M != N`, `D=1`, the `normalize` default, **negative dot products stay finite** |
| 1h | `L=1`, `L=2`; **diagonal survives**; int→float promotion; **input not mutated** |
| 1i | one-hot `P` must select exactly one value vector |
| 2b, 2d, 2e | cross-checked against **finite differences** — independent of the analytic formula |
| 2e | gradient must vanish at the true least-squares minimiser |
| 3c | `num_steps=0`; the minimiser is a fixed point; one hand-computed step |

Unimplemented parts report `skip`, so the file is usable from the start and the skips flip to passes as work
lands. It lives in `src/` but is **not** submitted — only `submission.py` is uploaded.

> The 1f bug is the argument for this file: `'b (d g) -> b g d'` produced the **right shape** and the **wrong
> values**, and the single shipped test would have caught it only by luck.
