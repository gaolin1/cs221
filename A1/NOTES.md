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
