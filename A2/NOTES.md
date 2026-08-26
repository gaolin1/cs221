# XCS221 Assignment 2 — Sentiment Classification

Due **Sunday, September 6, 11:59pm PT**. 50 points: **17.5 written + 32.5 coding**.

Dataset: tweets labelled with one of three emotions — joy / anger / fear (one-hot).

---

## Where the points are

| Problem | Type | Pts | What it is |
|---|---|---|---|
| **1** Bag-of-words & linear classification | Written | **5.5** | 1a features 0.5 · 1b softmax 1 · 1c cross-entropy 1 · **1d gradient 3** |
| **2** Embeddings & MLP | Written | **9.5** | 2a averaging 1 · 2b forward pass 2 · **2c backprop 6** · 2d analysis 0.5 |
| **3** Linear classifier | Coding | **17.5** | 3a–3f 2 each · **3g train 5** · 3h writeup 0.5 |
| **4** MLP + embeddings | Coding | **15** | 4a–4e 2 each · **4f train 5** |
| **5** Product ethics (continues from A1) | Written | **2.5** | 5a–5e 0.5 each |

**Two ideas carry ~30 of the 50 points.** Everything else is plumbing around them.

---

## The single highest-leverage fact to get from the videos

$$\frac{\partial L_{CE}}{\partial z_k} = p_k - y_k$$

The gradient of cross-entropy loss with respect to the **logits** is just *prediction minus target*.

That one result is:
- **1d** outright (3 pts) — it asks you to derive exactly this
- the core of **3e** `numpy_compute_gradients` (2 pts)
- what makes **3g** `train_linear_classifier` work (5 pts)

**~10 points hinge on this one line.** When it appears in lecture, stop and make sure you can reproduce the
derivation, not just the answer. The chain rule goes through the softmax, and the messy $\partial p_j/\partial z_k$
terms collapse — that collapse is the thing to watch.

Same pattern shows up again in problem 2 with **sigmoid + binary cross-entropy**: $\partial L/\partial z = \hat y - y$.
Same structure, same collapse. If you see them as one fact rather than two, 2c gets much easier.

---

## What to actively watch for, by topic

### Softmax (1b, 3c)
- Why exponentiate at all — turning arbitrary reals into positive numbers that sum to 1.
- **Numerical stability: subtract the max before exponentiating.** `exp(1000)` overflows. Lectures may gloss
  this; the coding part 3c needs it. Watch for `softmax(z) == softmax(z - max(z))`.

### Cross-entropy loss (1c, 3d)
- Why it is $-\log p_{\text{correct class}}$ — the one-hot $y$ zeroes out every other term.
- **Behaviour at the limits**: $p \to 1$ gives loss $\to 0$; $p \to 0$ gives loss $\to \infty$. 1c asks for
  exactly this explanation.
- The `epsilon` argument in 3d exists to stop $\log(0)$.

### The gradient (1d, 3e) — see above. The main event.

### Backpropagation (2c — 6 pts, biggest single item)
2c hands you a table and says *"express each partial in terms of partials above it"*. That is a direct
instruction about **what to watch for**: how a lecture sets up the backward chain, in order:

```
dL/dy_hat  ->  dL/dz  ->  dL/dh  ->  dL/dW(1), dL/db(1)
```

Specifically watch for:
- **the weight gradient pattern**: (gradient arriving from above) × (input to that layer)ᵀ — it is the same
  shape rule every time
- **the bias gradient**: just the gradient arriving from above (bias adds to everything, so it passes through)
- how gradients *flow backwards* through a layer: multiply by that layer's weights transposed

### Activation derivatives (2b, 2c, 4c)
Small but you cannot do 2c without them:
- **ReLU**: $\text{ReLU}'(x) = 1$ if $x > 0$, else $0$. A gate that passes or blocks.
- **Sigmoid**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ — expressed in terms of its own output, which is why you
  reuse the forward-pass value.

### Bag-of-words vs embeddings (1a, 2a, 2d, 3h)
The argument, not the math:
- BoW is **sparse, high-dimensional, and blind to word similarity** ("amazing" and "wonderful" are unrelated
  coordinates).
- Embeddings are **dense and place similar words near each other**.
- Averaging embeddings loses **word order** ("dog bites man" == "man bites dog") — that is the disadvantage 2a
  wants.

### Training loop (3g, 4f — 5 pts each, 10 total)
- epochs, learning rate, shuffling, when the weight update happens
- the update rule itself: `W -= lr * grad`
- how loss/accuracy is tracked per epoch

---

## What already carries over from A1

Do not re-learn these — A1 built them:

| A1 | A2 use |
|---|---|
| einsum / matmul shape discipline | 3c is *"using einops and NumPy only"*; every layer is a shape-check |
| gradients of a matmul (2c/2d) | the weight-gradient pattern in backprop is the same object |
| `gradient_descent_quadratic` (3c) | the toy version of 3g/4f — same loop, real model |
| least squares $\tfrac12\|Aw-b\|^2$ | same shape of story: predict, compare to target, square/penalise, descend |

The A1 einsum work pays off directly: a forward pass is `features @ W + b`, and a weight gradient is
`features.T @ (p - y)`. Both are einsum strings you can now read off by shape.

---

## Watching order (highest value first)

1. **Softmax + cross-entropy + its gradient** — unlocks 1b, 1c, 1d, 3c, 3d, 3e, 3g (~15 pts)
2. **Backpropagation through a small MLP** — unlocks 2b, 2c, 4c–4f (~28 pts)
3. **Word embeddings, why they beat BoW** — unlocks 2a, 2d, 3h, 4a, 4b (~7 pts)
4. **SGD training loops** — the 5-pointers, 3g and 4f

Problems 1 and 2 are entirely by-hand arithmetic on a 6-tweet toy dataset with given numbers. **They are worth
15 points and need no code at all** — bankable early, and doing them first makes the coding parts obvious
because you will have already executed the algorithm manually.

---

## Problem 5 continues from A1

5a–5e are about the *same product* you chose for A1 problem 5 — user base, intended audience, market share,
misuse risks, and gaps. Whatever you picked in A1 you are committed to. Worth checking that it has a real
privacy policy, active news coverage, and enough surface area for a safety discussion.
