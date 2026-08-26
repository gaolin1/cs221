#!/usr/bin/env python3
"""
Local stress tests -- NOT part of the submission. Do not upload this file.

The grader ships one hand-picked example per part; the hidden half is stripped
out (see the empty BEGIN_HIDE/END_HIDE blocks in grader.py). This file is the
honest substitute: for each function, compare against an *independent reference*
implementation on randomised shapes and edge cases.

That is the class of bug the hidden tests exist to catch -- code that fits the
single visible example but does not generalise. It is exactly how the 1f bug was
found: 'b (d g) -> b g d' produced the right *shape* and the wrong *values*.

Run:  python my_tests.py            # all implemented parts
      python my_tests.py 1g         # just one
"""
import sys
import traceback

import numpy as np

import submission as s

RNG = np.random.default_rng(20250830)
TOL = 1e-9

_results = []


def check(part, name, fn):
    """Run one case. Skips cleanly if the function is not implemented yet."""
    try:
        fn()
    except (Stub, NotImplementedError) as e:
        _results.append((part, name, "skip", str(e) or "not implemented"))
        return
    except Exception:
        tb = traceback.format_exc()
        # A `pass`-only stub returns None; report that as a skip, not a failure.
        if "NoneType" in tb or "must be a string" in tb:
            _results.append((part, name, "skip", "returns None (stub)"))
        else:
            _results.append((part, name, "FAIL", tb.strip().split("\n")[-1]))
        return
    _results.append((part, name, "ok", ""))


class Stub(Exception):
    """The function under test has not been written yet."""


def close(a, b, tol=TOL):
    if a is None:
        raise Stub("returned None")
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} != reference {b.shape}"
    assert np.allclose(a, b, rtol=tol, atol=tol), (
        f"max abs diff {np.max(np.abs(a - b)):.3e}"
    )


# ---------------------------------------------------------------- 1e
def test_1e():
    # Reference: the @ operator, which the problem forbids in submission.py
    # but which is perfectly good as an independent check here.
    for B, Di, Do in [(7, 5, 3), (1, 4, 4), (20, 1, 6), (3, 8, 1), (1, 1, 1), (64, 33, 17)]:
        x = RNG.standard_normal((B, Di))
        W = RNG.standard_normal((Di, Do))
        b = RNG.standard_normal(Do)
        check("1e", f"B={B},Din={Di},Dout={Do}",
              lambda x=x, W=W, b=b: close(s.linear_project(x, W, b), x @ W + b))

    # Bias must actually be added, not ignored: zero weights isolate it.
    x = RNG.standard_normal((5, 3))
    W = np.zeros((3, 4))
    b = np.arange(4.0)
    check("1e", "bias isolated (W=0)",
          lambda: close(s.linear_project(x, W, b), np.broadcast_to(b, (5, 4))))


# ---------------------------------------------------------------- 1f
def test_1f():
    from einops import rearrange
    for D, g in [(6, 1), (6, 2), (6, 3), (6, 6), (8, 4), (12, 3), (1, 1)]:
        for B in (1, 5):
            x = RNG.standard_normal((B, D))
            # Reference: plain C-order reshape.
            check("1f", f"B={B},D={D},g={g}",
                  lambda x=x, B=B, D=D, g=g: close(
                      rearrange(x, s.split_last_dim_pattern(), g=g),
                      x.reshape(B, g, D // g)))

    # The bug that shape-only checking misses: values must be contiguous chunks.
    x = np.arange(12.0).reshape(2, 6)
    check("1f", "values are contiguous chunks (not strided)",
          lambda: close(rearrange(x, s.split_last_dim_pattern(), g=3)[0],
                        np.array([[0., 1.], [2., 3.], [4., 5.]])))


# ---------------------------------------------------------------- 1g
def test_1g():
    for B, M, N, D in [(1, 2, 3, 2), (4, 3, 5, 6), (2, 1, 1, 8), (3, 7, 2, 1), (1, 1, 1, 1)]:
        A = RNG.standard_normal((B, M, D))
        C = RNG.standard_normal((B, N, D))
        ref = np.einsum("bmd,bnd->bmn", A, C)   # independent reference
        check("1g", f"raw B={B},M={M},N={N},D={D}",
              lambda A=A, C=C, ref=ref: close(s.normalized_inner_products(A, C, normalize=False), ref))
        check("1g", f"normalized B={B},M={M},N={N},D={D}",
              lambda A=A, C=C, ref=ref, D=D: close(
                  s.normalized_inner_products(A, C, normalize=True), ref / np.sqrt(D)))

    # normalize defaults to True -- a missing default is a silent half-failure.
    A = RNG.standard_normal((2, 3, 4))
    C = RNG.standard_normal((2, 5, 4))
    check("1g", "normalize defaults to True",
          lambda: close(s.normalized_inner_products(A, C),
                        np.einsum("bmd,bnd->bmn", A, C) / 2.0))

    # Negative scores must survive (an elementwise sqrt would emit nan here).
    A = -np.ones((1, 2, 3)); C = np.ones((1, 2, 3))
    check("1g", "negative dot products stay finite",
          lambda: close(s.normalized_inner_products(A, C, normalize=False), -3 * np.ones((1, 2, 2))))


# ---------------------------------------------------------------- 1h
def test_1h():
    def reference(scores):
        out = scores.astype(float).copy()
        L = out.shape[-1]
        r, c = np.triu_indices(L, k=1)
        out[..., r, c] = -np.inf
        return out

    for B, L in [(1, 4), (3, 5), (2, 1), (1, 2), (4, 8)]:
        sc = RNG.standard_normal((B, L, L))
        check("1h", f"B={B},L={L}",
              lambda sc=sc: close(s.mask_strictly_upper(sc), reference(sc)))

    # Diagonal must SURVIVE -- 'strictly' upper. Off-by-one lives here.
    sc = np.ones((1, 3, 3))
    check("1h", "diagonal survives",
          lambda: close(np.diagonal(s.mask_strictly_upper(sc), axis1=-2, axis2=-1),
                        np.ones((1, 3))))

    # Integer input must come back as float (-inf has no int representation).
    si = np.arange(2 * 3 * 3).reshape(2, 3, 3)
    check("1h", "int input -> float output",
          lambda: (_ for _ in ()).throw(AssertionError("dtype not float"))
          if s.mask_strictly_upper(si).dtype.kind != "f" else None)

    # Input must not be mutated in place.
    orig = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    backup = orig.copy()
    check("1h", "input not mutated",
          lambda: (s.mask_strictly_upper(orig), close(orig, backup))[1])


# ---------------------------------------------------------------- 1i
def test_1i():
    from einops import einsum
    for B, N, D in [(1, 3, 2), (4, 5, 6), (2, 1, 3), (3, 4, 1), (1, 1, 1)]:
        P = RNG.random((B, N)); P = P / P.sum(axis=1, keepdims=True)
        V = RNG.standard_normal((B, N, D))
        check("1i", f"B={B},N={N},D={D}",
              lambda P=P, V=V: close(einsum(P, V, s.prob_weighted_sum_einsum()),
                                     np.einsum("bn,bnd->bd", P, V)))

    # A one-hot P must select exactly one value vector -- catches a wrong contraction.
    P = np.array([[0., 1., 0.]])
    V = np.array([[[1., 2.], [3., 4.], [5., 6.]]])
    check("1i", "one-hot P selects row 1",
          lambda: close(einsum(P, V, s.prob_weighted_sum_einsum()), np.array([[3., 4.]])))


# ---------------------------------------------------------------- 2b
def test_2b():
    for d in [1, 3, 10, 100]:
        w = RNG.standard_normal(d); c = RNG.standard_normal(d)
        check("2b", f"d={d}", lambda w=w, c=c: close(s.gradient_warmup(w, c), 2.0 * (w - c)))

    # Cross-check against finite differences of f(w) = sum (w-c)^2.
    w = RNG.standard_normal(5); c = RNG.standard_normal(5)
    f = lambda v: np.sum((v - c) ** 2)
    eps = 1e-6
    fd = np.array([(f(w + eps * e) - f(w - eps * e)) / (2 * eps) for e in np.eye(5)])
    check("2b", "matches finite differences",
          lambda: close(s.gradient_warmup(w, c), fd, tol=1e-6))


# ---------------------------------------------------------------- 2d
def test_2d():
    for m, p, n in [(2, 3, 2), (4, 3, 5), (1, 1, 1), (6, 2, 3), (3, 7, 1)]:
        A = RNG.standard_normal((m, p)); B = RNG.standard_normal((p, n))
        gA_ref = np.broadcast_to(B.sum(axis=1), (m, p))
        gB_ref = np.broadcast_to(A.sum(axis=0)[:, None], (p, n))

        def run(A=A, B=B, gA_ref=gA_ref, gB_ref=gB_ref):
            got = s.matrix_grad(A, B)
            if got is None:
                raise Stub("returned None")
            gA, gB = got
            close(gA, gA_ref); close(gB, gB_ref)
        check("2d", f"m={m},p={p},n={n}", run)

    # Cross-check gradient of s = sum(A@B) against finite differences.
    A = RNG.standard_normal((3, 4)); B = RNG.standard_normal((4, 2))
    eps = 1e-6

    def fd_check():
        got = s.matrix_grad(A, B)
        if got is None:
            raise Stub("returned None")
        gA, _ = got
        num = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                Ap = A.copy(); Ap[i, j] += eps
                Am = A.copy(); Am[i, j] -= eps
                num[i, j] = ((Ap @ B).sum() - (Am @ B).sum()) / (2 * eps)
        close(gA, num, tol=1e-6)
    check("2d", "grad_A matches finite differences", fd_check)


# ---------------------------------------------------------------- 2e
def test_2e():
    for n, d, scale in [(5, 4, 1.0), (20, 10, 1.0), (3, 8, 1.0),
                        (50, 30, 3.0), (6, 5, 0.01), (1, 1, 1.0)]:
        A = RNG.standard_normal((n, d)) * scale
        b = RNG.standard_normal(n) * scale
        w = RNG.standard_normal(d) * scale
        check("2e", f"analytic n={n},d={d},scale={scale}",
              lambda A=A, b=b, w=w: close(s.lsq_grad(w, A, b), A.T @ (A @ w - b), tol=1e-8))
        check("2e", f"fd vs analytic n={n},d={d},scale={scale}",
              lambda A=A, b=b, w=w: close(s.lsq_finite_diff_grad(w, A, b, epsilon=1e-5),
                                          A.T @ (A @ w - b), tol=1e-4))

    # At the exact minimiser the gradient must vanish.
    A = RNG.standard_normal((10, 4)); b = RNG.standard_normal(10)
    w_star = np.linalg.lstsq(A, b, rcond=None)[0]
    check("2e", "gradient vanishes at the least-squares minimiser",
          lambda: close(s.lsq_grad(w_star, A, b), np.zeros(4), tol=1e-8))


# ---------------------------------------------------------------- 3c
def test_3c():
    # With a stable step size GD converges to the weighted mean.
    for n in [2, 5, 20]:
        x = RNG.standard_normal(n) * 10
        w = RNG.random(n) + 0.1
        star = (w * x).sum() / w.sum()
        lr = 0.2 / w.sum()
        check("3c", f"converges to weighted mean n={n}",
              lambda x=x, w=w, star=star, lr=lr: close(
                  s.gradient_descent_quadratic(x, w, 100.0, lr, 800), star, tol=1e-8))

    # num_steps=0 must return theta0 untouched.
    check("3c", "num_steps=0 returns theta0",
          lambda: close(s.gradient_descent_quadratic(np.array([1., 2.]), np.array([1., 1.]),
                                                     3.0, 0.1, 0), 3.0))

    # Starting exactly at the minimiser must not move.
    x = np.array([0., 10.]); w = np.array([1., 3.])
    star = (w * x).sum() / w.sum()
    check("3c", "starting at the minimiser is a fixed point",
          lambda: close(s.gradient_descent_quadratic(x, w, star, 0.01, 50), star))

    # One hand-computed step, to pin the update rule and the factor of 2.
    # f'(theta) = 2 * sum_i w_i (theta - x_i);  at theta=1, x=[0,2], w=[1,1]:
    #   grad = 2*((1-0) + (1-2)) = 0  -> theta stays at 1
    check("3c", "one hand-checked step",
          lambda: close(s.gradient_descent_quadratic(np.array([0., 2.]), np.array([1., 1.]),
                                                     1.0, 0.1, 1), 1.0))


TESTS = {"1e": test_1e, "1f": test_1f, "1g": test_1g, "1h": test_1h, "1i": test_1i,
         "2b": test_2b, "2d": test_2d, "2e": test_2e, "3c": test_3c}


def main():
    wanted = sys.argv[1:] or list(TESTS)
    for part in wanted:
        if part not in TESTS:
            print(f"unknown part {part!r}; choose from {', '.join(TESTS)}")
            return 2
        try:
            TESTS[part]()
        except Exception:
            _results.append((part, "<collection>", "FAIL", traceback.format_exc().strip().split("\n")[-1]))

    width = max(len(n) for _, n, _, _ in _results) if _results else 10
    last = None
    for part, name, status, detail in _results:
        if part != last:
            print(f"\n--- {part} ---")
            last = part
        mark = {"ok": "  ok  ", "skip": " skip ", "FAIL": " FAIL "}[status]
        print(f" [{mark}] {name:<{width}} {detail}")

    n_ok = sum(1 for r in _results if r[2] == "ok")
    n_fail = sum(1 for r in _results if r[2] == "FAIL")
    n_skip = sum(1 for r in _results if r[2] == "skip")
    print(f"\n{n_ok} passed, {n_fail} failed, {n_skip} skipped (not yet implemented)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
