#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
from einops import rearrange, einsum

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS
#############################################
TOLERANCE = 1e-4  # For measuring whether two floats are equal

def is_collection(x):
    return isinstance(x, list) or isinstance(x, tuple)

# Return whether two answers are equal.
def is_equal(true_answer, pred_answer, tolerance=TOLERANCE):
    # Handle floats specially
    if isinstance(true_answer, float) or isinstance(pred_answer, float):
        return abs(true_answer - pred_answer) < tolerance
    # Recurse on collections to deal with floats inside them
    if is_collection(true_answer) and is_collection(pred_answer) and len(true_answer) == len(pred_answer):
        for a, b in zip(true_answer, pred_answer):
            if not is_equal(a, b):
                return False
        return True
    if isinstance(true_answer, dict) and isinstance(pred_answer, dict):
        if len(true_answer) != len(pred_answer):
            return False
        for k, v in list(true_answer.items()):
            if not is_equal(pred_answer.get(k), v):
                return False
        return True

    # Numpy array comparison
    if type(true_answer).__name__ == 'ndarray':
        if isinstance(true_answer, np.ndarray) and isinstance(pred_answer, np.ndarray):
            if true_answer.shape != pred_answer.shape:
                return False
            for a, b in zip(true_answer, pred_answer):
                if not is_equal(a, b):
                    return False
            return True

    # Do normal comparison
    return true_answer == pred_answer

#########
# TESTS #
#########


class Test_1e(GradedTestCase):
    @graded()
    def test_0(self):
        """1e-0-basic: linear_project with bias (small deterministic)"""
        x = np.array([[1.0, 2.0, 3.0],
                    [0.0, -1.0, 4.0]])  # (B=2, Din=3)
        W = np.array([[1.0, 0.0],
                    [0.5, -1.0],
                    [2.0, 3.0]])        # (Din=3, Dout=2)
        b = np.array([0.1, -0.2])         # (Dout=2)
        expected = x @ W + b
        out = submission.linear_project(x, W, b)
        self.assertTrue(is_equal(expected, out), "Expected %s but got %s" % (expected, out))

    # BEGIN_HIDE
    # END_HIDE


class Test_1f(GradedTestCase):
    @graded()
    def test_0(self):
        """1f-0-basic: split_last_dim pattern applies correctly"""
        x = np.arange(12, dtype=float).reshape(2, 6)  # (B=2, D=6)
        num_groups = 3
        expected = x.reshape(2, num_groups, 6 // num_groups)
        pattern = submission.split_last_dim_pattern()
        out = rearrange(x, pattern, g=num_groups)
        self.assertTrue(is_equal(expected, out), "Expected %s but got %s" % (expected, out))
        
    # BEGIN_HIDE
    # END_HIDE

class Test_1g(GradedTestCase):
    @graded()
    def test_0(self):
        """1g-0-basic: normalized_inner_products small case + normalization"""
        A = np.array([[[1., 0.], [0., 1.]]])  # (B=1, M=2, D=2)
        Bm = np.array([[[1., 2.], [3., 4.], [0., 1.]]])  # (B=1, N=3, D=2)
        expected = np.einsum('bmd,bnd->bmn', A, Bm)
        out = submission.normalized_inner_products(A, Bm, normalize=False)
        self.assertTrue(is_equal(expected, out), "Expected %s but got %s" % (expected, out))
        # normalized
        D = A.shape[-1]
        expected_norm = expected / np.sqrt(D)
        out_norm = submission.normalized_inner_products(A, Bm, normalize=True)
        self.assertTrue(is_equal(expected_norm, out_norm), "Expected %s but got %s" % (expected_norm, out_norm))
    
    # BEGIN_HIDE
    # END_HIDE

class Test_1h(GradedTestCase):
    @graded()
    def test_0(self):
        """1h-0-basic: mask_strictly_upper sets upper triangle to -inf"""
        B, L = 1, 4
        scores = np.arange(B * L * L, dtype=float).reshape(B, L, L)
        out = submission.mask_strictly_upper(scores.copy())
        expected = scores.copy()
        triu_rows, triu_cols = np.triu_indices(L, k=1)
        expected[:, triu_rows, triu_cols] = -np.inf
        # Check non-inf values first
        non_inf_mask = ~np.isinf(expected)
        self.assertTrue(is_equal(expected[non_inf_mask], out[non_inf_mask]), "Expected %s but got %s" % (expected[non_inf_mask], out[non_inf_mask]))
        # Check inf values separately
        inf_mask = np.isinf(expected)
        self.assertTrue(is_equal(np.all(np.isinf(out[inf_mask])), True), "Expected all inf but got %s" % (out[inf_mask]))

    # BEGIN_HIDE
    # END_HIDE

class Test_1i(GradedTestCase):
    @graded()
    def test_0(self):
        """1i-0-basic: prob_weighted_sum einsum string deterministic"""
        P = np.array([[0.25, 0.25, 0.5]])  # (B=1, N=3)
        V = np.array([[[1., 0.], [0., 1.], [2., 2.]]])  # (B=1, N=3, D=2)
        expected = einsum(P, V, 'b n, b n d -> b d')
        pattern = submission.prob_weighted_sum_einsum()
        out = einsum(P, V, pattern)
        self.assertTrue(is_equal(expected, out), "Expected %s but got %s" % (expected, out))

    # BEGIN_HIDE
    # END_HIDE

class Test_2b(GradedTestCase):
    @graded()
    def test_0(self):
        """2b-0-basic: gradient_warmup deterministic"""
        w = np.array([1.0, -2.0, 3.0])
        c = np.array([0.0, 1.0, -1.0])
        expected = 2.0 * (w - c)
        out = submission.gradient_warmup(w, c)
        self.assertTrue(is_equal(expected, out), "Expected %s but got %s" % (expected, out))

    # BEGIN_HIDE
    # END_HIDE

class Test_2d(GradedTestCase):
    @graded()
    def test_0(self):
        """2d-0-basic: matrix_grad deterministic"""
        A = np.array([[2., 1., 3.],
                  [4., 5., 6.]])  # (m=2, p=3)
        B = np.array([[7., 8.],
                    [9., 0.],
                    [1., 2.]])      # (p=3, n=2)
        grad_A, grad_B = submission.matrix_grad(A, B)
        # Expected gradients
        row_sum_B = B.sum(axis=1)  # (3,)
        col_sum_A = A.sum(axis=0)  # (3,)
        expected_grad_A = np.ones((A.shape[0], 1)) @ row_sum_B[None, :]
        expected_grad_B = col_sum_A[:, None] @ np.ones((1, B.shape[1]))
        self.assertTrue(is_equal(expected_grad_A, grad_A), "Expected %s but got %s" % (expected_grad_A, grad_A))
        self.assertTrue(is_equal(expected_grad_B, grad_B), "Expected %s but got %s" % (expected_grad_B, grad_B))

    # BEGIN_HIDE
    # END_HIDE

class Test_2e(GradedTestCase):
    @graded()
    def test_0(self):
        """2e-0-basic: finite differences matches analytic gradient (single case)"""
        rng = np.random.default_rng(42)
        n, d = 5, 4
        A = rng.standard_normal((n, d))
        b = rng.standard_normal(n)
        w = rng.standard_normal(d)
        g_analytic = submission.lsq_grad(w, A, b)
        g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
        self.assertIsNotNone(g_analytic, "lsq_grad returned None — not implemented")
        self.assertIsNotNone(g_numeric, "lsq_finite_diff_grad returned None — not implemented")
        self.assertTrue(is_equal(g_analytic, g_numeric), "Expected %s but got %s" % (g_analytic, g_numeric))

    # BEGIN_HIDE
    # END_HIDE

class Test_3c(GradedTestCase):
    @graded()
    def test_0(self):
        """3c-0-basic: gradient descent converges to weighted average (deterministic)"""
        # Simple deterministic case: weighted average should be the minimizer
        x = np.array([0.0, 10.0])
        w = np.array([1.0, 3.0])
        theta_star = (w * x).sum() / w.sum()
        theta0 = 100.0
        lr = 0.25 / w.sum()  # stable stepsize (< 1/(2*sum w))
        theta = submission.gradient_descent_quadratic(x, w, theta0, lr, num_steps=200)
        self.assertTrue(is_equal(theta_star, theta), "Expected %s but got %s" % (theta_star, theta))

    # BEGIN_HIDE
    # END_HIDE

def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
