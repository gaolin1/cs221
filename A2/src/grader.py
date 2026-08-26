#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
from util import read_tweet_data
import numpy as np
import torch
import torch.nn as nn

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


class Test_3a(GradedTestCase):
    @graded()
    def test_0(self):
        """3a-0-basic: build_vocabulary test"""
        examples = ["hello world", "hello python", "world peace", "hello there"]
        vocab = submission.build_vocabulary(examples)

        self.assertTrue(vocab.size() > 0, "Vocabulary size should be greater than 0")
        self.assertTrue(vocab.get_index("hello") >= 2, "Index of 'hello' should be >= 2")
        self.assertTrue(vocab.get_index("world") >= 2, "Index of 'world' should be >= 2")
        self.assertTrue(vocab.get_index("nonexistent") == 1, "Index of 'nonexistent' should be 1")

class Test_3b(GradedTestCase):
    @graded()
    def test_0(self):
        """3b-0-basic: text_to_features test"""
        examples = ["hello world", "hello test", "world test"]
        vocab = submission.build_vocabulary(examples)
        text = "hello world"
        features = submission.text_to_features(text, vocab)

        self.assertTrue(isinstance(features, np.ndarray), "Features should be a numpy array")
        self.assertTrue(is_equal(features.shape, (vocab.size(),)), "Feature vector should have shape (vocab.size(),)")

        hello_idx = vocab.get_index("hello")
        world_idx = vocab.get_index("world")

        if hello_idx >= 0:
            self.assertTrue(is_equal(features[hello_idx], 1.0), "Feature for 'hello' should be 1.0")
        if world_idx >= 0:
            self.assertTrue(is_equal(features[world_idx], 1.0), "Feature for 'world' should be 1.0")

        text_repeated = "hello hello world"
        features_repeated = submission.text_to_features(text_repeated, vocab)
        if hello_idx >= 0:
            self.assertTrue(is_equal(features_repeated[hello_idx], 2.0), "Feature for 'hello' should be 2.0")
        if world_idx >= 0:
            self.assertTrue(is_equal(features_repeated[world_idx], 1.0), "Feature for 'world' should be 1.0")

        text_unknown = "hello unknown_word"
        features_unknown = submission.text_to_features(text_unknown, vocab)
        self.assertTrue(is_equal(features_unknown.shape, (vocab.size(),)), "Feature vector should have shape (vocab.size(),)")

        features_empty = submission.text_to_features("", vocab)
        self.assertTrue(is_equal(features_empty.shape, (vocab.size(),)), "Feature vector should have shape (vocab.size(),)")
        self.assertTrue(np.sum(features_empty) == 0, "Sum of features for empty text should be 0")

class Test_3c(GradedTestCase):
    @graded()
    def test_0(self):
        """3c-0-basic: numpy_softmax test"""
        logits = np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
        result = submission.numpy_softmax(logits)

        self.assertTrue(is_equal(result.shape, (2, 3)), "Result shape should be (2, 3)")

        row_sums = np.sum(result, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0), "Row sums should be 1.0")
        self.assertTrue(np.all(result > 0), "All elements should be greater than 0")
        self.assertTrue(np.all(result <= 1.0), "All elements should be less than or equal to 1.0")

class Test_3d(GradedTestCase):
    @graded()
    def test_0(self):
        """3d-0-basic: numpy_cross_entropy_loss test"""
        predictions = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        result = submission.numpy_cross_entropy_loss(predictions, targets)

        self.assertTrue(isinstance(result, (float, np.floating)), "Result should be a float")

        self.assertGreater(result, 0, "Result should be greater than 0")

        perfect_pred = np.array([[1.0-1e-10, 1e-10, 1e-10], [1e-10, 1.0-1e-10, 1e-10]])
        perfect_loss = submission.numpy_cross_entropy_loss(perfect_pred, targets)
        self.assertLess(perfect_loss, 0.001, "Perfect loss should be less than 0.001")

class Test_3e(GradedTestCase):
    @graded()
    def test_0(self):
        """3e-0-basic: numpy_compute_gradients test"""
        batch_size = 2
        num_features = 3
        num_classes = 3

        features = np.random.randn(batch_size, num_features)
        predictions = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
        targets = np.array([[1, 0, 0], [0, 0, 1]])

        grad_weights, grad_bias = submission.numpy_compute_gradients(features, predictions, targets)

        self.assertTrue(is_equal(grad_weights.shape, (num_features, num_classes)), "grad_weights should have shape (num_features, num_classes)")
        self.assertTrue(is_equal(grad_bias.shape, (1, num_classes)), "grad_bias should have shape (1, num_classes)")

        self.assertTrue(np.any(np.abs(grad_weights) > 1e-6), "grad_weights should have non-zero elements")

class Test_3f(GradedTestCase):
    @graded()
    def test_0(self):
        """3f-0-basic: predict_linear_classifier test"""
        np.random.seed(42)
        batch_size = 6
        num_features = 10
        num_classes = 3

        test_features = np.random.randn(batch_size, num_features)
        test_labels = np.eye(num_classes).repeat(batch_size // num_classes, axis=0)

        weights = np.random.randn(num_features, num_classes)
        bias = np.zeros((1, num_classes))

        accuracy = submission.predict_linear_classifier(test_features, test_labels, weights, bias)

        self.assertTrue(isinstance(accuracy, (float, np.floating)), "Accuracy should be a float")
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1")

class Test_3g(GradedTestCase):
    @graded()
    def test_0(self):
        """3g-0-basic: train_linear_classifier basic test"""
        np.random.seed(42)
        batch_size = 6
        num_features = 10
        num_classes = 3

        train_features = np.random.randn(batch_size, num_features)
        train_labels = np.eye(num_classes).repeat(batch_size // num_classes, axis=0)
        val_features = np.random.randn(3, num_features)
        val_labels = np.eye(num_classes)

        weights, bias = submission.train_linear_classifier(
            train_features, train_labels, val_features, val_labels,
            num_epochs=5, lr=0.01
        )

        self.assertTrue(is_equal(weights.shape, (num_features, num_classes)), "Weights should have shape (num_features, num_classes)")
        self.assertTrue(is_equal(bias.shape, (1, num_classes)), "Bias should have shape (1, num_classes)")

        self.assertTrue(np.any(np.abs(weights) > 1e-4), "Weights should have non-zero elements")

    # BEGIN_HIDE
    # END_HIDE

class Test_4a(GradedTestCase):
    @graded()
    def test_0(self):
        """4a-0-basic: text_to_average_embedding test"""
        examples = ["hello world test", "hello test"]
        vocab = submission.build_vocabulary(examples)

        embedding_dim = 8
        embedding_layer = nn.Embedding(vocab.size(), embedding_dim, padding_idx=0)

        text = "hello world"
        result = submission.text_to_average_embedding(text, vocab, embedding_layer)

        self.assertTrue(is_equal(result.shape, torch.Size([embedding_dim])), "Result should have shape (embedding_dim,)")
        self.assertTrue(torch.is_tensor(result), "Result should be a PyTorch tensor")

        empty_result = submission.text_to_average_embedding("", vocab, embedding_layer)
        self.assertTrue(is_equal(empty_result.shape, torch.Size([embedding_dim])), "Result should have shape (embedding_dim,)")

class Test_4b(GradedTestCase):
    @graded()
    def test_0(self):
        """4b-0-basic: extract_averaged_features test"""
        examples = ["hello world", "world peace", "hello peace"]
        vocab = submission.build_vocabulary(examples)

        embedding_dim = 10
        embedding_layer = nn.Embedding(vocab.size(), embedding_dim, padding_idx=0)

        texts = ["hello world", "peace world"]
        features = submission.extract_averaged_features(texts, vocab, embedding_layer)

        self.assertTrue(is_equal(features.shape, torch.Size([2, embedding_dim])), "Features should have shape (2, embedding_dim)")
        self.assertTrue(torch.is_tensor(features), "Features should be a PyTorch tensor")

class Test_4c(GradedTestCase):
    @graded()
    def test_0(self):
        """4c-0-basic: MLPClassifier test"""
        embedding_dim, hidden_dim, output_dim = 16, 8, 3
        model = submission.MLPClassifier(embedding_dim, hidden_dim, output_dim)

        x = torch.randn(4, embedding_dim)
        output = model(x)

        self.assertTrue(is_equal(output.shape, torch.Size([4, output_dim])), "Output should have shape (4, output_dim)")

        self.assertTrue(hasattr(model, 'fc1'), "Model should have attribute 'fc1'")
        self.assertTrue(hasattr(model, 'fc2'), "Model should have attribute 'fc2'")
        self.assertTrue(hasattr(model, 'relu'), "Model should have attribute 'relu'")

class Test_4d(GradedTestCase):
    @graded()
    def test_0(self):
        """4d-0-basic: torch utility functions test"""
        logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
        result = submission.torch_softmax(logits)
        self.assertTrue(is_equal(result.shape, torch.Size([2, 3])), "Result should have shape (2, 3)")
        row_sums = torch.sum(result, dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(2)), "Each row should sum to 1")

        predictions = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
        targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loss = submission.torch_cross_entropy_loss(predictions, targets)
        self.assertTrue(is_equal(loss.shape, torch.Size([])), "Loss should have shape ()")
        self.assertTrue(loss.item() > 0, "Loss should be positive")

        param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        grad = torch.tensor([0.1, 0.2, 0.3])
        lr = 0.5
        original_param = param.clone()
        submission.update_parameter(param, grad, lr)
        expected = original_param - lr * grad
        self.assertTrue(torch.allclose(param, expected), "Parameter update is incorrect")

class Test_4e(GradedTestCase):
    @graded()
    def test_0(self):
        """4e-0-basic: predict_mlp test"""
        examples = ["good happy", "bad sad", "great wonderful"]
        vocab = submission.build_vocabulary(examples)

        embedding_dim, hidden_dim, output_dim = 8, 4, 3
        embedding_layer = nn.Embedding(vocab.size(), embedding_dim, padding_idx=0)
        classifier = submission.MLPClassifier(embedding_dim, hidden_dim, output_dim)

        texts = ["good", "bad"]
        labels = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)

        accuracy = submission.predict_mlp(texts, labels, classifier, embedding_layer, vocab)

        self.assertTrue(isinstance(accuracy, float), "Accuracy should be a float")
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1")

class Test_4f(GradedTestCase):
    @graded()
    def test_0(self):
        """4f-0-basic: train_mlp_classifier basic test"""
        train_texts = ["happy good wonderful", "sad bad terrible", "love great amazing"]
        train_labels = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        val_texts = ["good", "bad"]
        val_labels = np.array([[0, 1, 0], [1, 0, 0]])

   
        classifier, embedding_layer, vocab = submission.train_mlp_classifier(
            train_texts, train_labels, val_texts, val_labels,
            num_epochs=3, lr=0.01, embedding_dim=8, hidden_dim=4,
            batch_size=2, num_classes=3
        )

        self.assertTrue(isinstance(classifier, submission.MLPClassifier), "Classifier should be an instance of MLPClassifier")
        self.assertTrue(isinstance(embedding_layer, nn.Embedding), "Embedding layer should be an instance of nn.Embedding")
        self.assertTrue(hasattr(vocab, 'size'), "Vocab should have a 'size' attribute")

        self.assertTrue(vocab.size() > 0, "Vocab size should be greater than 0")
    
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
