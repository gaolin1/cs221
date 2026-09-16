#!/usr/bin/env python3
import unittest, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase

import random, util, collections, json, math
import numpy as np

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################


#########
# TESTS #
#########

class Test_3a(GradedTestCase):

  @graded()
  def test_0(self):
    """3a-0-basic: Basic test of VI on problem 1."""
    
    mdp = util.NumberLineMDP()
    pi = submission.run_VI_over_numberLine(mdp)
    gold = {
        -1: 1,
        1: 2,
        0: 2
    }
    for key in pi:
        self.assertEqual(pi[key], gold[key], msg=f"Incorrect pi for the state: {key}")

  @graded()
  def test_1(self):
    """3a-1-basic: Test on arbitrary n, reward and penalty."""
    
    mdp = util.NumberLineMDP(10, 30, -1, 20)
    pi = submission.run_VI_over_numberLine(mdp)
    with open("3a-1-gold.json", "r") as f:
        gold = json.load(f)

    for key in gold:
        key_i = int(key)
        self.assertTrue(key_i in pi, msg=f"Optimal policy for state {key} not computed!")
        self.assertEqual(pi[key_i], gold[key], msg=f"Incorrect pi for the state: {key}")

  @graded(timeout=14, is_hidden=True)
  def test_2(self):
    """3a-2-hidden: Hidden test to make sure the code runs fast enough."""

    mdp = util.NumberLineMDP(n=500)
    pi = submission.run_VI_over_numberLine(mdp)
    # BEGIN_HIDE
    # END_HIDE

class Test_3b(GradedTestCase):
  
  @graded()
  def test_0(self):
    """3b-0-basic: Testing epsilon greedy for getAction."""

    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=1, explorationProb=0.2)
    rl.pi = {
        -1: 1,
        1: 2,
        0: 2
    }
    rl.numIters = 2e4
    counts = {
        -1: 0,
        0: 0,
        1: 0
    }
    for _ in range(10000):
        for state in range(-mdp.n + 1, mdp.n):
            action = rl.getAction(state)
            if action == rl.pi[state]:
                counts[state] += 1
    for key in counts:
        self.assertGreaterEqual(counts[key], 8800, msg=f"Too few optimal actions returned for the state {key}")
        self.assertLessEqual(counts[key], 9200, msg=f"Too few random actions returned for the state {key}")

  @graded()
  def test_1(self):
    """3b-1-basic: Casic test of incorporate feedback."""

    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=100, explorationProb=0.2)
    rl.numIters = 1
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.numIters = 100
    rl.incorporateFeedback(-1, 2, 10, -2, True)
    gold = {1:1, -1:2}

    self.assertEqual(rl.pi, gold, msg=f"Incorrect implementation of incorporateFeedback!")


  @graded(is_hidden=True)
  def test_2(self):
    """3b-2-hidden: Comprehensive test for incorporateFeedback."""

    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=100, explorationProb=0.2)
    rl.numIters = 1
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, -1, False)
    rl.incorporateFeedback(0, 2, -5, 1, False)
    rl.incorporateFeedback(0, 2, -5, -1, False)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.incorporateFeedback(1, 2, 50, 2, True)
    rl.incorporateFeedback(1, 2, -5, 0, False)
    rl.incorporateFeedback(-1, 1, -5, 0, False)
    rl.incorporateFeedback(-1, 2, -5, 0, False)
    rl.numIters = 100
    rl.incorporateFeedback(-1, 2, 10, -2, True)
    # BEGIN_HIDE
    # END_HIDE
        
  @graded(is_hidden=True)
  def test_3(self):
    """3b-3-hidden: Edge case handling."""

    mdp = util.NumberLineMDP()
    rl = submission.ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=1, explorationProb=0.2)
    counts = dict()
    for state in range(-mdp.n, mdp.n + 1):
        counts[state] = 0
    for _ in range(10000):
        for state in counts:
            action = rl.getAction(state)
            if action == 1:
                counts[state] += 1
    # BEGIN_HIDE
    # END_HIDE


class Test_4a(GradedTestCase):
  
  @graded()
  def test_0(self):
    """4a-0-basic: Basic test for incorporateFeedback."""

    mdp = util.NumberLineMDP()
    rl = submission.TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    self.assertEqual(0, rl.Q[(1, 2)])
    self.assertEqual(0, rl.Q[(1, 1)])
    self.assertEqual(-0.5, rl.Q[(0, 1)])
    
    rl.incorporateFeedback(1, 1, 50, 2, True)
    self.assertEqual(5.0, rl.Q[(1,1)])
    self.assertEqual(0, rl.Q[(1,2)])
    self.assertEqual(-0.5, rl.Q[(0,1)])

    rl.incorporateFeedback(-1, 2, -5, 0, False)
    self.assertEqual(5.0, rl.Q[(1, 1)])
    self.assertEqual(0, rl.Q[(1, 2)])
    self.assertEqual(-0.5, rl.Q[(0, 1)])
    self.assertEqual(0, rl.Q[(0, 2)])
    self.assertEqual(-0.5, rl.Q[(-1, 2)])

  @graded()
  def test_1(self):
    """4a-1-basic: Basic test for getAction."""

    mdp = util.NumberLineMDP()
    rl = submission.TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, -1, False)
    rl.incorporateFeedback(0, 2, -5, 1, False)
    rl.incorporateFeedback(0, 2, -5, -1, False)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.incorporateFeedback(1, 2, 50, 2, True)
    rl.incorporateFeedback(1, 2, -5, 0, False)
    rl.incorporateFeedback(-1, 1, -5, 0, False)
    rl.incorporateFeedback(-1, 1, 10, -2, True)
    rl.incorporateFeedback(-1, 2, -5, 0, False)
    pi = {
        -1: 1,
        0: 2,
        1: 1
    }
    for state in range(-mdp.n+1, mdp.n):
        self.assertEqual(pi[state], rl.getAction(state, explore=False), msg=f"Incorrect greedy action with the state {state}")

  @graded(is_hidden=True)
  def test_2(self):
    """4a-2-hidden: Hidden test for getAction."""

    mdp = util.NumberLineMDP()
    rl = submission.TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, 1, False)
    rl.incorporateFeedback(0, 1, -5, -1, False)
    rl.incorporateFeedback(0, 2, -5, 1, False)
    rl.incorporateFeedback(0, 2, -5, -1, False)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, 50, 2, True)
    rl.incorporateFeedback(1, 1, -5, 0, False)
    rl.incorporateFeedback(1, 2, 50, 2, True)
    rl.incorporateFeedback(1, 2, -5, 0, False)
    rl.incorporateFeedback(-1, 1, -5, 0, False)
    rl.incorporateFeedback(-1, 1, 10, -2, True)
    rl.incorporateFeedback(-1, 2, -5, 0, False)
    # BEGIN_HIDE
    # END_HIDE


class Test_4b(GradedTestCase):

  @graded()
  def test_0(self):
    """4b-0-basic: Basic test of fourierFeatureExtractor."""

    feature = submission.fourierFeatureExtractor((0.5, 0.3), 1)
    gold = np.load("4b-0-gold.npy", allow_pickle=True)
    self.assertEqual(gold.size, feature.size, f"Returned feature does not have the correct dimension!")
    gold_sorted = np.sort(gold)
    feature_sorted = np.sort(feature)
    for i in range(feature.size):
        self.assertTrue(math.isclose(feature_sorted[i], gold_sorted[i], abs_tol=1e-6), msg=f"Wrong value for an element of the feature: expected {str(gold_sorted[i])} but got {str(feature_sorted[i])}")


class Test_4c(GradedTestCase):

  @graded()
  def test_0(self):
    """4c-0-basic: Basic tests for getQ on FA."""

    mdp = util.ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
    rl = submission.FunctionApproxQLearning(36,
        lambda s, a: submission.fourierFeatureExtractor(s, a, maxCoeff=5, scale=[1, 15]),
        mdp.actions, mdp.discount, explorationProb=0.2)
    rl.W = np.zeros((36, 3))
    rl.incorporateFeedback((0, 0), 1, -1, (-0.2, -0.01), False)
    rl.incorporateFeedback((0.7, -0.03), 2, -1, (0.8, -0.01), False)
    rl.incorporateFeedback((-0.3, -0.05), 0, -1, (-0.4, -0.03), False)

    self.assertTrue(math.isclose(rl.getQ((0.2, -0.02), 1), -0.0074065262637628875, abs_tol=1e-6), msg=f"Wrong Q value computed for given state and action!")

  @graded()
  def test_1(self):
    """4c-1-basic: Basic tests for getAction on FA."""

    mdp = util.ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
    rl = submission.FunctionApproxQLearning(36,
        lambda s, a: submission.fourierFeatureExtractor(s, a, maxCoeff=5, scale=[1, 15]),
        mdp.actions, mdp.discount, explorationProb=0.2)
    rl.W = np.zeros((36, 3))
    rl.incorporateFeedback((0, 0), 1, -1, (-0.2, -0.01), False)
    rl.incorporateFeedback((0.7, -0.03), 2, -1, (0.8, -0.01), False)
    rl.incorporateFeedback((-0.3, -0.05), 0, -1, (-0.4, -0.03), False)

    action = rl.getAction((0.2, -0.02), explore=False)
    self.assertEqual(0, action, msg=f"Wrong action based on current weight!")
    
    action = rl.getAction((1, 0.03), explore=False)
    self.assertEqual(2, action, msg=f"Wrong action based on current weight!")
    
    action = rl.getAction((-0.6, -0.06), explore=False)
    self.assertEqual(0, action, msg=f"Wrong action based on current weight!")


  @graded(timeout=-1)
  def test_2(self):
    """4c-2-basic: Basic tests for incorporateFeedback on FA."""

    mdp = util.ContinuousGymMDP("MountainCar-v0", discount=0.999, timeLimit=1000)
    rl = submission.FunctionApproxQLearning(36,
        lambda s, a: submission.fourierFeatureExtractor(s, a, maxCoeff=5, scale=[1, 15]),
        mdp.actions, mdp.discount, explorationProb=0.2)
    rl.W = np.zeros((36, 3))
    rl.incorporateFeedback((0, 0), 1, -1, (-0.2, -0.01), False)
    rl.incorporateFeedback((0.7, -0.03), 2, -1, (0.8, -0.01), False)
    rl.incorporateFeedback((-0.3, -0.05), 0, -1, (-0.4, -0.03), False)
    gold = np.load("4c-2-gold.npy", allow_pickle=True)
        
    for i in range(36):
        self.assertTrue(any(np.all(np.isclose(gold[i], rl.W[j], atol=1e-6)) for j in range(36)), msg="Weight update incorrect!")
        self.assertTrue(any(np.all(np.isclose(gold[j], rl.W[i], atol=1e-6)) for j in range(36)), msg="Weight update incorrect!")


class Test_5c(GradedTestCase):

    @graded(timeout=60)
    def test_0(self):
        """5c-0-helper: Helper function to compare optimal policies over max speed constraints."""

        submission.compare_MDP_Strategies(submission.mdp1, submission.mdp2)

        self.skipTest("This test case is a helper function for students.")

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)