import math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Dict, Any, Optional, Iterable, Set
import gymnasium as gym
import numpy as np

import util
from util import ContinuousGymMDP, StateT, ActionT
from custom_mountain_car import CustomMountainCarEnv

############################################################
# Problem 3a
# Implementing Value Iteration on Number Line (from Problem 1)
'''
Given transition probabilities and rewards, computes and returns V and
the optimal policy pi for each state.
- succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
- Returns: Dictionary mapping each state to an action.
'''
def valueIteration(succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]], discount: float, epsilon: float = 0.001):
    # Define a mapping from states to Set[Actions] so we can determine all the actions that can be taken from s.
    # You may find this useful in your approach.
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        # Return Q(state, action) based on V(state)
        # ### START CODE HERE ###
        Q_value = 0
        for nextState, probability, reward in succAndRewardProb[(state, action)]:
            reward_and_previous_value = reward + discount * V[nextState]
            Q_value += probability * reward_and_previous_value
        return Q_value
        # ### END CODE HERE ###

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        # Return the policy given V.
        # ### START CODE HERE ###
        optimal_policy = dict()
        for current_state in V.keys():
            optimal_action = []
            for current_action in stateActions[current_state]:
                current_q = computeQ(V, current_state, current_action)
                optimal_action.append(current_q)
                max_q = max(optimal_action)
                if current_q == max_q:
                    optimal_policy[current_state] = current_action
        return optimal_policy
        # ### END CODE HERE ###

    print('Running valueIteration...')
    V = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
    numIters = 0
    while True:
        newV = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
        # update V values using the computeQ function above.
        # repeat until the V values for all states converge (changes between iterations are less than epsilon).
        # ### START CODE HERE ###
        state_diff = []
        for state in stateActions.keys():
            Q_action_value = []
            for action in stateActions[state]:
                Q_value = computeQ(V, state, action)
                Q_action_value.append(Q_value)
            max_Q_value = max(Q_action_value)
            newV[state] = max_Q_value
            state_diff.append(abs(max_Q_value - V[state]))
        if max(state_diff) < epsilon:
            break
        # ### END CODE HERE ###
        V = newV
        numIters += 1
    V_opt = V
    print(("valueIteration: %d iterations" % numIters))
    return computePolicy(V_opt)


# Runs value iteration algorithm on the number line MDP
# and prints out optimal policy for each state.
def run_VI_over_numberLine(mdp: util.NumberLineMDP):
    succAndRewardProb = {
        (-mdp.n + 1, 1): [
            (-mdp.n + 2, 0.2, mdp.penalty),
            (-mdp.n, 0.8, mdp.leftReward),
        ],
        (-mdp.n + 1, 2): [
            (-mdp.n + 2, 0.3, mdp.penalty),
            (-mdp.n, 0.7, mdp.leftReward),
        ],
        (mdp.n - 1, 1): [(mdp.n - 2, 0.8, mdp.penalty), (mdp.n, 0.2, mdp.rightReward)],
        (mdp.n - 1, 2): [(mdp.n - 2, 0.7, mdp.penalty), (mdp.n, 0.3, mdp.rightReward)],
    }

    for s in range(-mdp.n + 2, mdp.n - 1):
        succAndRewardProb[(s, 1)] = [
            (s + 1, 0.2, mdp.penalty),
            (s - 1, 0.8, mdp.penalty),
        ]
        succAndRewardProb[(s, 2)] = [
            (s + 1, 0.3, mdp.penalty),
            (s - 1, 0.7, mdp.penalty),
        ]

    pi = valueIteration(succAndRewardProb, mdp.discount)
    return pi


############################################################
# Problem 3b
# Model-Based Monte Carlo


class ModelBasedMonteCarlo(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, calcValIterEvery: int = 10000,
                 explorationProb: float = 0.2,) -> None:
        self.actions = actions
        self.discount = discount
        self.calcValIterEvery = calcValIterEvery
        self.explorationProb = explorationProb
        self.numIters = 0

        # (state, action) -> {nextState -> ct} for all nextState
        self.tCounts = defaultdict(lambda: defaultdict(int))
        # (state, action) -> {nextState -> totalReward} for all nextState
        self.rTotal = defaultdict(lambda: defaultdict(float))

        self.pi = {} # Optimal policy for each state. state -> action

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # ** Important ** due to explorationProb being quite small you should avoid querying for random numbers if not exploring,
    # to avoid getting stuck in a local optima!
    # Should return random action if the given state is not in self.pi.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always follow the policy if available.
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e6: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        # ### START CODE HERE ###
        if explore:
        # explore on, check prob to give choice
            random_pick = random.random()
            if random_pick < explorationProb:
                new_action = int(np.random.choice(self.actions))
        # if falls to follow, check, otherwise give random
            else:
                if state in self.pi.keys():
                    new_action = self.pi[state]
                else:
                    new_action = int(np.random.choice(self.actions))            
        else:
        # if not explore, check policy first
            if state in self.pi.keys():
                new_action = self.pi[state]
            else:
                new_action = int(np.random.choice(self.actions))
        return new_action
        # ### END CODE HERE ###

    # We will call this function with (s, a, r, s'), which is used to update tCounts and rTotal.
    # For every self.calcValIterEvery steps, runs value iteration after estimating succAndRewardProb.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):

        self.tCounts[(state, action)][nextState] += 1
        self.rTotal[(state, action)][nextState] += reward

        if self.numIters % self.calcValIterEvery == 0:
            # Estimate succAndRewardProb based on self.tCounts and self.rTotal.
            # Then run valueIteration and update self.pi.
            succAndRewardProb = defaultdict(list)
            # ### START CODE HERE ###
            for countState, countAction in self.tCounts.keys():
            # succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
                total_state_action = sum(self.tCounts[(countState, countAction)].values())
                for countNextState in self.tCounts[(countState, countAction)].keys():
                    next_state_count = self.tCounts[(countState, countAction)][countNextState]
                    probability = next_state_count/total_state_action
                    average_reward = self.rTotal[(countState, countAction)][countNextState]/next_state_count
                    succAndRewardProb[(countState, countAction)].append([countNextState, probability, average_reward])
            optimal_policy = valueIteration(succAndRewardProb, self.discount)
            self.pi = optimal_policy
            # ### END CODE HERE ###

############################################################
# Problem 4a
# Performs Tabular Q-learning. Read util.RLAlgorithm for more information.
'''
- actions: the list of valid actions
- discount: a number between 0 and 1, which determines the discount factor
- explorationProb: the epsilon value indicating how frequently the policy returns a random action
- intialQ: the value for intializing Q values.
'''
class TabularQLearning(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # ** Important ** due to explorationProb being quite small you should avoid querying for random numbers if not exploring,
    # to avoid getting stuck in a local optima!
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        # ### START CODE HERE ###
        if explore:
            random_pick = random.random()
            if random_pick < explorationProb:
                newAction = int(np.random.choice(self.actions))
            else:
                if len(self.Q.keys()) > 0:
                    max_q_values = float("-inf")
                    for current_action in self.actions:
                        current_q_value = self.Q[(state, current_action)]
                        if current_q_value > max_q_values:
                            newAction = current_action
                            max_q_values = current_q_value
                else:
                    newAction = int(np.random.choice(self.actions))
        else:
            if len(self.Q.keys()) > 0:
                max_q_values = float("-inf")
                for current_action in self.actions:
                    current_q_values = self.Q[(state, current_action)]
                    if current_q_values > max_q_values:
                        newAction = current_action
                        max_q_values = current_q_values
            else:
                newAction = int(np.random.choice(self.actions))
        return newAction
        # ### END CODE HERE ###

    # Call this function to get the step size to update the Q-values.
    def getStepSize(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |Q|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update the Q values using self.getStepSize()
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        # ### START CODE HERE ###
        if terminal:
            self.Q[(state, action)] += self.getStepSize() * (reward - self.Q[(state, action)])
        else:
            max_q_value = float("-inf")
            for next_action in self.actions:
                next_q_value = self.Q[(nextState, next_action)]
                if next_q_value > max_q_value:
                    new_action = next_action
                    max_q_value = next_q_value
            utility = reward + self.discount * self.Q[(nextState, new_action)]
            self.Q[(state, action)] += self.getStepSize() * (utility - self.Q[(state, action)])
        # ### END CODE HERE ###

############################################################
# Problem 4b: Fourier feature extractor

def fourierFeatureExtractor(
        state: StateT,
        action: ActionT,
        maxCoeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), maxCoeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]
    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    # Below, implement the fourier feature extractor as similar to the doc string provided.
    # The return shape should be 1 dimensional ((maxCoeff+1)^(len(state)),).
    #
    # HINT: refer to util.polynomialFeatureExtractor as a guide for
    # doing efficient arithmetic broadcasting in numpy.

    # ### START CODE HERE ###
    # calculates 0, d0s0...max_coeff(d0s0)
    initial_d_s = state[0] * scale[0] * np.arange(0, maxCoeff + 1)
    combined_d_s = initial_d_s
    for i in range(1, len(state)):
    # calcualtes 0, d1s1...max_coeff(d1s1)
        new_d_s = state[i] * scale[i] * np.arange(0, maxCoeff + 1)
        combined_d_s = np.add.outer(combined_d_s, new_d_s)
        combined_d_s = combined_d_s.flatten()
    features = np.cos(np.pi * combined_d_s)
    # ### END CODE HERE ###

    return features

############################################################
# Problem 4c: Q-learning with Function Approximation
# Performs Function Approximation Q-learning. Read util.RLAlgorithm for more information.
'''
- featureDim: the dimensionality of the output of the feature extractor
- featureExtractor: a function that takes a state and action and returns a numpy array representing the feature.
- actions: the list of valid actions
- discount: a number between 0 and 1, which determines the discount factor
- explorationProb: the epsilon value indicating how frequently the policy returns a random action
'''
class FunctionApproxQLearning(util.RLAlgorithm):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, explorationProb=0.2):
        self.featureDim = featureDim
        self.featureExtractor = featureExtractor
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.W = np.random.standard_normal(size=(featureDim, len(actions)))
        self.numIters = 0

    def getQ(self, state: np.ndarray, action: int) -> float:
        # ### START CODE HERE ###
        action_weights = self.W[:, action]
        features = self.featureExtractor(state, action)
        action_and_weights = action_weights * features
        return np.sum(action_and_weights)
        # ### END CODE HERE ###

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # ** Important ** due to explorationProb being quite small you should avoid querying for random numbers if not exploring,
    # to avoid getting stuck in a local optima!
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        # ### START CODE HERE ###
        if explore:
            random_pick = random.random()
            if random_pick < explorationProb:
                new_action = int(np.random.choice(self.actions))
            else:
                max_q_value = float("-inf")
                for current_action in self.actions:
                    current_q_value = self.getQ(state, current_action)
                    if current_q_value > max_q_value:
                        new_action = current_action
                        max_q_value = current_q_value
        else:
            max_q_value = float("-inf")
            for current_action in self.actions:
                current_q_value = self.getQ(state, current_action)
                if current_q_value > max_q_value:
                    new_action = current_action
                    max_q_value = current_q_value
        return new_action
        # ### END CODE HERE ###

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.005 * (0.99)**(self.numIters / 500)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update W using self.getStepSize()
    def incorporateFeedback(self, state: np.ndarray, action: int, reward: float, nextState: np.ndarray, terminal: bool) -> None:
        # ### START CODE HERE ###
        if terminal:
        # at terminal, we already know the truth, which is the award
            self.W[:, action] += self.getStepSize() * (reward -  self.getQ(state, action)) * self.featureExtractor(state, action)
        else:
        # get max q action get feedback on that, environment returns the reward and next state and if terminal
            max_q_value = float("-inf")
            for current_action in self.actions:
                current_q_value = self.getQ(nextState, current_action)
                if current_q_value > max_q_value:
                    next_action = current_action
                    max_q_value = current_q_value
        # then we get our bootstrapped best estimated target
            target = reward + self.discount * self.getQ(nextState, next_action)
        # then we get how far off we are
            predicted_error = target - self.getQ(state, action)
        # then get the value to nudge towards estimated target into to weight (e.g. feature) shape
            update_value = self.getStepSize() * self.featureExtractor(state, action) * predicted_error
            self.W[:, action] += update_value
        # ### END CODE HERE ###

############################################################
# Problem 5c: Constrained Q-learning

class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, force: float, gravity: float,
                 max_speed: Optional[float] = None,
                 explorationProb=0.2):
        super().__init__(featureDim, featureExtractor, actions,
                         discount, explorationProb)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # ** Important ** due to explorationProb being quite small you should avoid querying for random numbers if not exploring,
    # to avoid getting stuck in a local optima!
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action that is valid.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        # ### START CODE HERE ###
        def findAllowedActions(state, actions_list: List[int] = self.actions) -> list:
            current_position = state[0]
            current_velocity = state[1]
            allowed_actions = []
            speeds_list = {}
            if self.max_speed is None:
                return actions_list
            for action in actions_list:
                new_velocity = current_velocity + ((action - 1) * self.force) - (np.cos(3 * current_position) * self.gravity)
                speeds_list[action] = new_velocity
                if new_velocity < self.max_speed:
                    allowed_actions.append(action)
            if allowed_actions == []:
                lowest_speed_action = min(speeds_list, key=speeds_list.get)
                allowed_actions.append(lowest_speed_action)
            return allowed_actions

        safe_actions = findAllowedActions(state)
        if explore:
            random_pick = random.random()
            if random_pick < explorationProb:
                new_action = int(np.random.choice(safe_actions))
            else:
                max_q_value = float("-inf")
                for action in safe_actions:
                    current_q_value = self.getQ(state, action)
                    if current_q_value > max_q_value:
                        new_action = action
                        max_q_value = current_q_value                
        else:
            max_q_value = float("-inf")
            for action in safe_actions:
                current_q_value = self.getQ(state, action)
                if current_q_value > max_q_value:
                    new_action = action
                    max_q_value = current_q_value
        return new_action
        # ### END CODE HERE ###

############################################################
# This is helper code for comparing the predicted optimal
# actions for 2 MDPs with varying max speed constraints
gym.register(
    id="CustomMountainCar-v0",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

mdp1 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)
mdp2 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)

# This is a helper function for 5c. This function runs
# ConstrainedQLearning, then simulates various trajectories through the MDP
# and compares the frequency of various optimal actions.
def compare_MDP_Strategies(mdp1: ContinuousGymMDP, mdp2: ContinuousGymMDP):
    rl1 = ConstrainedQLearning(
        36,
        lambda s, a: fourierFeatureExtractor(s, a, maxCoeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        mdp1.env.unwrapped.force,
        mdp1.env.unwrapped.gravity,
        10000,
        explorationProb=0.2,
    )
    rl2 = ConstrainedQLearning(
        36,
        lambda s, a: fourierFeatureExtractor(s, a, maxCoeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        mdp2.env.unwrapped.force,
        mdp2.env.unwrapped.gravity,
        0.065,
        explorationProb=0.2,
    )
    sampleKRLTrajectories(mdp1, rl1)
    sampleKRLTrajectories(mdp2, rl2)

def sampleKRLTrajectories(mdp: ContinuousGymMDP, rl: ConstrainedQLearning):
    accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
    for n in range(100):
        traj = util.sample_RL_trajectory(mdp, rl)
        accelerate_left += traj.count(0)
        no_accelerate += traj.count(1)
        accelerate_right += traj.count(2)

    print(f"\nRL with MDP -> start state:{mdp.startState()}, max_speed:{rl.max_speed}")
    print(f"  *  total accelerate left actions: {accelerate_left}, total no acceleration actions: {no_accelerate}, total accelerate right actions: {accelerate_right}")
