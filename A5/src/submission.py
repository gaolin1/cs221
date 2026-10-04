from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.getDirection() gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    # ### START CODE HERE ###
    # recursive loop to get nested list of all states on a ghost index
    def getGhostChildren(num_agents, input_state, input_depth, input_idx: int = 1):
      states = []
      input_actions = input_state.getLegalActions(input_idx)
      # if no further actions then return the current states' score
      if input_actions:
        if input_idx == num_agents - 1:
          if input_depth - 1 == 0:
            # we are at the end, get the scores
            for input_action in input_actions:
              new_state = input_state.generateSuccessor(input_idx, input_action)
              agent_score = self.evaluationFunction(new_state)
              states.append(agent_score)
            return states
          else:
            input_depth -= 1
            input_idex = 0
            for input_action in input_actions:
              new_agent_state = input_state.generateSuccessor(input_idx, input_action)
              states.append(getMiniMaxScore(getGhostChildren(num_agents, new_agent_state, input_depth, input_idex), "max"))
        else:
          for input_action in input_actions:
            new_state = input_state.generateSuccessor(input_idx, input_action)
            states.append(getMiniMaxScore(getGhostChildren(num_agents, new_state, input_depth, input_idx + 1)))
      else:
        states.append(self.evaluationFunction(input_state))
      return states
    # recursive helper on applying min for all nested ghost scores
    def getMiniMaxScore(scores_list, operation: str = "min"):
      if operation == "max":
        return max(scores_list)
      else:
        return min(scores_list)
    
    # gets action
    ghost_state_scores = []
    num_agents = gameState.getNumAgents()
    # loop through all possible actions and get a nested list of all scores

    allowed_agent_actions = gameState.getLegalActions(self.index)
    for agent_action in allowed_agent_actions:
      initial_ghost_state = gameState.generateSuccessor(self.index, agent_action)
      ghost_scores = getGhostChildren(num_agents, initial_ghost_state, self.depth)
      ghost_min_score = getMiniMaxScore(ghost_scores)
      ghost_state_scores.append(ghost_min_score)

    action_index = ghost_state_scores.index(max(ghost_state_scores))
    best_agent_action = allowed_agent_actions[action_index]
    
    return best_agent_action
    # ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    
    # ### START CODE HERE ###
    # for min (ghost nodes) we prune if it's less than alpha
    # for max (agent nodes) we prune if it's higher than beta
    def getGhostChildren(num_agents, alpha: float, beta: float, input_state, input_depth, input_idx: int = 1):
      states = []
      input_actions = input_state.getLegalActions(input_idx)
      # if no further actions then return the current states' score
      if input_actions:
        if input_idx == num_agents - 1:
          if input_depth - 1 == 0:
            # we are at the end, get the scores for ghost leaf node
            for input_action in input_actions:
              new_state = input_state.generateSuccessor(input_idx, input_action)
              agent_score = self.evaluationFunction(new_state)
              states.append(agent_score)
              beta = updateAlphaBeta(agent_score, beta)
              if earlyPrune(agent_score, alpha):
                break
            return states
          else:
            # later ghost to pacman depth transition nodes
            input_depth -= 1
            input_idex = 0
            for input_action in input_actions:
              new_agent_state = input_state.generateSuccessor(input_idx, input_action)
              new_ghost_to_agent_score = getMiniMaxScore(getGhostChildren(num_agents, alpha, beta, new_agent_state, input_depth, input_idex), "max")
              states.append(new_ghost_to_agent_score)
              beta = updateAlphaBeta(new_ghost_to_agent_score, beta)
              if earlyPrune(new_ghost_to_agent_score, alpha):
                break
              
        else:
          # non last layer nodes
          max_node = False
          for input_action in input_actions:
            new_state = input_state.generateSuccessor(input_idx, input_action)
            new_score = getMiniMaxScore(getGhostChildren(num_agents, alpha, beta, new_state, input_depth, input_idx + 1))
            states.append(new_score)
            if input_idx == 0:
              max_node = True
            # determine if it's a max or min node
            if max_node:
              alpha = updateAlphaBeta(new_score, alpha, "alpha")
              if earlyPrune(new_score, beta, "beta"):
                break
            else:
              beta = updateAlphaBeta(new_score, beta)
              if earlyPrune(new_score, alpha):
                break
      else:
        states.append(self.evaluationFunction(input_state))
      return states
    
    def getMiniMaxScore(scores_list: list, operation: str = "min"):
      if operation == "max":
        return max(scores_list)
      else:
        return min(scores_list)

    def updateAlphaBeta(value: float, threshold: float, type: str = "beta"):
      new_threshold = threshold
      if type == "alpha":
        if value > threshold:
          new_threshold = value
      elif type == "beta":
        if value < threshold:
          new_threshold = value
      return new_threshold

    def earlyPrune(value: float, threshold: float, type: str = "alpha") -> bool:
      if value == threshold:
        return True
      if type == "beta":
        if value > threshold:
          return True
      elif type == "alpha":
        if value < threshold:
          return True
      return False
        
    # initializes the search layer
    ghost_state_scores = []
    num_agents = gameState.getNumAgents()
    alpha = float("-inf")
    ghost_beta = float("inf")

    allowed_agent_actions = gameState.getLegalActions(self.index)
    for action_index, agent_action in enumerate(allowed_agent_actions):
      ghost_move = gameState.generateSuccessor(self.index, agent_action)
      if action_index == 0:
        agent_alpha = alpha
      ghost_scores = getGhostChildren(num_agents, agent_alpha, ghost_beta, ghost_move, self.depth)
      ghost_min_score = getMiniMaxScore(ghost_scores)
      ghost_state_scores.append(ghost_min_score)
      if max(ghost_state_scores) > agent_alpha:
        agent_alpha = max(ghost_state_scores)

    action_index = ghost_state_scores.index(max(ghost_state_scores))
    best_action = allowed_agent_actions[action_index]
    
    return best_action
    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # ### START CODE HERE ###
    # recursive loop to get nested list of all states on a ghost index
    def getGhostChildren(num_agents, input_state, input_depth, input_idx: int = 1):
      states = []
      input_actions = input_state.getLegalActions(input_idx)
      # if no further actions then return the current states' score
      if input_actions:
        if input_idx == num_agents - 1:
          if input_depth - 1 == 0:
            # we are at the end, get the scores
            for input_action in input_actions:
              new_state = input_state.generateSuccessor(input_idx, input_action)
              agent_score = self.evaluationFunction(new_state)
              states.append(agent_score)
            return states
          else:
            input_depth -= 1
            input_idex = 0
            for input_action in input_actions:
              new_agent_state = input_state.generateSuccessor(input_idx, input_action)
              states.append(getMiniMaxScore(getGhostChildren(num_agents, new_agent_state, input_depth, input_idex), "max"))
        else:
          for input_action in input_actions:
            new_state = input_state.generateSuccessor(input_idx, input_action)
            new_score = getGhostChildren(num_agents, new_state, input_depth, input_idx + 1)
            states.append(sum(new_score)/len(new_score))
      else:
        states.append(self.evaluationFunction(input_state))
      return states
    # recursive helper on applying min for all nested ghost scores
    def getMiniMaxScore(scores_list, operation: str = "min"):
      if operation == "max":
        return max(scores_list)
      else:
        return min(scores_list)
    
    # gets action
    ghost_state_scores = []
    num_agents = gameState.getNumAgents()
    # loop through all possible actions and get a nested list of all scores

    allowed_agent_actions = gameState.getLegalActions(self.index)
    for agent_action in allowed_agent_actions:
      initial_ghost_state = gameState.generateSuccessor(self.index, agent_action)
      ghost_scores = getGhostChildren(num_agents, initial_ghost_state, self.depth)
      ghost_min_score = sum(ghost_scores)/len(ghost_scores)
      ghost_state_scores.append(ghost_min_score)

    action_index = ghost_state_scores.index(max(ghost_state_scores))
    best_agent_action = allowed_agent_actions[action_index]
    
    return best_agent_action

    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """
  
  # ### START CODE HERE ###
  oldScore = currentGameState.getScore()
  # - log d from distance to food, food feature
  food_weight = 10
  # get agent position
  agent_position = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  num_food = currentGameState.getNumFood()
  if num_food == 0:
    food_feature = 1
  else:
    nearest_food = None
    position_x, position_y = agent_position
    for radius in range(1, food.width + food.height):
      for dx in range(-radius, radius + 1):
        dy = radius - abs(dx)
        x = position_x + dx
        if not (0 <= x < food.width):
          continue
        for y in {position_y + dy, position_y - dy}:
          if 0 <= y < food.height and food[x][y]:
              nearest_food = radius
              break
        if nearest_food:
          break
      if nearest_food:
        break
    food_feature = 1 / (1 + nearest_food)

  # ghost feature to maximize chance to eat ghost
  #danger_weight = 100
  catch_weight = 50
  ghost_states = currentGameState.getGhostStates()
  #danger_penalty = 0
  catch_award = 0
  scared = False
  for ghost in ghost_states:
    ghost_position = ghost.getPosition()
    disance_to_ghost = manhattanDistance(agent_position, ghost_position)
    if ghost.scaredTimer > 0:
      scared = True
      if ghost.scaredTimer > disance_to_ghost:
        catch_award += 1 / (1 + disance_to_ghost)
    #elif disance_to_ghost <= 2:
    #    danger_penalty += 1 / (1 + disance_to_ghost)

  # capsule rewards, for pushing for capsule
  capsule_distances = []
  capsule_weight = 20
  capsules = currentGameState.getCapsules()
  if not scared:
    for capsule in capsules:
      capsule_distance = manhattanDistance(agent_position, capsule)
      capsule_distances.append(capsule_distance)
  if capsule_distances:
    capsule_feature = 1 / (1 + min(capsule_distances))
  else:
    capsule_feature = 1
  
  better = oldScore + (food_weight * food_feature) + (catch_weight * catch_award)  + (capsule_weight * capsule_feature) - (20 * num_food) - (50 * len(capsules))
  return better
  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction
