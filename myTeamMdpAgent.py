# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util, copy
from game import Directions
import game
from abc import ABC, abstractmethod
import sys, os, platform
from util import nearestPoint, Queue
import re
import subprocess
from collections import Counter

CD = os.path.dirname(os.path.abspath(__file__))
# FF_EXECUTABLE_PATH = "{}/../../bin/ff".format(CD)

FF_EXECUTABLE_PATH = "ff"

PACMAN_DOMAIN_FILE = f"{CD}/pacman-domain.pddl"
GHOST_DOMAIN_FILE = f"{CD}/ghost-domain.pddl"

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveMdpAgent', second = 'DefensivePDDLAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

####################
# Game MDP Model #
####################

class MDP(ABC):

  @abstractmethod
  def getStates(self, gameState):
    """
    Return a list of all states in the MDP.
    Not generally possible for large MDPs.
    """
    pass

  @abstractmethod
  def getStartState(self, gameState):
    """
    Return the start state of the MDP.
    """
    pass

  @abstractmethod
  def getPossibleActions(self, state, gameState):
    """
    Return list of possible actions from 'state'.
    """
    passctmethod

  @abstractmethod
  def getTransitionStatesAndProbs(self, state, action, gameState):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.

    Note that in Q-Learning and reinforcment
    learning in general, we do not know these
    probabilities nor do we directly model them.
    """
    pass

  @abstractmethod
  def getReward(self, state, action, nextState, gameState):
    """
    Get the reward for the state, action, nextState transition.

    Not available in reinforcement learning.
    """
    pass

  @abstractmethod
  def isTerminal(self, state, gameState):
    """
    Returns true if the current state is a terminal state.  By convention,
    a terminal state has zero future rewards.  Sometimes the terminal state(s)
    may have no possible actions.  It is also common to think of the terminal
    state as having a self-loop action 'pass' with zero reward; the formulations
    are equivalent.
    """
    pass

class gameStateMdpModel(MDP):
    '''
      MDP Model for GameState
    '''
    def __init__(self, gameState, index):
      self.foodReward = 1.0
      self.ghostReward = -3.0
      self.walls = gameState.getWalls().asList()
      self.wallsDict = {wallPos: True for wallPos in self.walls}
      self.layout = gameState.data.layout
      self.startState = gameState.getAgentPosition(index)
      self.isOnRedTeam = gameState.isOnRedTeam(index)

    def getStates(self, agentX):
      layout = self.layout
      states = list()
      walls = self.walls
      start = 1
      end = layout.width
      # reducing state space
      if self.isOnRedTeam:
        start = agentX
      else:
        end = agentX
      for x in range(start, end):
        for y in range(1, layout.height):
            state = (x, y)
            # avoiding walls to decrease state space
            if state not in walls:
              states.append(state)
      return states

    def getStartState(self):
      return self.startState

    def getPossibleActions(self, state):
      possibleActions = []
      x, y = state
      if (x,y+1) not in self.wallsDict:
        possibleActions.append(Directions.NORTH)
      if (x,y-1) not in self.wallsDict:
        possibleActions.append(Directions.SOUTH)
      if (x-1,y) not in self.wallsDict:
        possibleActions.append(Directions.WEST)
      if (x+1,y) not in self.wallsDict:
        possibleActions.append(Directions.EAST)
      return possibleActions

    def getTransitionStatesAndProbs(self, state, action):
      if self.isTerminal(state):
        return []
      x, y = state
      # no uncertainity in movements
      if action == Directions.NORTH:
        return [((x,y+1),1.0)]
      elif action == Directions.SOUTH:
        return [((x,y-1),1.0)]
      elif action == Directions.WEST:
        return [((x-1,y),1.0)]
      elif action == Directions.EAST:
        return [((x+1,y),1.0)]
      else:
        return [((x,y),1.0)]

    def setData(self, nextFoods, nextGhosts):
      self.foods = nextFoods
      self.ghosts = nextGhosts

    def getReward(self, state, action, nextState):
      return 0

    def isTerminal(self, state):
      pass

##########################
## Value Iteration Algo ##
##########################

class ValueIterationMDP:
  '''
    Value Iteration Implementation for given MDP Model
  '''
  def __init__(self, mdp, gameState, index, cAI, d=0.9):
    self.mdp = mdp
    self.discount = d
    weights = {
      'ghostDistance': 1.0
    }
    # set V(s) = 0 for all s
    self.values = util.Counter()
    agentPos = gameState.getAgentPosition(index)
    agentX = agentPos[0]
    self.mdpStates = self.mdp.getStates(agentX)
    self.terminalNodes = {}

    # Q-value initilization with rewards
    ghostsPos = []
    features = util.Counter()
    for state in self.mdp.ghosts:
      self.values[state] = self.mdp.ghostReward
      self.terminalNodes[state] = "#"
      ghostsPos.append(state)

    # food reward shaping
    for foodPos in self.mdp.foods:
      ghostDistance = 1
      if len(ghostsPos) > 0:
        ghostDistance =  min([cAI.getMazeDistance(ghostPos, foodPos) for ghostPos in ghostsPos])
        # ghost is at food - not rewarding at all
        if ghostDistance == 0:
          ghostDistance = -9999
      features['ghostDistance'] = 1/ghostDistance
      val = features * weights
      self.values[foodPos] = self.mdp.foodReward * val
      self.terminalNodes[foodPos] = "#"

  def performIterations(self, iterations=100):
    for i in range(iterations):
      newQValues = copy.deepcopy(self.values)
      for s in self.mdpStates:
        if s in self.terminalNodes:
          # go to next iteration
          continue
        action, QValue = self.nextActionFromValues(s)
        newQValues[s] = QValue
      # print(newQValues)
      self.values = copy.deepcopy(newQValues)
    print(f'Iterations {iterations} Done')
    # print(self.values)

  def getValueOfState(self, state):
      return self.values[state]

  def computeQValue(self, state, action):
    qValue = 0
    # get transition probabilities
    nextActionWithProbs = self.mdp.\
      getTransitionStatesAndProbs(state, action)
    for nextState, prob in nextActionWithProbs:
      qValue += prob * (self.mdp.getReward(state, action, nextState) +
                        self.discount * self.getValueOfState(nextState))
    return qValue

  def nextActionFromValues(self, state):
    actions = self.mdp.getPossibleActions(state)
    bestQValue = 0
    optimalAction = actions[0]
    # consider all actions Q-values
    for action in actions:
      nextActionWithProbs = self.mdp. \
        getTransitionStatesAndProbs(state, action)
      currentQValue = 0
      for nextState, prob in nextActionWithProbs:
        currentQValue += prob * (self.mdp.getReward(state, action, nextState) +
                          self.discount * self.getValueOfState(nextState))
      if currentQValue > bestQValue:
        bestQValue = currentQValue
        optimalAction = action
    return (optimalAction, bestQValue)

##########
# Agents #
##########

class OffensiveMdpAgent(CaptureAgent):
  """
  An offensive agent based on MDP value iteration approach
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.mdpModel = gameStateMdpModel(gameState, self.index)
    self.discount = 0.9
    self.isBoundary = False
    self.pddlProblemGenerator = PDDLProblemGenerator(gameState, self)
    self.boundaryPositions = self.pddlProblemGenerator.getBoundaryPos(gameState, 1)
    self.start = gameState.getAgentPosition(self.index)
    self.boundaryX =  self.pddlProblemGenerator.getBoundaryX(gameState)

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    features = {"problemObjective": None,
                "threshold": 0.6,
                "generateGrid": False}
    start = time.time()
    agentCurrentState = gameState.getAgentPosition(self.index)
    foods = self.getFood(gameState).asList()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
    foodLeft = len(self.getFood(gameState).asList())

    if agentCurrentState == self.start:
      # agent died
      self.isBoundary = False

    if agentCurrentState in self.boundaryPositions \
        and agentCurrentState != self.start:
      print("boundary reached")
      self.isBoundary = True

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    # reach boundary first
    if not self.isBoundary:
      # make him reach boundary first
      plannerPosition, plan, \
      problemObjective, planner = self.getPlan(gameState, features)
      action = planner.get_legal_action(agentCurrentState, plannerPosition)
      print(f'Action Planner: {action}')
      return action
    else:
      self.mdpModel.setData(foods, ghosts)
      self.valueIterator = ValueIterationMDP(self.mdpModel,
                                             gameState, self.index,
                                             self,
                                             self.discount)

      # perform 100 iterations across complete game states
      # Q-value should have some convergence
      self.valueIterator.performIterations(iterations=100)

      optimalAction, QValue = self.valueIterator.\
        nextActionFromValues(agentCurrentState)
      print(f'MDP Planner: {optimalAction}, Q-Value: {QValue}')
      print('Eval time for agent %d: %.4f' % (self.index, time.time() - start))
      return optimalAction

  def getPlan(self, gameState, features):
    problem_file, problemObjective = self.pddlProblemGenerator.generatePddlProblem(gameState, features)
    planner = PlannerFF(PACMAN_DOMAIN_FILE, problem_file)
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    return (plannerPosition, plan, problemObjective, planner)

###############
# PDDL Runner #
###############

class PDDLProblemGenerator():

  def __init__(self, gameState, captureAgentInstance):
    self.createPacmanDomain()
    self.cAIns = captureAgentInstance
    self.start = gameState.getAgentPosition(self.cAIns.index)
    self.index = self.cAIns.index
    self.masterFoods = self.cAIns.getFood(gameState).asList()
    self.masterCapsules = self.cAIns.getCapsules(gameState)
    self.homePos = self.getBoundaryPos(gameState, 1)
    self.pddlFluentGrid = self.generatePDDLFluentStatic(gameState)
    self.pddlObject = self.generatePddlObject(gameState)
    self.foodEaten = 0
    self.currScore = self.cAIns.getScore(gameState)
    self.history = Queue()

    self.stuck = False
    self.capsuleTimer = 0
    self.superPacman = False
    self.foodCarrying = 0

  def createPacmanDomain(self):
    pacman_domain_file = open(PACMAN_DOMAIN_FILE, "w")
    domain_statement = """
    (define (domain pacman)

      (:requirements
          :typing
          :negative-preconditions
      )

      (:types
          foods cells
      )

      (:predicates
          (cell ?p)

          ;pacman's cell location
          (at-pacman ?loc - cells)

          ;food cell location
          (at-food ?f - foods ?loc - cells)

          ;Indicates if a cell location has a ghost
          (has-ghost ?loc - cells)

          ;Indicated if a cell location has a capsule
          (has-capsule ?loc - cells)

          ;connects cells
          (connected ?from ?to - cells)

          ;pacman is carrying food
          (carrying-food)

          ;capsule eaten
          (capsule-eaten)

          ;want to die
          (want-to-die)

          ;die
          (die)
      )

      ; move pacman to location with no ghost
      (:action move
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (not (has-ghost ?to))
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      (:action move-no-restriction
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (want-to-die)
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      ; move pacman to food location if there no ghost
      (:action eat-food
          :parameters (?loc - cells ?f - foods)
          :precondition (and
                          (at-pacman ?loc)
                          (at-food ?f ?loc)
                        )
          :effect (and
                      ;; add
                      (carrying-food)
                      ;; del
                      (not (at-food ?f ?loc))
                  )
      )

      ; move pacman to food location if there no ghost
      (:action eat-capsule
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (has-capsule ?loc)
                        )
          :effect (and
                      ;; add
                      (capsule-eaten)
                      ;; del
                      (not (has-capsule ?loc))
                  )
      )

      (:action move-after-capsule-eaten
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (capsule-eaten)
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      ; move pacman to ghost location to die
      (:action get-eaten
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (has-ghost ?loc)
                        )
          :effect (and
                      ;; add
                      (die)
                      ;; del
                      ;; (not (has-ghost ?loc))
                  )
      )
    )
    """
    pacman_domain_file.write(domain_statement)
    pacman_domain_file.close()

  def getBoundaryPos(self, gameState, span=4):
    """
    Get Boundary Position for Home to set as return when chased by ghost
    """
    layout = gameState.data.layout
    x = layout.width / 2 - 1 if self.cAIns.red else layout.width / 2
    xSpan = [x - i for i in range(span)] if self.cAIns.red else [x + i for i in range(span)]
    walls = gameState.getWalls().asList()
    homeBound = list()
    for x in xSpan:
      pos = [(int(x), y) for y in range(layout.height) if (x, y) not in walls]
      homeBound.extend(pos)
    return homeBound

  def generatePddlObject(self, gameState):
    """
    Function for creating PDDL objects for the problem file.
    """

    # Get Cell Locations without walls and Food count for object setup.
    allPos = gameState.getWalls().asList(False)
    food_len = len(self.cAIns.getFood(gameState).asList())

    # Create Object PDDl line definition of objects.
    objects = list()
    cells = [f'cell{pos[0]}_{pos[1]}' for pos in allPos]
    cells.append("- cells\n")
    foods = [f'food{i+1}' for i in range(food_len)]
    foods.append("- foods\n")

    objects.append("\t(:objects \n")
    objects.append(f'\t\t{" ".join(cells)}')
    objects.append(f'\t\t{" ".join(foods)}')
    objects.append("\t)\n")

    return "".join(objects)

  def generatePDDLFluentStatic(self, gameState, remove=[]):
    # Set Adjacency Position
    allPos = gameState.getWalls().asList(False)
    connected = list()
    for pos in allPos:
      if (pos[0] + 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]+1}_{pos[1]})\n')
      if (pos[0] - 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]-1}_{pos[1]})\n')
      if (pos[0], pos[1] + 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]+1})\n')
      if (pos[0], pos[1] - 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]-1})\n')

    return "".join(connected)

  def generatePddlFluent(self, gameState, features):
    """
    Function for creating PDDL fluents for the problem file.
    """

    # Set Pacman Position
    pacmanPos = gameState.getAgentPosition(self.index)
    at_pacman = f'\t\t(at-pacman cell{pacmanPos[0]}_{pacmanPos[1]})\n'

    # Set Food Position
    foods = self.cAIns.getFood(gameState).asList()
    if len(foods) != 0:
      at_food = [f'\t\t(at-food food{i+1} cell{food[0]}_{food[1]})\n' for i, food in enumerate(foods)]

    # Set Ghost(s) positions
    has_ghost = list()
    enemies = [gameState.getAgentState(i) for i in self.cAIns.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    for ghost in ghosts:
      ghostPos = ghost.getPosition()
      if ghost.scaredTimer <= 3:
        has_ghost.append(f'\t\t(has-ghost cell{int(ghostPos[0])}_{int(ghostPos[1])})\n')

    # Set Capsule Position
    capsules = self.cAIns.getCapsules(gameState)
    has_capsule = [f'\t\t(has-capsule cell{capsule[0]}_{capsule[1]})\n' for capsule in capsules]

    fluents = list()
    fluents.append("\t(:init \n")
    fluents.append(at_pacman)
    fluents.append("".join(at_food))
    fluents.append("".join(has_ghost))
    fluents.append("".join(has_capsule))
    if features["problemObjective"] == "DIE":
      print("WANT_TO_DIE")
      fluents.append(f"\t\t(want-to-die)\n")
    fluents.append(self.pddlFluentGrid)
    fluents.append("\t)\n")

    return "".join(fluents)

  def generatePddlGoal(self, gameState, features):
    """
    Function for creating PDDL goals for the problem file.
    """
    print('======New Action========')

    problemObjective = None
    gameTimeLeft = gameState.data.timeleft
    pacmanPos = gameState.getAgentPosition(self.index)
    foods = self.cAIns.getFood(gameState).asList()
    capsules = self.cAIns.getCapsules(gameState)
    thres = features["threshold"]

    # Get History of locations
    if len(self.history.list) < 8:
      self.history.push(pacmanPos)
    elif len(self.history.list) == 8:
      print(self.history.list)
      count = Counter(self.history.list).most_common()
      self.stuck = True if count and count[0][1] >= 3 and count[0][1] == count[1][1] else False
      if self.stuck: print('I am Stuck! Moving Out of Sight!')
      self.history.pop()
      self.history.push(pacmanPos)

    # Get Food Eaten Calculation based on current Game Score
    newScore = self.cAIns.getScore(gameState)
    if newScore > self.currScore:
      self.foodEaten += newScore - self.currScore
      self.currScore = newScore
    else:
      self.currScore = newScore

    goal = list()
    goal.append('\t(:goal (and\n')

    # Find if a ghost is in the proximity of pacman
    enemies = [gameState.getAgentState(i) for i in self.cAIns.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    ghostDistance = 999999
    scaredTimer = 99999
    if len(ghosts) > 0:
      ghostDistance, scaredTimer = self.getGhostDistanceAndTimers(pacmanPos, ghosts)
      print("ghosts visible: 0.6")
      thres = features["threshold"]
    else:
      print("ghosts not visible: 1.0")
      thres = 1

    print(f'Pacman at {pacmanPos}')

    if features["problemObjective"] is None:
      closestHome, closestCap = self.compareCapsuleAndHomeDist(gameState, pacmanPos)
      # gameTimeLeft decrease by 4 for every 1 move - anticipate + come back
      if ((closestHome * 4) + 20) >= gameTimeLeft:
        print("run back home - time running out")
        problemObjective = self.goBackHomeHardObjective(gameState, goal, pacmanPos)
      # if ghost is really close RUN to capsule if any or RUN BACK!
      elif self.stuck:
        problemObjective = self.goBackStartObjective(goal)
      elif ghostDistance <= 3 and scaredTimer <= 3:
        flag = self.getFlag(gameState, thres, foods)
        if not flag and len(capsules) > 0:
          problemObjective = self.addEatCapsuleObjective(goal)
        else:
          problemObjective = self.goBackHomeHardObjective(gameState, goal, pacmanPos)
      else:
        # not being chased by ghost
        # or ghost is scared
        flag = self.getFlag(gameState, thres, foods)
        if len(foods) > 2 and not flag:
          problemObjective = self.eatFoodObjective(goal)
        else:
          problemObjective = self.goBackHomeHardObjective(gameState, goal, pacmanPos)
    else:
      # fallback goals
      problemObjective = self.tryFallBackGoals(goal, features, gameState, pacmanPos)

    goal.append('\t))\n')
    return ("".join(goal), problemObjective)

  def getGhostDistanceAndTimers(self, pacmanPos, ghosts):
    dists = [self.cAIns.getMazeDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    timers = [ghost.scaredTimer for ghost in ghosts]
    ghostDistance = min(dists)
    scaredTimer = min(timers)
    print(f'Ghost Alert with Dist: {ghostDistance} | scaredTimer: {scaredTimer}')
    return (ghostDistance, scaredTimer)

  def getBoundaryX(self, gameState):
    return gameState.data.layout.width / 2 - 1 if self.cAIns.red else gameState.data.layout.width / 2

  def compareCapsuleAndHomeDist(self, gameState, pacmanPos):
    x = self.getBoundaryX(gameState)

    if len(self.cAIns.getCapsules(gameState)) > 0:
      closestCap = min([self.cAIns.getMazeDistance(pacmanPos, cap) for cap in self.cAIns.getCapsules(gameState)])
      closestHome = min([self.cAIns.getMazeDistance(pacmanPos, pos) for pos in self.homePos if pos[0] == x])
    else:
      closestHome = 1
      closestCap = 10

    return (closestHome, closestCap)

  def getFlag(self, gameState, threshold, foods):
    foodLeft = len(self.masterFoods) - self.foodEaten
    foodCaryingPer = (foodLeft - len(foods)) / foodLeft
    flag = True if foodCaryingPer > threshold else False
    return flag

  def addEatCapsuleObjective(self, goal):
    print('Going to Capsule')
    goal.append(f'\t\t(capsule-eaten)\n')
    return "EAT_CAPSULE"

  def goBackStartObjective(self, goal):
    print('Going to Start Position')
    goal.append(f'\t\t(at-pacman cell{self.start[0]}_{self.start[1]})\n')
    return "GO_START"

  def goBackHomeHardObjective(self, gameState, goal, pacmanPos):
    print('Going Back Home [Running Away]')
    x = self.getBoundaryX(gameState)
    if pacmanPos in self.homePos:
      goal.append(f'\t\t(at-pacman cell{self.start[0]}_{self.start[1]})\n')
    else:
      goal.append('\t\t(or\n')
      for pos in self.homePos:
        if pos[0] == x:
          goal.append(f'\t\t\t(at-pacman cell{pos[0]}_{pos[1]})\n')
      goal.append('\t\t)\n')
    return "COME_BACK_HOME"

  def eatFoodObjective(self, goal):
    print('Going To Eat Food')
    goal.append(f'\t\t(carrying-food)\n')
    return "EAT_FOOD"

  def tryFallBackGoals(self, goal, features, gameState, pacmanPos):
    if features["problemObjective"] == "COME_BACK_HOME":
      print('Going Back Home [FALLBACK]')
      return self.goBackHomeHardObjective(gameState, goal, pacmanPos)
    elif features["problemObjective"] == "DIE":
      print('Die for respawn [FALLBACK]')
      goal.append(f'\t\t(die)\n')
      return "DIE"

  def generatePddlProblem(self, gameState, features):
    """
    Generates a file for Creating PDDL problem file for current state.
    """
    problem = list()
    problem.append(f'(define (problem p{self.index}-pacman)\n')
    problem.append('\t(:domain pacman)\n')
    problem.append(self.pddlObject)
    problem.append(self.generatePddlFluent(gameState, features))
    goalStatement, goalObjective = self.generatePddlGoal(gameState, features)
    problem.append(goalStatement)
    problem.append(')')

    problem_file = open(f"{CD}/pacman-problem-{self.index}.pddl", "w")
    problem_statement = "".join(problem)
    problem_file.write(problem_statement)
    problem_file.close()
    return (f"pacman-problem-{self.index}.pddl", goalObjective)

#######################
## Metric FF Planner ##
#######################

class PlannerFF():

  def __init__(self, domain_file, problem_file):
    self.domain_file = domain_file
    self.problem_file = problem_file

  def run_planner(self):
    cmd = [f"{FF_EXECUTABLE_PATH}",
           "-o", self.domain_file,
           "-f", f"{CD}/{self.problem_file}"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
    return result.stdout.splitlines() if result.returncode == 0 else None

  def parse_solution(self, output):
    newX = -1
    newY = -1
    targetPlan = None
    try:
      if output is not None:
        # parse standard output
        plan = self.parse_ff_output(output)
        if plan is not None:
          # pick first plan
          targetPlan = plan[0]
          if 'reach-goal' not in targetPlan:
            targetPlan = targetPlan.split(' ')
            if "move" in targetPlan[0].lower():
              start = targetPlan[1].lower()
              end = targetPlan[2].lower()
              coor = self.get_coor_from_loc(end)
              newX = int(coor[0])
              newY = int(coor[1])
            else:
              start = targetPlan[1].lower()
              coor = self.get_coor_from_loc(start)
              newX = int(coor[0])
              newY = int(coor[1])
          else:
            print('Already in goal')
        else:
          print('No plan!')
    except:
      print('Something wrong happened with PDDL parsing')

    return ((newX, newY), targetPlan)

  def parse_ff_output(self, lines):
    plan = []
    for line in lines:
      search_action = re.search(r'\d: (.*)$', line)
      if search_action:
        plan.append(search_action.group(1))

      # Empty Plan
      if line.find("ff: goal can be simplified to TRUE.") != -1:
        return []
      # No Plan
      if line.find("ff: goal can be simplified to FALSE.") != -1:
        return None

    if len(plan) > 0:
      return plan
    else:
      print('should never have ocurred!')
      return None

  def get_legal_action(self, myPos, plannerPos):
    posX, posY = myPos
    plannerX, plannerY = plannerPos
    if plannerX == posX and plannerY == posY:
      return "Stop"
    elif plannerX == posX and plannerY == posY + 1:
      return "North"
    elif plannerX == posX and plannerY == posY - 1:
      return "South"
    elif plannerX == posX + 1 and plannerY == posY:
      return "East"
    elif plannerX == posX - 1 and plannerY == posY:
      return "West"
    else:
      # no plan found
      print('Planner Returned Nothing.....')
      return "Stop"

  def get_coor_from_loc(self, loc):
    return loc.split("cell")[1].split("_")

########################
# Defensive PDDL Agent #
########################

class DefensivePDDLAgent(CaptureAgent):
  """
  A Classical PDDL approach based Defensive Agent
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''
    self.createGhostDomain()
    self.start = gameState.getAgentPosition(self.index)
    self.pddlFluentGrid = self.generatePDDLFluentStatic(gameState)
    self.pddlObject = self.generatePddlObject(gameState)
    self.boundaryPos = self.getBoundaryPos(gameState, 1)
    self.masterFoods = self.getFoodYouAreDefending(gameState).asList()
    self.masterCapsules = self.getCapsulesYouAreDefending(gameState)
    self.history = Queue()
    self.stuck = False

  def createGhostDomain(self):
    ghost_domain_file = open(GHOST_DOMAIN_FILE, "w")
    domain_statement = """
      (define (domain ghost)

          (:requirements
              :typing
              :negative-preconditions
          )

          (:types
              invaders cells
          )

          (:predicates
              (cell ?p)

              ;pacman's cell location
              (at-ghost ?loc - cells)

              ;food cell location
              (at-invader ?i - invaders ?loc - cells)

              ;Indicated if a cell location has a capsule
              (has-capsule ?loc - cells)

              ;connects cells
              (connected ?from ?to - cells)

          )

          ; move ghost towards the goal state of invader
          (:action move
              :parameters (?from ?to - cells)
              :precondition (and 
                  (at-ghost ?from)
                  (connected ?from ?to)
              )
              :effect (and
                          ;; add
                          (at-ghost ?to)
                          ;; del
                          (not (at-ghost ?from))       
                      )
          )

          ; kill invader
          (:action kill-invader
              :parameters (?loc - cells ?i - invaders)
              :precondition (and 
                              (at-ghost ?loc)
                              (at-invader ?i ?loc)
                            )
              :effect (and
                          ;; add

                          ;; del
                          (not (at-invader ?i ?loc))
                      )
          )
      )
      """
    ghost_domain_file.write(domain_statement)
    ghost_domain_file.close()

  def generatePddlObject(self, gameState):
    """
    Function for creating PDDL objects for the problem file.
    """

    # Get Cell Locations without walls and Food count for object setup.
    allPos = gameState.getWalls().asList(False)
    invader_len = len(self.getOpponents(gameState))

    # Create Object PDDl line definition of objects.
    objects = list()
    cells = [f'cell{pos[0]}_{pos[1]}' for pos in allPos]
    cells.append("- cells\n")
    invaders = [f'invader{i+1}' for i in range(invader_len)]
    invaders.append("- invaders\n")

    objects.append("\t(:objects \n")
    objects.append(f'\t\t{" ".join(cells)}')
    objects.append(f'\t\t{" ".join(invaders)}')
    objects.append("\t)\n")

    return "".join(objects)

  def getBoundaryPos(self, gameState, span=4):
    """
    Get Boundary Position for Home to set as return when chased by ghost
    """
    layout = gameState.data.layout
    x = layout.width / 2 - 1 if self.red else layout.width / 2
    xSpan = [x - i for i in range(span)] if self.red else [x + i for i in range(span)]
    walls = gameState.getWalls().asList()
    homeBound = list()
    for x in xSpan:
      pos = [(int(x), y) for y in range(layout.height) if (x, y) not in walls]
      homeBound.extend(pos)
    return homeBound

  def generatePDDLFluentStatic(self, gameState):
    # Set Adjacency Position
    allPos = gameState.getWalls().asList(False)
    connected = list()
    for pos in allPos:
      if (pos[0] + 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]+1}_{pos[1]})\n')
      if (pos[0] - 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]-1}_{pos[1]})\n')
      if (pos[0], pos[1] + 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]+1})\n')
      if (pos[0], pos[1] - 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1]-1})\n')

    return "".join(connected)

  def generatePddlFluent(self, gameState):
    """
    Function for creating PDDL fluents for the problem file.
    """

    # Set Self Position
    pacmanPos = gameState.getAgentPosition(self.index)
    at_ghost = f'\t\t(at-ghost cell{pacmanPos[0]}_{pacmanPos[1]})\n'

    # Set Invader(s) positions
    has_invaders = list()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    for i, invader in enumerate(invaders):
      invaderPos = invader.getPosition()
      has_invaders.append(f'\t\t(at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')

    # Set Capsule Position
    capsules = self.getCapsulesYouAreDefending(gameState)
    has_capsule = [f'\t\t(has-capsule cell{capsule[0]}_{capsule[1]})\n' for capsule in capsules]

    fluents = list()
    fluents.append("\t(:init \n")
    fluents.append(at_ghost)
    fluents.append("".join(has_invaders))
    fluents.append("".join(has_capsule))
    fluents.append(self.pddlFluentGrid)
    fluents.append("\t)\n")

    return "".join(fluents)

  def generatePddlGoal(self, gameState):
    """
    Function for creating PDDL goals for the problem file.
    """
    goal = list()
    goal.append('\t(:goal (and\n')

    myPos = gameState.getAgentPosition(self.index)
    foods = self.getFoodYouAreDefending(gameState).asList()

    # Get History of locations
    if len(self.history.list) < 8:
      self.history.push(myPos)
    elif len(self.history.list) == 8:
      print(self.history.list)
      count = Counter(self.history.list).most_common()
      self.stuck = True if count and count[0][1] >= 3 and count[0][1] == count[1][1] else False
      if self.stuck: print('I am Stuck def! Moving Out of Sight!')
      self.history.pop()
      self.history.push(myPos)

    if self.stuck:
      invaders = []
    else:
      # Find Invaders and set their location.
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

      for i, invader in enumerate(invaders):
        invaderPos = invader.getPosition()
        goal.append(f'\t\t(not (at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')

    # If No Invaders are detected (Seen 5 steps)
    if not invaders:

      # If Food has not been eaten, Guard the Capsules or Foods
      if len(self.masterFoods) == len(foods):

        self.boundaryPos = [] if myPos in self.boundaryPos else self.boundaryPos
        if myPos not in self.boundaryPos and len(self.boundaryPos) > 0:
          goal.append('\t\t(or\n')
          for pos in self.boundaryPos:
            goal.append(f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n')
          goal.append('\t\t)\n')

        elif myPos not in self.masterCapsules and len(self.getCapsulesYouAreDefending(gameState)) > 0:
          capsules = self.getCapsulesYouAreDefending(gameState)
          if len(capsules) > 1:
            goal.append('\t\t(or\n')
            goal.extend([f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n' for pos in capsules])
            goal.append('\t\t)\n')
          else:
            pos = capsules[0]
            goal.append(f'\t\t(at-ghost cell{pos[0]}_{pos[1]})\n')

        else:
          goal.append('\t\t(or\n')
          for food in foods:
            if myPos != food:
              goal.append(f'\t\t\t(at-ghost cell{food[0]}_{food[1]})\n')
          goal.append('\t\t)\n')

      # If Food have been eaten Rush to the food location.
      else:
        eatenFood = list(set(self.masterFoods) - set(foods))
        if myPos in eatenFood:
          self.masterFoods = foods
          eatenFood.remove(myPos)
        if len(eatenFood) > 1:
          goal.append('\t\t(or\n')
          goal.extend([f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n' for pos in eatenFood])
          goal.append('\t\t)\n')
        elif len(eatenFood) == 1:
          pos = eatenFood[0]
          goal.append(f'\t\t(at-ghost cell{pos[0]}_{pos[1]})\n')
        else:
          goal.append(f'\t\t(at-ghost cell{self.start[0]}_{self.start[1]})\n')

    goal.append('\t))\n')
    return "".join(goal)

  def generatePddlProblem(self, gameState):
    """
    Generates a file for Creating PDDL problem file for current state.
    """
    problem = list()
    problem.append(f'(define (problem p{self.index}-ghost)\n')
    problem.append('\t(:domain ghost)\n')
    problem.append(self.pddlObject)
    problem.append(self.generatePddlFluent(gameState))
    problem.append(self.generatePddlGoal(gameState))
    problem.append(')')

    problem_file = open(f"{CD}/ghost-problem-{self.index}.pddl", "w")
    problem_statement = "".join(problem)
    problem_file.write(problem_statement)
    problem_file.close()
    return f"ghost-problem-{self.index}.pddl"

  def chooseAction(self, gameState):
    agentPosition = gameState.getAgentPosition(self.index)
    problem_file = self.generatePddlProblem(gameState)
    planner = PlannerFF(GHOST_DOMAIN_FILE, problem_file)
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    action = planner.get_legal_action(agentPosition, plannerPosition)
    # print(f'Action Planner: {action}')
    # actions = gameState.getLegalActions(self.index)
    return action