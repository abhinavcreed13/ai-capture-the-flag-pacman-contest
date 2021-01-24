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

# Attribution Information
# Implementation of Ford-Fulkerson algorithm:
# Credits to: https://github.com/bigbighd604/Python/blob/master/graph/Ford-Fulkerson.py

# Implementation of Approximate Q Learning Agent
# Inspired from CS 188 Project 3: https://inst.eecs.berkeley.edu/~cs188/su20/project3/

# Implementation of ContextManager Signal Timer
# Credits to: https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
import sys, os, platform
import re
import subprocess
import math
from util import nearestPoint, Queue
from collections import Counter
import signal
from contextlib import contextmanager
from capture import GameState
from distanceCalculator import Distancer


# FF real path
CD = os.path.dirname(os.path.abspath(__file__))
# FF_EXECUTABLE_PATH = "{}/../../bin/ff".format(CD)

FF_EXECUTABLE_PATH = "ff"

PACMAN_DOMAIN_FILE = f"{CD}/pacman-domain.pddl"
GHOST_DOMAIN_FILE = f"{CD}/ghost-domain.pddl"

RED_ATTACKERS = 0
BLUE_ATTACKERS = 0
RED_DEFENDERS = 0
BLUE_DEFENDERS = 0

AGENT_1_FOOD_EATEN = 0
AGENT_2_FOOD_EATEN = 0
TOTAL_FOODS = 0
TOTAL_FOOD_COLLECTED = 0

AGENT_1_CLOSEST_FOOD = None
AGENT_2_CLOSEST_FOOD = None
AGENT_1_POSITION = None
AGENT_2_POSITION = None
AGENT_1_MODE = None
AGENT_2_MODE = None
AGENT_1_AREA = None
AGENT_2_AREA = None
FOOD_DEPTHS = None
ANTICIPATER = []
DEPTH_LIMIT = False

DEBUG = False
DEFENSIVE_BLOCKING_MODE = False

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MasterPDDLAgent', second='MasterPDDLAgent'):
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

##############################
# Helper Classes & Functions #
##############################

class Edge(object):
  def __init__(self, u, v, w):
    self.source = u
    self.target = v
    self.capacity = w

  def __repr__(self):
    return f"{self.source}->{self.target}:{self.capacity}"

  def __eq__(self, other):
    return self.source == other.source and self.target == other.target

  def __hash__(self):
    return hash(f"{self.source}->{self.target}:{self.capacity}")

class FlowNetwork(object):
  def __init__(self):
    self.adj = {}
    self.flow = {}

  def AddVertex(self, vertex):
    self.adj[vertex] = []

  def GetEdges(self, v):
    return self.adj[v]

  def AddEdge(self, u, v, w=0):
    if u == v:
      raise ValueError("u == v")
    edge = Edge(u, v, w)
    redge = Edge(v, u, w)
    edge.redge = redge
    redge.redge = edge
    self.adj[u].append(edge)
    self.adj[v].append(redge)
    # Intialize all flows to zero
    self.flow[edge] = 0
    self.flow[redge] = 0

  def FindPath(self, start, goal, path=[]):
    """
    Run a BFS as a inside algorithm to find the path in a Max Flow graph
    """
    node, pathCost = start, 0
    frontier = util.Queue()
    visited = set()

    if start == goal:
      return path

    while node != goal:
      successors = [(edge.target, edge) for edge in self.GetEdges(node)]
      for successor, edge in successors:
        residual = edge.capacity - self.flow[edge]
        intPath = (edge, residual)
        if residual > 0 and not intPath in path and intPath not in visited:
          visited.add(intPath)
          frontier.push((successor, path + [(edge, residual)], pathCost + 1))

      if frontier.isEmpty():
        return None
      else:
        node, path, pathCost = frontier.pop()

    return path

  def MaxFlow(self, source, target):
    """
    Find the MaxFlow + a variable to keep edges which are reachable from sink point T.
    """
    targetEdges = {}
    path = self.FindPath(source, target)
    while path is not None:
      targetEdges[path[0]] = path
      flow = min(res for edge, res in path)
      for edge, _ in path:
        self.flow[edge] += flow
        self.flow[edge.redge] -= flow

      path = self.FindPath(source, target)
    maxflow = sum([self.flow[edge] for edge in self.GetEdges(source)])
    return maxflow, targetEdges

  def FindBottlenecks(self, source, target):
    """
    Find Bottleneck position using the Ford Fulkerson Algorithm. The idea is, We have the edges
    which are reachable from  sink point T. Get the edges which are reachable from source S.
    The bottleneck positions are nodes connecting these both sets.
    """
    _, targetEdges = self.MaxFlow(source, target)
    paths = targetEdges.values()

    bottlenecks = []
    for path in paths:
      for edge, _ in path:
        if self.FindPath(source, edge.target) is None:
          bottlenecks.append(edge.source)
          break
    return bottlenecks

class SortedEdges(dict):
  '''
  Edges dict for Max Flow Problem
  '''

  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)

  def __getitem__(self, key):
    return dict.__getitem__(self, tuple(sorted(key)))

  def __setitem__(self, key, val):
    return dict.__setitem__(self, tuple(sorted(key)), val)

  def __contains__(self, key):
    return dict.__contains__(self, tuple(sorted(key)))

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    # signal.alarm(seconds)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        # signal.alarm(0)
        signal.setitimer(signal.ITIMER_REAL, 0)

##############
# Base Agent #
##############

class BaseAgent(CaptureAgent):
  """
    Helper functions for all Agents
  """
  ## Bayesian Inference Functions Starts
  def initalize(self, enemy, startPos):
    """
    Uniformly initialize belief distributions for opponent positions.
    """
    self.obs[enemy] = util.Counter()
    self.obs[enemy][startPos] = 1.0

  def setTruePos(self, enemy, pos):
    """
    Fix the position of an opponent in an agent's belief distributions.
    """
    trueObs = util.Counter()
    trueObs[pos] = 1.0
    self.obs[enemy] = trueObs

  def elapseTime(self, enemy, gameState):
    """
    Elapse belief distributions for an agent's position by one time step.
    Assume opponents move randomly, but also check for any food lost from
    the previous turn.
    """
    possiblePos = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    allObs = util.Counter()
    for prevPos, prevProb in self.obs[enemy].items():
      newObs = util.Counter()
      for pos in possiblePos(prevPos[0], prevPos[1]):
        if pos in self.legalPositions:
          newObs[pos] = 1.0
      newObs.normalize()
      for newPos, newProb in newObs.items():
        allObs[newPos] += newProb * prevProb

    foods = self.getFoodYouAreDefending(gameState).asList()
    prevFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList() \
      if self.getPreviousObservation() is not None else list()
    if len(foods) < len(prevFoods):
      eatenFood = list(set(foods) - set(prevFoods))
      for food in eatenFood:
        allObs[food] = 1.0 / len(self.getOpponents(gameState))

    self.obs[enemy] = allObs

  def observe(self, enemy, gameState):
    """
    Updates beliefs based on the distance observation and Pacman's
    position.
    """
    allnoise = gameState.getAgentDistances()
    noisyDistance = allnoise[enemy]
    myPos = gameState.getAgentPosition(self.index)
    teamPos = [gameState.getAgentPosition(team) for team in self.getTeam(gameState)]
    allObs = util.Counter()

    for pos in self.legalPositions:
      teamDist = [team for team in teamPos if util.manhattanDistance(team, pos) <= 5]
      if teamDist:
        allObs[pos] = 0.0
      else:
        trueDistance = util.manhattanDistance(myPos, pos)
        posProb = gameState.getDistanceProb(trueDistance, noisyDistance)
        allObs[pos] = posProb * self.obs[enemy][pos]

    if allObs.totalCount():
      allObs.normalize()
      self.obs[enemy] = allObs
    else:
      self.initalize(enemy, gameState.getInitialAgentPosition(enemy))

  def approxPos(self, enemy):
    """
    Return the highest probably  enemy position
    """
    return self.obs[enemy].argMax()

  def getAnticipatedGhosts(self, gameState):
    anticipatedGhosts = []
    # Bayesian Inference Update Beliefs Function
    # =============================================================
    for enemy in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(enemy)
      if not pos:
        self.elapseTime(enemy, gameState)
        self.observe(enemy, gameState)
      else:
        self.setTruePos(enemy, pos)

    # Display The Distribution On the board
    # self.displayDistributionsOverPositions(self.obs.values())

    for enemy in self.getOpponents(gameState):
      anticipatedPos = self.approxPos(enemy)
      enemyGameState = gameState.getAgentState(enemy)
      # if not enemyGameState.isPacman and enemyGameState.scaredTimer <= 3:
      anticipatedGhosts.append((enemyGameState, anticipatedPos))

      # # Sanity Check
      # if enemyGameState.isPacman:
      #   if DEBUG: print(f'Enemy is Pacman at {anticipatedPos}')
      # else:
      #   if DEBUG: print(f'Enemy is Ghost at {anticipatedPos}')

    # if DEBUG: print('===========Anticipator==============')

    return anticipatedGhosts
    # =============================================================
  ## Bayesian Inference Functions Ends.

  ## Finding bottleneck functions starts

  def findBottlenecks(self, gameState, defMode=True):

    if defMode:
      startingPos = self.boundaryPos
      endingPos = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
      network, source = self.getFlowNetwork(gameState, startingPos=startingPos)
    else:
      startingPos = self.enemyBoundaryPos
      endingPos = self.getFood(gameState).asList() + self.getCapsules(gameState)
      network, source = self.getFlowNetwork(gameState, startingPos=startingPos, defMode=defMode)

    bottleneckCounter = Counter()
    bottleneckPosition = dict()

    for pos in endingPos:
      bottlenecks = network.FindBottlenecks(source, pos)
      if len(bottlenecks) == 1:
        bottleneckCounter[bottlenecks[0]] += 1
        if bottlenecks[0] in bottleneckPosition.keys():
          bottleneckPosition[bottlenecks[0]].append(pos)
        else:
          bottleneckPosition[bottlenecks[0]] = [pos]

      for edge in network.flow:
        network.flow[edge] = 0

    return bottleneckCounter, bottleneckPosition

  def getFlowNetwork(self, gameState, startingPos=None, endingPos=None, defMode=True):
    '''
    Returns the flow network using the Ford Fulkerson Algo.
    '''
    source = (-1, -1)
    sink = (-2, -2)

    walls = gameState.getWalls()
    legalPositions = gameState.getWalls().asList(False)
    if self.red:
      atHome = lambda x: (x < walls.width / 2) if defMode else lambda x: (x >= walls.width / 2)
    else:
      atHome = lambda x: (x >= walls.width / 2) if defMode else lambda x: (x < walls.width / 2)  
      
    actionPos = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    possiblePos = [(x, y) for x, y in legalPositions if (x, y) in legalPositions and (atHome(x))]

    # Make source and sink

    network = FlowNetwork()

    # Add all vertices
    for pos in possiblePos:
      network.AddVertex(pos)
    network.AddVertex(source)
    network.AddVertex(sink)

    # Add normal edges
    edges = SortedEdges()
    for pos in possiblePos:
      newPos = actionPos(pos[0], pos[1])
      for move in newPos:
        if move in possiblePos:
          edges[(pos, move)] = 1

    # Add edges from source
    for pos in startingPos or []:
      edges[(source, pos)] = float('inf')

    # Add edges from foods/capsules
    for pos in endingPos or []:
      edges[(pos, sink)] = float('inf')

    for edge in edges:
      network.AddEdge(edge[0], edge[1], edges[edge])

    ret = (network,)

    if startingPos is not None:
      ret = ret + (source,)
    if endingPos is not None:
      ret = tuple(ret) + (sink,)

    return ret

  def getTopkBottleneck(self, gameState, k, defMode=True):
    bottleneckCounter, bottleneckPos = self.findBottlenecks(gameState, defMode=defMode)
    return bottleneckCounter.most_common(k), bottleneckPos

  def flipPos(self, gameState, pos):
    walls = gameState.getWalls()
    width = walls.width - 1
    height = walls.height - 1
    return (width - pos[0], height - pos[1])

  def convertBottleNeckOff(self, gameState, bottleNeckCount, bottleNeckPos):
    offBottleNeckCount = list()
    offBottleNeckPos = dict()
    for _, tup in enumerate(bottleNeckCount):
      flip = self.flipPos(gameState, tup[0])
      offBottleNeckCount.append((flip, tup[1]))
    for _, pos in enumerate(bottleNeckPos.keys()):
      flipPos = self.flipPos(gameState, pos)
      flipFoods = list()
      for food in bottleNeckPos[pos]:
        flipFoods.append(self.flipPos(gameState, food))
      offBottleNeckPos[flipPos] = flipFoods

    return offBottleNeckCount, offBottleNeckPos

  def getDepthPos(self, gameState, bottleNeckPos, offBottleNeckCount):
    capsules = self.getCapsules(gameState)
    foodDepth = dict()
    for pos, count in offBottleNeckCount:
        coveredPos = bottleNeckPos[pos]
        for food in coveredPos:
          if food not in capsules:
            foodDepth[food] = self.getMazeDistance(pos, food) * 2
    foods = self.getFood(gameState).asList()
    foodsRemaining = list(set(foods) - set(foodDepth.keys()))
    for food in foodsRemaining:
      foodDepth[food] = 0
    foodDepth = {k: v for k, v in sorted(foodDepth.items(), key=lambda item: item[1])}
    return foodDepth

  ## Finding bottleneck functions end

  ## Helper Functions starts
  def isEnemyEnteredTerritory(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    return len(enemyHere) > 0

  def numberOfInvaders(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    return len(enemyHere)

  def numberOfGhosts(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghostsHere = [a for a in enemies if not a.isPacman]
    return len(ghostsHere)

  def allInvadersKilled(self, myCurrPos, gameState, nextAction):
    currentInvaderDistance = None
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myCurrPos, a.getPosition()) for a in invaders]
      currentInvaderDistance = min(dists)

    if currentInvaderDistance == 1:
      successor = gameState.generateSuccessor(self.index, nextAction)
      myNextState = successor.getAgentState(self.index)
      myNextPos = myNextState.getPosition()
      nextEnemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      nextInvaders = [a for a in nextEnemies if a.isPacman and a.getPosition() != None]
      if len(nextInvaders) == 0:
        return True

    return False

  def getBoundaryX(self, gameState):
    return gameState.data.layout.width / 2 - 1 if self.red else gameState.data.layout.width / 2

  def getBoundaryPos(self, gameState, span=4, defMode=True):
    """
    Get Boundary Position for Home to set as return when chased by ghost
    """
    flag = self.red if defMode else (not self.red)
    layout = gameState.data.layout
    x = layout.width / 2 - 1 if flag else layout.width / 2
    enemy = 1 if flag else -1
    xSpan = [x - i for i in range(span)] if flag else [x + i for i in range(span)]
    walls = gameState.getWalls().asList()
    homeBound = list()
    for x in xSpan:
      pos = [(int(x), y) for y in range(layout.height) if (x, y) not in walls and (x + enemy, y) not in walls]
      homeBound.extend(pos)
    return homeBound

  def checkIfStuck(self, myCurrPos):
    """
        Check if agent is stuck
        Make sure to add: self.history = Queue() in initialState of Agent
    """
    if len(self.history.list) < 8:
      self.history.push(myCurrPos)
    elif len(self.history.list) == 8:
      count = Counter(self.history.list).most_common()
      try:
        self.stuck = True if count and count[0][1] >= 3 and count[0][1] == count[1][1] else False
        if self.stuck:
          if DEBUG:
            print('I am Stuck! Needs Approximation')
          self.approximationMode = True
        self.history.pop()
        self.history.push(myCurrPos)
      except:
        if DEBUG: print("!! STUCK PROBLEM - Reinit History !!")
        if DEBUG: print(self.history.list)
        self.stuck = False
        self.history = Queue()

  def isSurrounded(self, gameState):
    """
      Finding corner foods: Experimental
    """
    walls = gameState.getWalls()
    foods = self.masterFoods
    cornerFoods = list()

    for food in foods:
      x, y = food
      possiblePos = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
      count = 0
      for pos in possiblePos:
        if walls[pos[0]][pos[1]] or (pos[0], pos[1]) in cornerFoods:
          count += 1

      if count >= 3:
        cornerFoods.append(food)

    return cornerFoods

  def crossedBoundary(self, myPos, gameState):
    if self.red:
      targetBoundaryPoint = self.agent.getBoundaryX(gameState) + 1
      if myPos[0] > targetBoundaryPoint:
        return True
    else:
      targetBoundaryPoint = self.agent.getBoundaryX(gameState) - 1
      if myPos[0] < targetBoundaryPoint:
        return True

  def dangerousEnemies(our_agent_index: int, gameState: GameState):
    '''
      collect enemies that are capable of eating out pacman
    '''
    enemy_indexes = gameState.getBlueTeamIndices() if gameState.isOnRedTeam(
      our_agent_index) else gameState.getRedTeamIndices()
    enemy_states = [(gameState.getAgentState(enemy_index), enemy_index) for enemy_index in enemy_indexes]
    dangerous_enemies = [(enemy_state.getPosition(), enemy_index) for enemy_state, enemy_index in enemy_states \
                         if (not enemy_state.isPacman) \
                         and (enemy_state.scaredTimer == 0) \
                         and (enemy_state.getPosition() is not None) \
                         ]
    return dangerous_enemies

  def edibleEnemies(our_agent_index: int, gameState: GameState):
    '''
      Return the index and location of enemy agent that could be eaten
    '''
    if gameState.getAgentState(our_agent_index).scaredTimer != 0:
      # our agent is scared and cannot eat anything
      return []

    enemy_indexes = gameState.getBlueTeamIndices() if gameState.isOnRedTeam(
      our_agent_index) else gameState.getRedTeamIndices()
    enemy_states = [(gameState.getAgentState(enemy_index), enemy_index) for enemy_index in enemy_indexes]
    eddible_enemies = [(enemy_state.getPosition(), enemy_index) for enemy_state, enemy_index in enemy_states \
                       if (enemy_state.isPacman) \
                       and (enemy_state.getPosition() is not None) \
                       ]
    return eddible_enemies

  def simulateEnemyChase(our_agent_index: int, gameState: GameState, distancer: Distancer):
    '''
    Takes a game state and an agent.
    If there are enemys that could eat out_agent, updates the game state to reflect this
    args:
        our_agent_index: int
        gameState: GameState that you want to simulate enemy move
        distancer: can be found automatically on capture agent captureagent.distancer
    '''
    # NOTE: this is likely to be a very expensive method as it currently works with an entire game state object
    # calc is only applicable when we are a pacman
    if not gameState.getAgentState(our_agent_index).isPacman:
      return gameState

    dangerous_enemies = dangerousEnemies(our_agent_index, gameState)
    my_pos = gameState.getAgentPosition(our_agent_index)
    close_enemies = [(enemy_pos, enemy_index) for enemy_pos, enemy_index in dangerous_enemies if
                     distancer.getDistance(enemy_pos, my_pos) <= 6]

    # update close note scare enemies to chase pacman
    enemy_best_move = {}  # key: enemy_index value: best_action
    for enemy_pos, enemy_index in close_enemies:
      enemy_actions = gameState.getLegalActions(enemy_index)
      # TODO: make it select between the best options randomly
      _, best_action = min(
        [(distancer.getDistance(Actions.getSuccessor(enemy_pos, action), my_pos), action) for action in
         enemy_actions])
      if min(
          [distancer.getDistance(Actions.getSuccessor(enemy_pos, action), my_pos) for action in enemy_actions]) == 0:
        1 + 1
      enemy_best_move[enemy_index] = best_action

    # apply action to the state
    for enemy_index, best_action in enemy_best_move.items():
      gameState = gameState.generateSuccessor(enemy_index, best_action)

    return gameState

    ## Helper Functions ends

##################
## Master Agent ##
##################

class MasterPDDLAgent(BaseAgent):

  def registerInitialState(self, gameState):
    global TOTAL_FOODS, FOOD_DEPTHS
    CaptureAgent.registerInitialState(self, gameState)
    self.TIME_LIMIT = 0.9
    # self.TIME_LIMIT = 9
    self.AGENT_MODE = None
    self.start = gameState.getAgentPosition(self.index)
    TOTAL_FOODS = len(self.getFood(gameState).asList())
    self.currScore = self.getScore(gameState)

    # Bayesian Inference
    self.legalPositions = gameState.getWalls().asList(False)
    self.obs = {}
    for enemy in self.getOpponents(gameState):
      self.initalize(enemy, gameState.getInitialAgentPosition(enemy))

    # identify stuck & switch
    self.history = Queue()
    self.stuck = False
    self.approximationMode = False
    self.lastFoodEaten = 0
    self.movesApproximating = 0
    self.approximationThresholdAgent1 = 10
    self.approximationThresholdAgent2 = 10

    # agents pool
    # create attacking agent
    attackingAgent = offensivePDDLAgent(self.index)
    attackingAgent.registerInitialState(gameState)
    attackingAgent.observationHistory = self.observationHistory

    # create defending agent
    defendingAgent = defensivePDDLAgent(self.index)
    defendingAgent.registerInitialState(gameState)
    defendingAgent.observationHistory = self.observationHistory

    # create approximating attack agent
    approxAttackAgent = approxQLearningOffense(self.index)
    approxAttackAgent.registerInitialState(gameState)
    approxAttackAgent.observationHistory = self.observationHistory

    self.agentsPool = {
      "ATTACKING_AGENT": attackingAgent,
      "DEFENDING_AGENT": defendingAgent,
      "APPROXIMATING_AGENT": approxAttackAgent
    }

    # get current agent
    self.agent = self.getAgent(gameState)

    # Bottleneck Calculation
    self.foodBottleNeckThreshold = 0.2
    self.boundaryPos = self.getBoundaryPos(gameState, 1)
    self.enemyBoundaryPos = self.getBoundaryPos(gameState, span=1, defMode=False)
    self.masterCapsules = self.getCapsulesYouAreDefending(gameState)
    self.offensiveCapsules = self.getCapsules(gameState)
    self.masterFoods = self.getFoodYouAreDefending(gameState).asList()
    tic = time.perf_counter()
    self.bottleNeckCount, self.bottleNeckPos = self.getTopkBottleneck(gameState, 5)
    # self.offBottleNeckCount, self.offBottleNeckPos = self.convertBottleNeckOff(gameState, self.bottleNeckCount,
    #                                                                            self.bottleNeckPos)
    # self.offBottleNecksDepth = self.getDepthPos(gameState, self.offBottleNeckPos, self.offBottleNeckCount)
    # if FOOD_DEPTHS is None:
    #   FOOD_DEPTHS = self.offBottleNecksDepth
    # self.depth0FoodCounts = len([d for d in FOOD_DEPTHS.values() if d == 0])
    # self.offBottleNeckCount, self.offBottleNeckPos = self.getTopkBottleneck(gameState, 5, defMode=False)

    # for pos, count in self.bottleNeckCount:
    #   if count >= 0:
    #     if DEBUG:
    #       # cFood = len(self.masterFoods) - len(list(set(self.masterFoods) - set(self.bottleNeckPos[pos])))
    #       # cCapsule = len([cap for cap in self.masterCapsules if cap in self.bottleNeckPos[pos]])
    #       # print(f"Bottleneck {pos} contains {count} dots with {cFood} foods and {cCapsule} capsules")
    #       if DEBUG: self.debugDraw(pos, [1, 0, 0])
    #       # print(self.bottleNeckPos[pos])
    # for pos, count in self.offBottleNeckCount:
    #   if count >= 0:
    #     if DEBUG:
    #       if DEBUG: self.debugDraw(pos, [1, 0, 0])
    toc = time.perf_counter()
    if DEBUG: print(f"Ran in {toc - tic:0.4f} seconds")

    # diverging strategy
    self.initialDivergingStrategy = True if len(self.offensiveCapsules) == 0 else False
    self.crossedBoundaryOnce = False

  def getAgent(self, gameState):
    if self.index == 0:
      return self.getAttackModeAgent(gameState)
    elif self.index == 1:
      return self.getAttackModeAgent(gameState)
    elif self.index == 2:
      return self.getAttackModeAgent(gameState)
    else:
      return self.getAttackModeAgent(gameState)

  def getAttackModeAgent(self, gameState):
    global RED_ATTACKERS, BLUE_ATTACKERS, RED_DEFENDERS, BLUE_DEFENDERS
    global AGENT_1_AREA, AGENT_2_AREA
    # AGENT_1_AREA = None
    # AGENT_2_AREA = None
    if self.AGENT_MODE == "DEFEND":
      self.AGENT_MODE = "ATTACK"
      if gameState.isOnRedTeam(self.index):
        RED_ATTACKERS += 1
        RED_DEFENDERS -= 1
      else:
        BLUE_ATTACKERS += 1
        BLUE_DEFENDERS -= 1
      return self.agentsPool["ATTACKING_AGENT"]
    elif self.AGENT_MODE == "APPROXIMATE":
      self.AGENT_MODE = "ATTACK"
      return self.agentsPool["ATTACKING_AGENT"]
    else:
      self.AGENT_MODE = "ATTACK"
      if gameState.isOnRedTeam(self.index):
        RED_ATTACKERS += 1
      else:
        BLUE_ATTACKERS += 1
      # agent = offensivePDDLAgent(self.index)
      # agent.registerInitialState(gameState)
      # agent.observationHistory = self.observationHistory
      return self.agentsPool["ATTACKING_AGENT"]

  def getDefendModeAgent(self, gameState):
    global RED_ATTACKERS, BLUE_ATTACKERS, RED_DEFENDERS, BLUE_DEFENDERS
    # switching mode
    if self.AGENT_MODE == "ATTACK":
      self.AGENT_MODE = "DEFEND"
      if gameState.isOnRedTeam(self.index):
        RED_ATTACKERS -= 1
        RED_DEFENDERS += 1
      else:
        BLUE_ATTACKERS -= 1
        BLUE_DEFENDERS += 1
      # agent = defensivePDDLAgent(self.index)
      # agent.registerInitialState(gameState)
      # agent.observationHistory = self.observationHistory
      return self.agentsPool["DEFENDING_AGENT"]
    else:
      self.AGENT_MODE = "DEFEND"
      if gameState.isOnRedTeam(self.index):
        RED_DEFENDERS += 1
      else:
        BLUE_DEFENDERS += 1
      # agent = defensivePDDLAgent(self.index)
      # agent.registerInitialState(gameState)
      # agent.observationHistory = self.observationHistory
      return self.agentsPool["DEFENDING_AGENT"]

  def getApproximateModeAgent(self):
    self.AGENT_MODE = "APPROXIMATE"
    return self.agentsPool["APPROXIMATING_AGENT"]

  def chooseAction(self, gameState):
    global AGENT_1_POSITION, AGENT_2_POSITION
    global AGENT_1_MODE, AGENT_2_MODE, DEPTH_LIMIT, TOTAL_FOOD_COLLECTED
    if DEBUG: print(f"======MODE: #{self.index} - {self.AGENT_MODE}======")
    start = time.time()
    try:
      with time_limit(self.TIME_LIMIT):
        # if not util.flipCoin(0.7):
        #   time.sleep(2)
        myCurrPos = gameState.getAgentPosition(self.index)
        myState = gameState.getAgentState(self.index)
        scared = True if myState.scaredTimer > 5 else False
        foods = self.getFood(gameState).asList()
        numGhosts = self.numberOfGhosts(gameState)

        # sync agents mode
        self.syncAgentMode()

        # clean agents if dead
        if myCurrPos == self.start: self.cleanAgent(gameState)

        # init agents
        if self.index == 0 or self.index == 1: self.initAgent1(myCurrPos, foods)
        else: self.initAgent2(myCurrPos)

        # no ghosts - no depth limit - no capsules
        # if len(self.masterCapsules) == 0:
        #   if numGhosts == 0 and DEPTH_LIMIT:
        #     print("Turning off depth limit - no ghosts")
        #     DEPTH_LIMIT = False
        #   # if ghosts back - check if food is collected
        #   elif numGhosts > 0:
        #     if TOTAL_FOOD_COLLECTED >= self.depth0FoodCounts - 2:
        #       print("Turning off depth limit - depth 0 food collected")
        #       DEPTH_LIMIT = False
        #     else:
        #       print("Depth limit enabled - ghosts back")
        #       DEPTH_LIMIT = True

        # check if agent crossed boundary once
        # if not self.crossedBoundaryOnce and self.AGENT_MODE == "ATTACK":
        #   self.crossedBoundaryOnce = self.crossedBoundary(myCurrPos, gameState)

        # check if agent is stuck
        # Both Agent 1 and attacking Agent 2
        if (self.index == 0 or self.index == 1):
          self.checkIfStuck(myCurrPos)
        elif (self.index == 2 or self.index == 3) and self.AGENT_MODE == "ATTACK":
          self.checkIfStuck(myCurrPos)

        # For Agent 1 and Agent 2 in attack mode
        if self.approximationMode:
          self.makeApproximationAgent()

        if self.AGENT_MODE == "APPROXIMATE":
          self.handleApproximationFlow(myCurrPos, gameState)

        # diverge the attack - if both attacking
        # if (self.index == 0 or self.index == 1) and self.AGENT_MODE == "ATTACK":
        #   returnedAction = self.divergeAttackAgent1Flow(myCurrPos, gameState)
        #   if returnedAction is not None:
        #     return returnedAction
        # elif (self.index == 2 or self.index == 3) and self.AGENT_MODE == "ATTACK":
        #   returnedAction = self.divergeAttackAgent2Flow(myCurrPos, gameState)
        #   if returnedAction is not None:
        #     return returnedAction

        # if agent is at boundary
        if myCurrPos[0] == self.getBoundaryX(gameState):
          self.atBoundaryLogic(myCurrPos, gameState)

        if len(self.offensiveCapsules) > 0:
          # Agent 2 - Defensive block check
          # put all blocking decisions here
          if (self.index == 2 or self.index == 3) \
              and self.AGENT_MODE != "BLOCKING" and myState.scaredTimer <= 3:
            returnedAction = self.checkAndGetBlockingDefensiveAgent(gameState)
            AGENT_2_MODE = "BLOCKING"
            if returnedAction is not None:
              return returnedAction

          # Agent 2 - Defensive block check again
          if self.AGENT_MODE == "BLOCKING" and (self.index == 2 or self.index == 3):
            returnedAction = self.handleBlockingAgentFlow(myCurrPos, gameState)
            if returnedAction is not None:
              return returnedAction

        # Agent 1 not approximating
        if not self.approximationMode:
          # Agent 1
          if self.AGENT_MODE == "ATTACK" and (self.index == 0 or self.index == 1) and not scared:
            returnedAction = self.agent1AttackingFlow(myCurrPos, foods, gameState)
            if returnedAction is not None:
              return returnedAction

        # Agent 2
        if self.AGENT_MODE == "ATTACK" \
            and (self.index == 2 or self.index == 3):
          returnedAction = self.agent2AttackingFlow(myCurrPos, foods, gameState)
          if returnedAction is not None:
            return returnedAction

        nextAction = self.agent.chooseAction(gameState)

        # Agent 1
        if self.AGENT_MODE == "DEFEND" and (self.index == 0 or self.index == 1):
          self.agent1DefendingFlow(scared, gameState)

        # Agent 2
        if self.AGENT_MODE == "DEFEND" and (self.index == 2 or self.index == 3):
          self.agent2DefendingFlow(myCurrPos, gameState, nextAction)

        if DEBUG: print('Eval time for agent %d: %.4f seconds' % (self.index, time.time() - start))

        return nextAction

    except TimeoutException as e:
      if DEBUG: print('==== Time Limit Exceeded: #%d: %.4f seconds ====' % (self.index, time.time() - start))
      newAction = self.agentsTimeoutFlow(gameState)
      if DEBUG: print('Eval time for agent %d: %.4f seconds' % (self.index, time.time() - start))
      return newAction

  def checkAndGetBlockingDefensiveAgent(self, gameState):
    global DEFENSIVE_BLOCKING_MODE
    bestBottleNeckPos, bestBottleNeckCount = self.bottleNeckCount[0]
    # masterCapsules = self.getCapsulesYouAreDefending(gameState)
    cFood = len(self.masterFoods) - len(list(set(self.masterFoods) - set(self.bottleNeckPos[bestBottleNeckPos])))
    cCapsule = len([cap for cap in self.masterCapsules if cap in self.bottleNeckPos[bestBottleNeckPos]])
    bottleNeckFood = cFood / len(self.masterFoods)
    # it covers all capsules
    if cCapsule == len(self.masterCapsules):
      # 20% food blocked
      if bottleNeckFood > self.foodBottleNeckThreshold:
        # then tell defender to go there and stop
        if DEBUG: print("Turning defensive agent into blocking agent")
        self.agent = self.getDefendModeAgent(gameState)
        self.AGENT_MODE = "BLOCKING"
        DEFENSIVE_BLOCKING_MODE = True
        return self.agent.chooseAction(gameState, {
          "BlockingPos": bestBottleNeckPos
        })
    return None

  def handleBlockingAgentFlow(self, myCurrPos, gameState):
    global DEFENSIVE_BLOCKING_MODE
    if myCurrPos == self.start:
      # I died being a blocker
      if DEBUG: print("Died being blocker - switch back")
      self.agent = self.getDefendModeAgent(gameState)
      DEFENSIVE_BLOCKING_MODE = False
    else:
      bestBottleNeckPos, bestBottleNeckCount = self.bottleNeckCount[0]
      if myCurrPos != bestBottleNeckPos:
        return self.agent.chooseAction(gameState, {
          "BlockingPos": bestBottleNeckPos
        })
      else:
        if DEBUG: print("STOP HERE")
        return "Stop"

  def makeApproximationAgent(self):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN
    global AGENT_1_AREA, AGENT_2_AREA
    self.movesApproximating += 1
    if DEBUG: print(f"Approximate Move: {self.movesApproximating}")
    if self.AGENT_MODE != "APPROXIMATE" and self.AGENT_MODE == "ATTACK":
      # initilize ApproximateQLearning
      if DEBUG: print("Turning Attack PDDL into Approximating Agent")
      if self.index == 0 or self.index == 1:
        self.lastFoodEaten = AGENT_1_FOOD_EATEN
        AGENT_1_AREA = None
      else:
        self.lastFoodEaten = AGENT_2_FOOD_EATEN
        AGENT_2_AREA = None
      self.agent = self.getApproximateModeAgent()

  def handleApproximationFlow(self, myCurrPos, gameState):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN
    if ((self.index == 0 or self.index == 1)
        and self.lastFoodEaten != AGENT_1_FOOD_EATEN) \
        or ((self.index == 2 or self.index == 3)
            and self.lastFoodEaten != AGENT_2_FOOD_EATEN) \
        or ((self.index == 0 or self.index == 1)
            and self.movesApproximating > self.approximationThresholdAgent1) \
        or ((self.index == 2 or self.index == 3)
            and self.movesApproximating > self.approximationThresholdAgent2):
      if DEBUG: print("Food Eaten/Threshold Reached -> switching back to Attack PDDL")
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      self.agent = self.getAttackModeAgent(gameState)
    # elif myCurrPos == self.start:
    #   self.history = Queue()
    #   self.approximationMode = False
    #   self.movesApproximating = 0
    #   if self.index == 0 or self.index == 1:
    #     if DEBUG: print("I died approximating - turn into offensive agent")
    #     self.agent = self.getAttackModeAgent(gameState)
    #   else:
    #     if DEBUG: print("I died approximating - turn into defensive agent")
    #     self.agent = self.getDefendModeAgent(gameState)
    elif self.red and myCurrPos[0] <= self.getBoundaryX(gameState) - 3:
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      # if self.index == 0:
      if DEBUG: print("back home - attack PDDL ON - stop approximating")
      self.agent = self.getAttackModeAgent(gameState)
      # else:
      #   if DEBUG: print("back home - defend PDDL ON - stop approximating")
      #   self.agent = self.getDefendModeAgent(gameState)
    elif not self.red and myCurrPos[0] >= self.getBoundaryX(gameState) + 3:
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      # if self.index == 1:
      if DEBUG: print("back home - attack PDDL ON - stop approximating")
      self.agent = self.getAttackModeAgent(gameState)
      # else:
      #   if DEBUG: print("back home - defend PDDL ON - stop approximating")
      #   self.agent = self.getDefendModeAgent(gameState)

  def atBoundaryLogic(self, myCurrPos, gameState):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN, TOTAL_FOODS
    global TOTAL_FOOD_COLLECTED, DEPTH_LIMIT
    if DEBUG: print("At Boundary - Food Collected/Resetting")
    newScore = self.getScore(gameState)
    if self.index == 0 or self.index == 1:
      if newScore > self.currScore:
        TOTAL_FOODS -= AGENT_1_FOOD_EATEN
        self.currScore = newScore
        # TOTAL_FOOD_COLLECTED += AGENT_1_FOOD_EATEN
      AGENT_1_FOOD_EATEN = 0
    else:
      if newScore > self.currScore:
        TOTAL_FOODS -= AGENT_2_FOOD_EATEN
        self.currScore = newScore
        # TOTAL_FOOD_COLLECTED += AGENT_2_FOOD_EATEN
      AGENT_2_FOOD_EATEN = 0
    # if TOTAL_FOOD_COLLECTED >= self.depth0FoodCounts - 2:
    #   print("Turning off depth limit")
    #   DEPTH_LIMIT = False

  def setAgent1ClosestFood(self, myCurrPos, foods):
    global AGENT_1_CLOSEST_FOOD
    foodDists = [(food, self.getMazeDistance(myCurrPos, food)) for food in foods]
    if len(foodDists) > 0:
      minFoodPos = min(foodDists, key=lambda t: t[1])[0]
      AGENT_1_CLOSEST_FOOD = minFoodPos
    else:
      AGENT_1_CLOSEST_FOOD = None

  def agent1AttackingFlow(self, myCurrPos, foods, gameState):
    if self.numberOfInvaders(gameState) == 2:
      if self.red:
        if myCurrPos[0] <= self.agent.getBoundaryX(gameState):
          if DEBUG: print("back home - defensive mode ON - heavy invaders")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if DEBUG: print("Heavy Invaders - decrease threshold")
          return self.agent.chooseAction(gameState, {
            "threshold": 0.30
          })
      else:
        if myCurrPos[0] >= self.agent.getBoundaryX(gameState):
          if DEBUG: print("back home - defensive mode ON - heavy invaders")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if DEBUG: print("Heavy Invaders - decrease threshold")
          return self.agent.chooseAction(gameState, {
            "threshold": 0.30
          })

  def agent2AttackingFlow(self, myCurrPos, foods, gameState):
    # if myCurrPos == self.start:
    #   if DEBUG: print("I died - create new offensive agent")
    #   self.agent = self.getAttackModeAgent(gameState)
    if len(foods) <= 2:
      if DEBUG: print("len(foods) <= defensive mode ON")
      self.agent = self.getDefendModeAgent(gameState)
    # come back home
    elif self.isEnemyEnteredTerritory(gameState):
      if self.red:
        if myCurrPos[0] <= self.agent.getBoundaryX(gameState):
          if DEBUG: print("back home - defensive mode ON")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if DEBUG: print("stay offensive - go back home")
          return self.agent.chooseAction(gameState, {
            "problemObjective": "COME_BACK_HOME"
          })
      else:
        if myCurrPos[0] >= self.agent.getBoundaryX(gameState):
          if DEBUG: print("back home - defensive mode ON")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if DEBUG: print("stay offensive - go back home")
          return self.agent.chooseAction(gameState, {
            "problemObjective": "COME_BACK_HOME"
          })

  def agent1DefendingFlow(self, scared, gameState):
    if scared:
      if DEBUG: print("Turn into offensive")
      self.agent = self.getAttackModeAgent(gameState)
    elif self.numberOfInvaders(gameState) < 2:
      if DEBUG: print("Invaders reduced - switching back to attack mode")
      self.agent = self.getAttackModeAgent(gameState)

  def agent2DefendingFlow(self, myCurrPos, gameState, nextAction):
    if self.allInvadersKilled(myCurrPos, gameState, nextAction) \
        or not self.isEnemyEnteredTerritory(gameState):
      if DEBUG: print("EATEN ALL INVADERS | No enemy")
      # turn it into offensive
      self.agent = self.getAttackModeAgent(gameState)

  def agentsTimeoutFlow(self, gameState):
    newAction = None
    if ((self.index == 0 or self.index == 1) and self.AGENT_MODE == "ATTACK") \
        or ((self.index == 2 or self.index == 3) and self.AGENT_MODE == "ATTACK"):
      if DEBUG: print(f"=== Get Approximate Action: #{self.index} ====")
      agent = self.agentsPool["APPROXIMATING_AGENT"]
      newAction = agent.chooseAction(gameState)
    else:
      if DEBUG: print(f"==== Get Random Action: #{self.index} ====")
      newAction = random.choice(gameState.getLegalActions(self.index))
    return newAction

  def syncAgentMode(self):
    global AGENT_1_MODE, AGENT_2_MODE
    if self.index == 0 or self.index == 1:
      AGENT_1_MODE = self.AGENT_MODE
    else:
      AGENT_2_MODE = self.AGENT_MODE

  def cleanAgent(self, gameState):
    global AGENT_1_AREA, AGENT_2_AREA
    # if DEBUG: print("Died - Enabled Initial Diverging Strategy")
    self.initialDivergingStrategy = True if len(self.offensiveCapsules) == 0 else False
    AGENT_1_AREA = None
    AGENT_2_AREA = None
    # failsafe
    if self.AGENT_MODE == "APPROXIMATE":
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      if DEBUG: print("Turning Approximate agent into attacking agent")
      self.agent = self.getAttackModeAgent(gameState)

  def initAgent1(self, myCurrPos, foods):
    AGENT_1_POSITION = myCurrPos
    self.setAgent1ClosestFood(myCurrPos, foods)

  def initAgent2(self, myCurrPos):
    AGENT_2_POSITION = myCurrPos

  def divergeAttackAgent1Flow(self, myCurrPos, gameState):
    global AGENT_1_AREA, AGENT_2_AREA
    # no capsule situation
    if self.initialDivergingStrategy and AGENT_1_AREA is None:
      if DEBUG: print("Disable Initial Diverging Strategy")
      AGENT_1_AREA = "Up"
      self.initialDivergingStrategy = False
    # back home after running/eating
    # fired only when agent is back home after crossing boundary once
    elif ((self.red and myCurrPos[0] < self.agent.getBoundaryX(gameState)) \
          or (not self.red and myCurrPos[0] > self.agent.getBoundaryX(gameState))) \
        and AGENT_1_AREA is None and self.crossedBoundaryOnce:
      area = "Up" if myCurrPos[1] / (gameState.data.layout.height - 1) <= 0.5 else "Down"
      # 5% exploration
      if util.flipCoin(0.05):
        if area == "Up":
          area = "Down"
        elif area == "Down":
          area = "Up"
      if AGENT_2_AREA is not None and AGENT_2_AREA == area:
        if area == "Up":
          area = "Down"
        elif area == "Down":
          area = "Up"
      AGENT_1_AREA = area
      # print(AGENT_1_AREA)

    if AGENT_1_AREA is not None:
      boundaryPositions = self.getBoundaryPos(gameState, span=1)
      if AGENT_1_AREA == "Up":
        agent1BoundaryPos = boundaryPositions[-1]
      else:
        agent1BoundaryPos = boundaryPositions[0]
      if self.red:
        agent1BoundaryPos = (agent1BoundaryPos[0] + 1, agent1BoundaryPos[1])
        targetBoundaryPoint = self.agent.getBoundaryX(gameState) + 1
      else:
        agent1BoundaryPos = (agent1BoundaryPos[0] - 1, agent1BoundaryPos[1])
        targetBoundaryPoint = self.agent.getBoundaryX(gameState) - 1
      if myCurrPos[0] != targetBoundaryPoint:
        if DEBUG: print(f"Trying to reach the boundary: {AGENT_1_AREA}, {agent1BoundaryPos}")
        return self.agent.chooseAction(gameState, {
          "reachBoundaryPoint": agent1BoundaryPos
        })
      else:
        if DEBUG: print(f"Reached boundary point: {AGENT_1_AREA}, {agent1BoundaryPos}")
        AGENT_1_AREA = None

  def divergeAttackAgent2Flow(self, myCurrPos, gameState):
    global AGENT_1_AREA, AGENT_2_AREA
    if self.initialDivergingStrategy and AGENT_2_AREA is None:
      AGENT_2_AREA = "Down"
      if DEBUG: print("Disable Initial Diverging Strategy")
      self.initialDivergingStrategy = False
    # back home after running/eating
    elif ((self.red and myCurrPos[0] < self.agent.getBoundaryX(gameState)) \
          or (not self.red and myCurrPos[0] > self.agent.getBoundaryX(gameState))) \
        and AGENT_2_AREA is None and self.crossedBoundaryOnce:
      area = "Up" if myCurrPos[1] / (gameState.data.layout.height - 1) <= 0.5 else "Down"
      # 5% exploration
      if util.flipCoin(0.05):
        if area == "Up":
          area = "Down"
        elif area == "Down":
          area = "Up"
      if AGENT_1_AREA is not None and AGENT_1_AREA == area:
        if area == "Up":
          area = "Down"
        elif area == "Down":
          area = "Up"
      AGENT_2_AREA = area

    if AGENT_2_AREA is not None:
      boundaryPositions = self.getBoundaryPos(gameState, span=1)
      if AGENT_2_AREA == "Up":
        agent2BoundaryPos = boundaryPositions[-1]
      else:
        agent2BoundaryPos = boundaryPositions[0]
      if self.red:
        agent2BoundaryPos = (agent2BoundaryPos[0] + 1, agent2BoundaryPos[1])
        targetBoundaryPoint = self.agent.getBoundaryX(gameState) + 1
      else:
        agent2BoundaryPos = (agent2BoundaryPos[0] - 1, agent2BoundaryPos[1])
        targetBoundaryPoint = self.agent.getBoundaryX(gameState) - 1
      if myCurrPos[0] != targetBoundaryPoint:
        if DEBUG: print(f"Trying to reach the boundary: {AGENT_2_AREA}, {agent2BoundaryPos}")
        return self.agent.chooseAction(gameState, {
          "reachBoundaryPoint": agent2BoundaryPos
        })
      else:
        if DEBUG: print(f"Reached boundary point: {AGENT_2_AREA}, {agent2BoundaryPos}")
        AGENT_2_AREA = None

###################
# Offensive Agent #
###################

class offensivePDDLAgent(BaseAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''
    self.createPacmanDomain()
    self.start = gameState.getAgentPosition(self.index)
    self.masterFoods = self.getFood(gameState).asList()
    self.cornerFoods = self.isSurrounded(gameState)
    self.masterCapsules = self.getCapsules(gameState)
    self.homePos = self.getBoundaryPos(gameState, 1)
    self.pddlFluentGrid = self.generatePDDLFluentStatic(gameState)
    self.pddlObject = self.generatePddlObject(gameState)
    self.foodEaten = 0
    self.depthLimit = True
    self.currScore = self.getScore(gameState)
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

  def generatePddlObject(self, gameState):
    """
    Function for creating PDDL objects for the problem file.
    """

    # Get Cell Locations without walls and Food count for object setup.
    allPos = gameState.getWalls().asList(False)
    # print(gameState.getWalls().asList(True))
    food_len = len(self.masterFoods)

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
    global FOOD_DEPTHS, DEPTH_LIMIT
    # Set Pacman Position
    at_food = None
    pacmanPos = gameState.getAgentPosition(self.index)
    at_pacman = f'\t\t(at-pacman cell{pacmanPos[0]}_{pacmanPos[1]})\n'

    # Set Food Position
    foods = self.getFood(gameState).asList()
    # if len(self.masterCapsules) == 0:
    #   print(f"DEPTH_LIMIT: {DEPTH_LIMIT}")
    #   currFoods = self.getFood(gameState).asList()
    #   at_food = list()
    #   for i, foodPos in enumerate(FOOD_DEPTHS.keys()):
    #     foodDepth = FOOD_DEPTHS[foodPos]
    #     if foodPos in currFoods:
    #       if DEPTH_LIMIT and foodDepth == 0:
    #         at_food.append(f'\t\t(at-food food{i + 1} cell{foodPos[0]}_{foodPos[1]})\n')
    #       elif not DEPTH_LIMIT:
    #         at_food.append(f'\t\t(at-food food{i + 1} cell{foodPos[0]}_{foodPos[1]})\n')
    if len(foods) != 0:
      if AGENT_1_CLOSEST_FOOD and self.index == 2 or self.index == 3:
        if DEBUG: print(f"Avoid Food: {AGENT_1_CLOSEST_FOOD}")
        at_food = [f'\t\t(at-food food{i+1} cell{food[0]}_{food[1]})\n'
                   for i, food in enumerate(foods)
                   if food != AGENT_1_CLOSEST_FOOD]
      else:
        at_food = [f'\t\t(at-food food{i+1} cell{food[0]}_{food[1]})\n' for i, food in enumerate(foods)]

    # Set Ghost(s) positions
    has_ghost = list()
    # if len(ANTICIPATER) == 0:
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    for ghost in ghosts:
      ghostPos = ghost.getPosition()
      if ghost.scaredTimer <= 3:
        has_ghost.append(f'\t\t(has-ghost cell{int(ghostPos[0])}_{int(ghostPos[1])})\n')
    # else:
    #   for ghostState, ghostPos in ANTICIPATER:
    #     if not ghostState.isPacman and ghostState.scaredTimer <= 3:
    #       has_ghost.append(f'\t\t(has-ghost cell{int(ghostPos[0])}_{int(ghostPos[1])})\n')

    # add ghosts in blind spot
    if len(features["blindSpots"]) > 0:
      for blindSpot in features["blindSpots"]:
        has_ghost.append(f'\t\t(has-ghost cell{int(blindSpot[0])}_{int(blindSpot[1])})\n')

    # Set Capsule Position
    capsules = self.getCapsules(gameState)
    has_capsule = [f'\t\t(has-capsule cell{capsule[0]}_{capsule[1]})\n' for capsule in capsules]

    fluents = list()
    fluents.append("\t(:init \n")
    fluents.append(at_pacman)
    if at_food:
      fluents.append("".join(at_food))
    fluents.append("".join(has_ghost))
    fluents.append("".join(has_capsule))
    if features["problemObjective"] == "DIE":
      if DEBUG: print("WANT_TO_DIE")
      fluents.append(f"\t\t(want-to-die)\n")
    fluents.append(self.pddlFluentGrid)
    fluents.append("\t)\n")

    return "".join(fluents)

  def generatePddlGoal(self, gameState, features):
    """
    Function for creating PDDL goals for the problem file.
    """
    if DEBUG: print(f'======New Offensive Action: #{self.index}========')

    problemObjective = None
    gameTimeLeft = gameState.data.timeleft
    pacmanPos = gameState.getAgentPosition(self.index)
    foods = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)
    thres = features["threshold"]

    # Get Food Eaten Calculation based on current Game Score
    newScore = self.getScore(gameState)
    if newScore > self.currScore:
      self.foodEaten += newScore - self.currScore
      self.currScore = newScore
    else:
      self.currScore = newScore

    goal = list()
    goal.append('\t(:goal (and\n')

    # Find if a ghost is in the proximity of pacman
    # if len(ANTICIPATER) == 0:
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    # ghostState = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    # else:
    #   ghosts = [ghostPos for ghostState, ghostPos in ANTICIPATER if not ghostState.isPacman]
    #   ghostState = [ghostState for ghostState, ghostPos in ANTICIPATER if not ghostState.isPacman]

    ghostDistance = 999999
    scaredTimer = 99999
    if len(ghosts) > 0:
      ghostDistance, scaredTimer = self.getGhostDistanceAndTimers(pacmanPos, ghosts)
      thres = features["threshold"]
      # if DEBUG: print(f"Ghosts: {thres}")
    # else:
    #   if DEBUG: print("Ghosts: 1.0")
    #   thres = 1

    if DEBUG: print(f'Pacman at {pacmanPos}')

    if features["problemObjective"] is None:
      closestHome, closestCap = self.compareCapsuleAndHomeDist(gameState, pacmanPos)
      # gameTimeLeft decrease by 4 for every 1 move - anticipate + come back
      if ((closestHome * 4) + 100) >= gameTimeLeft:
        if DEBUG: print(f"Timer Objective #1: {closestHome}")
        problemObjective = self.goBackHomeHardObjective(gameState, goal, pacmanPos)
      # if ghost is really close RUN to capsule if any or RUN BACK!
      elif self.stuck:
        problemObjective = self.goBackStartObjective(goal)
      elif ghostDistance <= 3 and scaredTimer <= 3:
        flag = self.getFlag(gameState, thres, foods, pacmanPos)
        if not flag and len(capsules) > 0:
          problemObjective = self.addEatCapsuleObjective(goal)
        else:
          problemObjective = self.goBackHomeHardObjective(gameState, goal, pacmanPos)
      else:
        # not being chased by ghost
        # or ghost is scared
        flag = self.getFlag(gameState, thres, foods, pacmanPos)
        if len(foods) > 2 and not flag:
          problemObjective = self.eatFoodObjective(goal)
        else:
          problemObjective = self.goBackHomeHardObjective(gameState, goal, pacmanPos)
    elif features["problemObjective"] == "GO_TO_BOUNDARY_POINT":
      problemObjective = self.goToBoundaryPoint(goal, features, gameState, pacmanPos)
    else:
      # fallback goals
      problemObjective = self.tryFallBackGoals(goal, features, gameState, pacmanPos)

    goal.append('\t))\n')
    return ("".join(goal), problemObjective)

  def getGhostDistanceAndTimers(self, pacmanPos, ghosts):
    dists = [self.getMazeDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    timers = [ghost.scaredTimer for ghost in ghosts]
    ghostDistance = min(dists)
    scaredTimer = min(timers)
    if DEBUG: print(f'Ghost Alert with Dist: {ghostDistance} | scaredTimer: {scaredTimer}')
    return (ghostDistance, scaredTimer)

  def compareCapsuleAndHomeDist(self, gameState, pacmanPos):
    x = self.getBoundaryX(gameState)

    if len(self.getCapsules(gameState)) > 0:
      closestCap = min([self.getMazeDistance(pacmanPos, cap) for cap in self.getCapsules(gameState)])
      closestHome = min([self.getMazeDistance(pacmanPos, pos) for pos in self.homePos if pos[0] == x])
    else:
      closestHome = 1
      closestCap = 10

    return (closestHome, closestCap)

  def getFlag(self, gameState, threshold, foods, pacmanPos):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN, TOTAL_FOODS
    global DEFENSIVE_BLOCKING_MODE
    totalFoodEaten = AGENT_1_FOOD_EATEN + AGENT_2_FOOD_EATEN
    if self.currScore < 0 and totalFoodEaten + self.currScore > 0 \
        and DEFENSIVE_BLOCKING_MODE:
      if DEBUG: print("Defensive Blocking: BACK HOME")
      return True
    try:
      foodEatenPer = totalFoodEaten / TOTAL_FOODS
    except:
      if DEBUG: print("Re-syncing total foods")
      TOTAL_FOODS = len(self.getFood(gameState).asList())
      foodEatenPer = 0
    if DEBUG: print(f"Relative Food Eaten: {round(foodEatenPer,2) * 100}%")
    # foodLeft = len(self.masterFoods) - self.foodEaten
    # foodCaryingPer = (foodLeft - len(foods)) / foodLeft
    foodDists = [self.getMazeDistance(pacmanPos, food) for food in foods]
    if len(foodDists) > 0:
      minDistance = min([self.getMazeDistance(pacmanPos, food) for food in foods])
    else:
      minDistance = 99999
    # so close to food - eat and then run back
    flag = True if foodEatenPer > threshold and minDistance > 1 else False
    return flag

  def addEatCapsuleObjective(self, goal):
    if DEBUG: print('Objective #2')
    goal.append(f'\t\t(capsule-eaten)\n')
    return "EAT_CAPSULE"

  def goBackStartObjective(self, goal):
    if DEBUG: print('Objective #3')
    goal.append(f'\t\t(at-pacman cell{self.start[0]}_{self.start[1]})\n')
    return "GO_START"

  def goBackHomeHardObjective(self, gameState, goal, pacmanPos):
    if DEBUG: print('Objective #4')
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
    if DEBUG: print('Objective #5')
    goal.append(f'\t\t(carrying-food)\n')
    return "EAT_FOOD"

  def goToBoundaryPoint(self, goal, features, gameState, pacmanPos):
    if DEBUG: print('Objective goToBoundaryPoint')
    boundaryPoint = features["reachBoundaryPoint"]
    goal.append(f'\t\t(at-pacman cell{boundaryPoint[0]}_{boundaryPoint[1]})\n')
    return "GO_TO_BOUNDARY_POINT"

  def tryFallBackGoals(self, goal, features, gameState, pacmanPos):
    if features["problemObjective"] == "COME_BACK_HOME":
      if DEBUG: print('Objective #6 [FALLBACK]')
      return self.goBackHomeHardObjective(gameState, goal, pacmanPos)
    elif features["problemObjective"] == "DIE":
      if DEBUG: print('Objective #7 [FALLBACK]')
      goal.append(f'\t\t(die)\n')
      return "DIE"

  def goBackHomeWithFoodObjective(self, gameState, goal, pacmanPos):
    if DEBUG: print('Objective #7')
    x = self.getBoundaryX(gameState)
    goal.append(f'\t\t(carrying-food)\n')
    if pacmanPos in self.homePos:
      goal.append(f'\t\t(at-pacman cell{self.start[0]}_{self.start[1]})\n')
    else:
      goal.append('\t\t(or\n')
      for pos in self.homePos:
        if pos[0] == x:
          goal.append(f'\t\t\t(at-pacman cell{pos[0]}_{pos[1]})\n')
      goal.append('\t\t)\n')
    return "COME_BACK_HOME_WITH_FOOD"

  def generatePddlProblem(self, gameState, features):
    """
    Generates a file for Creating PDDL problem file for current state.
    """
    problem = list()
    problem.append(f'(define (problem p{self.index}-pacman)\n')
    problem.append('\t(:domain pacman)\n')
    # problem.append(self.pddlObject)
    problem.append(self.generatePddlObject(gameState))
    problem.append(self.generatePddlFluent(gameState, features))
    goalStatement, goalObjective = self.generatePddlGoal(gameState, features)
    problem.append(goalStatement)
    problem.append(')')

    problem_file = open(f"{CD}/pacman-problem-{self.index}.pddl", "w")
    problem_statement = "".join(problem)
    problem_file.write(problem_statement)
    problem_file.close()
    return (f"pacman-problem-{self.index}.pddl", goalObjective)

  def chooseAction(self, gameState, overridefeatures=None):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN, TOTAL_FOODS
    global DEFENSIVE_BLOCKING_MODE
    # print(f"TOTAL_FOODS -> {TOTAL_FOODS}")
    # print(f"AGENT_1_FOOD_EATEN: {AGENT_1_FOOD_EATEN}")
    # print(f"AGENT_2_FOOD_EATEN: {AGENT_2_FOOD_EATEN}")
    # global ANTICIPATER
    features = {"problemObjective": None,
                "threshold": 0.65,
                "generateGrid": False,
                "blindSpots": [],
                "reachBoundaryPoint": None}

    if overridefeatures:
      if "problemObjective" in overridefeatures:
        if DEBUG: print("Overriding problemObjective")
        features["problemObjective"] = overridefeatures["problemObjective"]
      if "threshold" in overridefeatures:
        if DEBUG: print("Overriding threshold")
        features["threshold"] = overridefeatures["threshold"]
      if "reachBoundaryPoint" in overridefeatures:
        if DEBUG: print("Overriding reachBoundaryPoint")
        features["reachBoundaryPoint"] = overridefeatures["reachBoundaryPoint"]
        features["problemObjective"] = "GO_TO_BOUNDARY_POINT"

    if DEFENSIVE_BLOCKING_MODE:
      if DEBUG: print("Defensive blocking - Threshold down to 50%")
      features["threshold"] = 0.50

    agentPosition = gameState.getAgentPosition(self.index)
    if agentPosition == self.start:
      if self.index == 0 or self.index == 1:
        # print(f"DIE: AGENT_1_FOOD_EATEN = 0")
        AGENT_1_FOOD_EATEN = 0
      else:
        # print(f"DIE: AGENT_2_FOOD_EATEN = 0")
        AGENT_2_FOOD_EATEN = 0

    self.checkBlindSpot(agentPosition, gameState, features)

    plannerPosition, plan, \
    problemObjective, planner = self.getPlan(gameState, features)

    # fallback logic
    if plan is None:
      plannerPosition, plan, \
      problemObjective, planner = self.tryFallbackPlans(gameState, features,
                                                        problemObjective)
      if plan is None:
        plannerPosition, plan, \
        problemObjective, planner = self.tryFallbackPlans(gameState, features,
                                                          problemObjective)
        # fallback failed -> die
        if plan is None:
          problemObjective = "DIE"
          plannerPosition, plan, \
          problemObjective, planner = self.tryFallbackPlans(gameState, features,
                                                            problemObjective)

    action = planner.get_legal_action(agentPosition, plannerPosition)
    if DEBUG: print(f'Action Planner: {action}')

    # anticipate what will happen next
    myFoods = self.getFood(gameState).asList()
    if len(myFoods) > 0:
      distToFood = min([self.getMazeDistance(agentPosition, food) for food in myFoods])
    else:
      distToFood = 99999
    # I am 1 step away, will I be able to eat it?
    if distToFood == 1:
      nextGameState = gameState.generateSuccessor(self.index, action)
      nextFoods = self.getFood(nextGameState).asList()
      if len(myFoods) - len(nextFoods) == 1:
        # I will eat food
        if self.index == 0 or self.index == 1:
          AGENT_1_FOOD_EATEN += 1
        else:
          AGENT_2_FOOD_EATEN += 1
    return action

  def tryFallbackPlans(self, gameState, features, problemObjective):
    # no plan found for eating capsule
    if problemObjective == "EAT_CAPSULE":
      if DEBUG: print("No plan found for Objective #1")
      # try coming back home
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "EAT_FOOD":
      if DEBUG: print("No plan found for Objective #2")
      # try coming back home
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "GO_START":
      if DEBUG: print("No plan found for Objective #3")
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "GO_TO_BOUNDARY_POINT":
      if DEBUG: print("No plan found for GO_TO_BOUNDARY_POINT - Normal Flow")
      features["problemObjective"] = None
      return self.getPlan(gameState, features)
    elif problemObjective == "DIE" or problemObjective == "COME_BACK_HOME":
      if DEBUG: print("No plan found for Objective #4")
      features["problemObjective"] = "DIE"
      return self.getPlan(gameState, features)

  def getPlan(self, gameState, features):
    problem_file, problemObjective = self.generatePddlProblem(gameState, features)
    planner = PlannerFF(PACMAN_DOMAIN_FILE, problem_file)
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    return (plannerPosition, plan, problemObjective, planner)

  def euclideanDistance(self, xy1, xy2):
    return round(((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5, 5)

  def checkBlindSpot(self, agentPosition, gameState, features):
    # get cell location with walls
    allPos = gameState.getWalls().asList(False)
    capsules = self.getCapsules(gameState)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghostsPos = [a.getPosition() for a in enemies if not a.isPacman
                 and a.getPosition() != None and a.scaredTimer <= 3]
    if len(ghostsPos) > 0:
      ghostsDist = [(ghost, self.getMazeDistance(agentPosition, ghost)) for ghost in ghostsPos]
      minGhostPos, minGhostDistance = min(ghostsDist, key=lambda t: t[1])
      minGhostEucDistance = self.euclideanDistance(agentPosition, minGhostPos)
      # if DEBUG: print(minGhostDistance, minGhostEucDistance)
      if minGhostDistance == 2:
        # if minGhostEucDistance == round(math.sqrt(2), 5):
        if DEBUG: print("!! Blind Spot - anticipate ghosts positions !!")
        ghostX, ghostY = minGhostPos
        if (ghostX + 1, ghostY) in allPos and (ghostX + 1, ghostY) not in capsules\
            and (ghostX + 1, ghostY) not in ghostsPos:
          features["blindSpots"].append((ghostX + 1, ghostY))
        if (ghostX - 1, ghostY) in allPos and (ghostX - 1, ghostY) not in capsules \
            and (ghostX - 1, ghostY) not in ghostsPos:
          features["blindSpots"].append((ghostX - 1, ghostY))
        if (ghostX, ghostY - 1) in allPos and (ghostX, ghostY - 1) not in capsules \
            and (ghostX, ghostY - 1) not in ghostsPos:
          features["blindSpots"].append((ghostX, ghostY - 1))
        if (ghostX, ghostY + 1) in allPos and (ghostX, ghostY + 1) not in capsules \
            and (ghostX, ghostY + 1) not in ghostsPos:
          features["blindSpots"].append((ghostX, ghostY + 1))

##########################################
# Approximate Q learning Offensive Agent #
##########################################

class approxQLearningOffense(BaseAgent):

  def registerInitialState(self, gameState):
    self.epsilon = 0.05
    self.alpha = 0.2
    self.discount = 0.9
    # self.numTraining = NUM_TRAINING
    self.episodesSoFar = 0

    # Weights v1
    # self.weights = {'closest-food': -3.099192562140742,
    #                 'bias': -9.280875042529367,
    #                 '#-of-ghosts-1-step-away': -16.6612110039328,
    #                 'eats-food': 11.127808437648863}

    # Weights after training with anticipated ghost simulation
    self.weights = {'closest-food': -3.3579045624987347,
                    'bias': -45.43391457666167,
                    '#-of-ghosts-1-step-away': -77.03143989656415,
                    'eats-food': 21.07088642634957}

    self.start = gameState.getAgentPosition(self.index)
    self.featuresExtractor = FeaturesExtractor(self)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
        Picks among the actions with the highest Q(s,a).
    """
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN
    if DEBUG: print(f'======New Offensive Approx Action: #{self.index}========')
    legalActions = gameState.getLegalActions(self.index)
    agentPosition = gameState.getAgentPosition(self.index)

    if DEBUG: print(f'Pacman at {agentPosition}')

    if len(legalActions) == 0:
      return None

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in legalActions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    action = None
    # if TRAINING:
    #   for action in legalActions:
    #     self.updateWeights(gameState, action)
    if not util.flipCoin(self.epsilon):
      if DEBUG: print("Try Exploiting")
      action = self.getPolicy(gameState)
    else:
      if DEBUG: print("Try Exploring")
      action = random.choice(legalActions)
    if DEBUG: print(f"Approx Action: {action}")

    # anticipate what will happen next
    myFoods = self.getFood(gameState).asList()
    distToFood = min([self.getMazeDistance(agentPosition, food) for food in myFoods])
    # I am 1 step away, will I be able to eat it?
    if distToFood == 1:
      nextGameState = gameState.generateSuccessor(self.index, action)
      nextFoods = self.getFood(nextGameState).asList()
      if len(myFoods) - len(nextFoods) == 1:
        # I will eat food
        if self.index == 0 or self.index == 1:
          AGENT_1_FOOD_EATEN += 1
        else:
          AGENT_2_FOOD_EATEN += 1

    return action

  def getWeights(self):
    return self.weights

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    # features vector
    features = self.featuresExtractor.getFeatures(gameState, action)
    return features * self.weights

  def update(self, gameState, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.featuresExtractor.getFeatures(gameState, action)
    oldValue = self.getQValue(gameState, action)
    futureQValue = self.getValue(nextState)
    difference = (reward + self.discount * futureQValue) - oldValue
    # for each feature i
    for feature in features:
      newWeight = self.alpha * difference * features[feature]
      self.weights[feature] += newWeight
    # print(self.weights)

  def updateWeights(self, gameState, action):
    nextState = self.getSuccessor(gameState, action)
    reward = self.getReward(gameState, nextState)
    self.update(gameState, action, nextState, reward)

  def getReward(self, gameState, nextState):
    reward = 0
    agentPosition = gameState.getAgentPosition(self.index)

    # check if I have updated the score
    if self.getScore(nextState) > self.getScore(gameState):
      diff = self.getScore(nextState) - self.getScore(gameState)
      reward = diff * 10

    # check if food eaten in nextState
    myFoods = self.getFood(gameState).asList()
    distToFood = min([self.getMazeDistance(agentPosition, food) for food in myFoods])
    # I am 1 step away, will I be able to eat it?
    if distToFood == 1:
      nextFoods = self.getFood(nextState).asList()
      if len(myFoods) - len(nextFoods) == 1:
        reward = 10

    # check if I am eaten
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward = -100

    return reward

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    CaptureAgent.final(self, state)
    if DEBUG: print(self.weights)
    # did we finish training?

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return 0.0
    bestAction = self.getPolicy(gameState)
    return self.getQValue(gameState, bestAction)

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
    actionVals = {}
    bestQValue = float('-inf')
    for action in legalActions:
      targetQValue = self.getQValue(gameState, action)
      actionVals[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
    bestActions = [k for k, v in actionVals.items() if v == bestQValue]
    # random tie-breaking
    return random.choice(bestActions)

  def getPolicy(self, gameState):
    return self.computeActionFromQValues(gameState)

  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)

class FeaturesExtractor:

  def __init__(self, agentInstance):
    self.agentInstance = agentInstance

  def getFeatures(self, gameState, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = self.agentInstance.getFood(gameState)
    walls = gameState.getWalls()
    enemies = [gameState.getAgentState(i) for i in self.agentInstance.getOpponents(gameState)]
    ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
    # ghosts = state.getGhostPositions()

    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    agentPosition = gameState.getAgentPosition(self.agentInstance.index)
    x, y = agentPosition
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if len(ghosts) > 0:
    #   minGhostDistance = min([self.agentInstance.getMazeDistance(agentPosition, g) for g in ghosts])
    #   if minGhostDistance < 3:
    #     features["minGhostDistance"] = minGhostDistance

    # successor = self.agentInstance.getSuccessor(gameState, action)
    # features['successorScore'] = self.agentInstance.getScore(successor)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0

    # capsules = self.agentInstance.getCapsules(gameState)
    # if len(capsules) > 0:
    #   closestCap = min([self.agentInstance.getMazeDistance(agentPosition, cap) for cap in self.agentInstance.getCapsules(gameState)])
    #   features["closestCapsule"] = closestCap

    dist = self.closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height)
    features.divideAll(10.0)
    # print(features)
    return features

  def closestFood(self, pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        return dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None

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
            if DEBUG: print('Already in goal')
        else:
          if DEBUG: print('No plan!')
    except:
      if DEBUG: print('Something wrong happened with PDDL parsing')

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
      if DEBUG: print('should never have ocurred!')
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
      if DEBUG: print('Planner Returned Nothing.....')
      return "Stop"

  def get_coor_from_loc(self, loc):
    return loc.split("cell")[1].split("_")

###################
# Defensive Agent #
###################

class defensivePDDLAgent(BaseAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''
    self.createGhostDomain()
    self.start = gameState.getAgentPosition(self.index)
    self.pddlFluentGrid = self.generatePDDLFluentStatic(gameState)
    # self.pddlObject = self.generatePddlObject(gameState, features)
    self.boundaryPos = self.getBoundaryPos(gameState, 1)
    self.masterCapsules = self.getCapsulesYouAreDefending(gameState)
    self.masterFoods = self.getFoodYouAreDefending(gameState).asList()
    self.currScore = self.getScore(gameState)
    self.numFoodDef = len(self.masterFoods)
    self.target = list()

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

  def generatePddlObject(self, gameState, features = None):
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
    if features is None:
      invaders = [f'invader{i+1}' for i in range(invader_len)]
    else:
      invaders = [f'invader0']
    invaders.append("- invaders\n")

    objects.append("\t(:objects \n")
    objects.append(f'\t\t{" ".join(cells)}')
    objects.append(f'\t\t{" ".join(invaders)}')
    objects.append("\t)\n")

    return "".join(objects)

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

  def generatePddlFluent(self, gameState, features = None):
    """
    Function for creating PDDL fluents for the problem file.
    """

    # Set Self Position
    pacmanPos = gameState.getAgentPosition(self.index)
    at_ghost = f'\t\t(at-ghost cell{pacmanPos[0]}_{pacmanPos[1]})\n'

    # Set Invader(s) positions
    has_invaders = list()

    if features is None:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      for i, invader in enumerate(invaders):
        invaderPos = invader.getPosition()
        has_invaders.append(f'\t\t(at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')
    else:
      blockingPos = features["BlockingPos"]
      has_invaders.append(f'\t\t(at-invader invader0 cell{int(blockingPos[0])}_{int(blockingPos[1])})\n')

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

  def generatePddlGoal(self, gameState, features = None):
    """
    Function for creating PDDL goals for the problem file.
    """
    if DEBUG: print(f'======New Defensive Action: #{self.index}========')
    goal = list()
    goal.append('\t(:goal (and\n')

    myPos = gameState.getAgentPosition(self.index)
    if DEBUG: print(f'Ghost at {myPos}')
    foods = self.getFoodYouAreDefending(gameState).asList()
    prevFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList() \
      if self.getPreviousObservation() is not None else list()
    targetFood = list()
    invaders = list()
    Eaten = False

    # Get Food Defending Calculation based on current Game Score
    newScore = self.getScore(gameState)
    if newScore < self.currScore:
      self.numFoodDef -= self.currScore - newScore
      self.currScore = newScore
    else:
      self.currScore = newScore

    # myState = gameState.getAgentState(self.index)
    # scared = True if myState.scaredTimer > 2 else False
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    if features is None:
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      for i, invader in enumerate(invaders):
        invaderPos = invader.getPosition()
        goal.append(f'\t\t(not (at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')
    else:
      blockingPos = features["BlockingPos"]
      goal.append(f'\t\t(not (at-invader invader0 cell{int(blockingPos[0])}_{int(blockingPos[1])}))\n')
      invaders = []

    if len(foods) < self.numFoodDef:
      Eaten = True
      targetFood = list(set(prevFoods) - set(foods))
      if targetFood:
        self.target = targetFood
    elif self.numFoodDef == len(foods):
      Eaten = False
      self.target = list()
      if DEBUG: print(f'Handling #1')

    # If No Invaders are detected (Seen 5 steps)
    if not invaders:
      # If Food has not been eaten, Guard the Capsules or Foods
      if not Eaten:
        if myPos not in self.boundaryPos and len(enemyHere) == 0:
          if DEBUG: print(f'Going to #1')
          goal.extend(self.generateRedundantGoal(self.boundaryPos, myPos))
        elif myPos not in self.masterCapsules and len(self.getCapsulesYouAreDefending(gameState)) > 0:
          if DEBUG: print(f'Going to #2')
          capsules = self.getCapsulesYouAreDefending(gameState)
          goal.extend(self.shufflePddlGoal(capsules, myPos))
        else:
          if DEBUG: print(f'Going to #3')
          goal.extend(self.generateRedundantGoal(foods, myPos))
      # If Food have been eaten Rush to the food location.
      else:
        if DEBUG: print(f'Going to #4')
        if myPos in self.target:
          self.target.remove(myPos)
        goal.extend(self.shufflePddlGoal(self.target, myPos))

    goal.append('\t))\n')
    return "".join(goal)

  def generateRedundantGoal(self, compare, myPos):
    goal = list()
    goal.append('\t\t(or\n')
    for pos in compare:
      if myPos != pos:
        goal.append(f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n')
    goal.append('\t\t)\n')
    return goal

  def shufflePddlGoal(self, target, myPos):
    goal = list()
    if len(target) > 1:
      goal.append('\t\t(or\n')
      goal.extend([f'\t\t\t(at-ghost cell{pos[0]}_{pos[1]})\n' for pos in target])
      goal.append('\t\t)\n')
    elif len(target) == 1:
      goal.append(f'\t\t(at-ghost cell{target[0][0]}_{target[0][1]})\n')
    else:
      goal.extend(self.generateRedundantGoal(self.boundaryPos, myPos))
    return goal

  def generatePddlProblem(self, gameState, features = None):
    """
    Generates a file for Creating PDDL problem file for current state.
    """
    problem = list()
    problem.append(f'(define (problem p{self.index}-ghost)\n')
    problem.append('\t(:domain ghost)\n')
    # problem.append(self.pddlObject)
    problem.append(self.generatePddlObject(gameState, features))
    problem.append(self.generatePddlFluent(gameState, features))
    problem.append(self.generatePddlGoal(gameState, features))
    problem.append(')')

    problem_file = open(f"{CD}/ghost-problem-{self.index}.pddl", "w")
    problem_statement = "".join(problem)
    problem_file.write(problem_statement)
    problem_file.close()
    return f"ghost-problem-{self.index}.pddl"

  def chooseAction(self, gameState, features = None):
    # global ANTICIPATER
    agentPosition = gameState.getAgentPosition(self.index)
    problem_file = self.generatePddlProblem(gameState, features)
    planner = PlannerFF(GHOST_DOMAIN_FILE, problem_file)
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    action = planner.get_legal_action(agentPosition, plannerPosition)
    if DEBUG: print(f'Action Planner: {action}')
    # actions = gameState.getLegalActions(self.index)
    return action