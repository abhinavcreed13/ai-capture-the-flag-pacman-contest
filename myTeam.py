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

# Generic modules
import typing as t
import collections
import math
from datetime import timedelta, datetime
import random, time
import sys, os, platform
import re
import subprocess
import signal
from contextlib import contextmanager

# Pacman modules
from captureAgents import CaptureAgent
import util
from game import Directions, Actions, Configuration
import game
from util import nearestPoint, Queue
from collections import Counter
from capture import GameState
from distanceCalculator import Distancer

# FF real path
CD = os.path.dirname(os.path.abspath(__file__))

FF_EXECUTABLE_PATH = "ff"

PACMAN_DOMAIN_FILE = f"{CD}/pacman-domain.pddl"
GHOST_DOMAIN_FILE = f"{CD}/ghost-domain.pddl"

# global variables
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
AGENT_1_STUCK = False
AGENT_2_STUCK = False

FOOD_DEPTHS = None
ANTICIPATER = []
DEPTH_LIMIT = 2
DEPTH_LIMIT_ON = True
POWERED = False
POWERED_TIMER = 0

MCTS_DEBUG = False
PDDL_DEBUG = False
DEFENSIVE_BLOCKING_MODE = False


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MasterAgent', second='MasterAgent'):
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
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try:
    yield
  finally:
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

    invaders = self.numberOfInvaders(gameState)
    enemyState = gameState.getAgentState(enemy)
    if enemyState.isPacman:
      eatenFood = self.getFoodDiff(gameState)
      if eatenFood:
        for food in eatenFood:
          allObs[food] = 1.0 / invaders
        allObs.normalize()

    self.obs[enemy] = allObs

  def getFoodDiff(self, gameState):
    foods = self.getFoodYouAreDefending(gameState).asList()
    prevFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList() \
      if self.getPreviousObservation() is not None else list()
    eatenFood = []
    if len(foods) < len(prevFoods):
      eatenFood = list(set(prevFoods) - set(foods))
    return eatenFood

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
    values = list(self.obs.items())
    if values.count(max(values)) < 5:
      return self.obs[enemy].argMax()
    else:
      return None

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

    for enemy in self.getOpponents(gameState):
      anticipatedPos = self.approxPos(enemy)
      enemyGameState = gameState.getAgentState(enemy) if anticipatedPos else None
      anticipatedGhosts.append((enemyGameState, anticipatedPos))

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
      inCapsule = [True for capsule in capsules if capsule in coveredPos]
      for food in coveredPos:
        if food not in capsules:
          if not inCapsule:
            foodDepth[food] = self.getMazeDistance(pos, food) * 2
          else:
            foodDepth[food] = 0
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

  def getGhostPositions(self, gameState, pacmanPos):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
    dists = [self.getMazeDistance(pacmanPos, ghost) for ghost in ghosts]
    minDis = min(dists) if dists else 999999
    return ghosts, minDis

  def getMinGhostAnticipatedDistance(self, pacmanPos):
    global ANTICIPATER
    dists = [(ghostState, self.getMazeDistance(pacmanPos, ghostPos)) for ghostState, ghostPos in ANTICIPATER]
    minGhostState, minDis = min(dists, key=lambda t: t[1]) if dists else (None, 999999)
    return minGhostState, minDis

  def isGhostSpawnedBack(self):
    global ANTICIPATER
    for enemyStartPos in self.enemyStartPositions:
      for ghostState, ghostPos in ANTICIPATER:
        if enemyStartPos == ghostPos:
          return ghostPos
    return None

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
          if PDDL_DEBUG:
            print('I am Stuck! Needs Approximation')
          self.approximationMode = True
        self.history.pop()
        self.history.push(myCurrPos)
      except:
        if PDDL_DEBUG: print("!! STUCK PROBLEM - Reinit History !!")
        if PDDL_DEBUG: print(self.history.list)
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

  def updatePoweredStatus(self, agentPos, gameState, prevgameState):
    global POWERED, POWERED_TIMER
    global DEPTH_LIMIT, DEPTH_LIMIT_ON
    capsules = self.getCapsules(prevgameState) if prevgameState else self.getCapsules(gameState)
    if agentPos in capsules:
      if PDDL_DEBUG: print("PacMan Powered - No depth limit now")
      POWERED_TIMER = 40 * 2
      POWERED = True
      DEPTH_LIMIT = -1
      DEPTH_LIMIT_ON = False
    else:
      if POWERED_TIMER > 0:
        ghostState, currMinDis = self.getMinGhostAnticipatedDistance(agentPos)
        spawnedGhostPos = self.isGhostSpawnedBack()
        if spawnedGhostPos:
          # POWERED = True
          distance = self.getMazeDistance(agentPos, spawnedGhostPos)
          if PDDL_DEBUG: print(f'Ghost Spawned: {spawnedGhostPos}, {distance} - Powered Down')
          POWERED = False
          POWERED_TIMER = 0
        else:
          POWERED = True
          POWERED_TIMER -= 1
      else:
        POWERED = False
        POWERED_TIMER = 0
    if POWERED:
      if PDDL_DEBUG: print(f'POWERED ============== {POWERED_TIMER}')

  def areWeWinning(self, gameState):
    newScore = self.getScore(gameState)
    if newScore > 0:
      return True
    else:
      return False

  def isGameTied(self):
    newScore = self.getScore(gameState)
    if newScore == 0:
      return True
    else:
      return False


##################
## Master Agent ##
##################

class MasterAgent(BaseAgent):

  def registerInitialState(self, gameState):
    global TOTAL_FOODS, FOOD_DEPTHS
    CaptureAgent.registerInitialState(self, gameState)
    self.TIME_LIMIT = 0.9
    self.AGENT_MODE = None
    self.start = gameState.getAgentPosition(self.index)
    TOTAL_FOODS = len(self.getFood(gameState).asList())
    self.currScore = self.getScore(gameState)
    self.enemyStartPositions = []
    self.prevGhostDist = None

    # Bayesian Inference
    self.legalPositions = gameState.getWalls().asList(False)
    self.obs = {}
    for enemy in self.getOpponents(gameState):
      self.enemyStartPositions.append(gameState.getInitialAgentPosition(enemy))
      self.initalize(enemy, gameState.getInitialAgentPosition(enemy))

    # identify stuck & switch
    self.history = Queue()
    self.stuck = False
    self.approximationMode = False
    self.lastFoodEaten = 0
    self.movesApproximating = 0
    self.approximationThresholdAgent1 = 5
    self.approximationThresholdAgent2 = 5

    # agents pool
    # create attacking agent
    attackingAgent = OffensivePDDLAgent(self.index)
    attackingAgent.registerInitialState(gameState)
    attackingAgent.observationHistory = self.observationHistory

    # create defending agent
    defendingAgent = DefensivePDDLAgent(self.index)
    defendingAgent.registerInitialState(gameState)
    defendingAgent.observationHistory = self.observationHistory

    # create approximating attack agent
    approxAttackAgent = ApproxQLearningOffense(self.index)
    approxAttackAgent.registerInitialState(gameState)
    approxAttackAgent.observationHistory = self.observationHistory

    # create mcts agent
    mctsAgentInstance = MctsAgent(self.index)
    mctsAgentInstance.registerInitialState(gameState)
    mctsAgentInstance.observationHistory = self.observationHistory

    self.agentsPool = {
      "ATTACKING_AGENT": attackingAgent,
      "DEFENDING_AGENT": defendingAgent,
      "APPROXIMATING_AGENT": approxAttackAgent,
      "MCTS_AGENT": mctsAgentInstance
    }

    # get current agent
    self.agent = self.getAgent(gameState)

    # Bottleneck Calculation
    self.foodBottleNeckThresNoCapsuleMap = 0.7
    self.foodBottleNeckThresOnCapsuleMap = 0.6
    self.boundaryPos = self.getBoundaryPos(gameState, 1)
    self.enemyBoundaryPos = self.getBoundaryPos(gameState, span=1, defMode=False)
    self.masterCapsules = self.getCapsulesYouAreDefending(gameState)
    self.offensiveCapsules = self.getCapsules(gameState)
    self.isNoCapsuleMap = True if len(self.offensiveCapsules) == 0 else False
    self.masterFoods = self.getFoodYouAreDefending(gameState).asList()
    tic = time.perf_counter()
    self.bottleNeckCount, self.bottleNeckPos = self.getTopkBottleneck(gameState, 5)
    self.offBottleNeckCount, self.offBottleNeckPos = self.convertBottleNeckOff(gameState, self.bottleNeckCount,
                                                                               self.bottleNeckPos)
    self.offBottleNecksDepth = self.getDepthPos(gameState, self.offBottleNeckPos, self.offBottleNeckCount)
    if FOOD_DEPTHS is None:
      FOOD_DEPTHS = self.offBottleNecksDepth
    self.depthFoodCounts = len([d for d in FOOD_DEPTHS.values() if d <= DEPTH_LIMIT])
    toc = time.perf_counter()
    if PDDL_DEBUG: print(f"Ran in {toc - tic:0.4f} seconds")

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
      return self.agentsPool["DEFENDING_AGENT"]
    else:
      self.AGENT_MODE = "DEFEND"
      if gameState.isOnRedTeam(self.index):
        RED_DEFENDERS += 1
      else:
        BLUE_DEFENDERS += 1
      return self.agentsPool["DEFENDING_AGENT"]

  def getApproximateModeAgent(self, isMcts=False):
    self.AGENT_MODE = "APPROXIMATE"
    if isMcts:
      return self.agentsPool["MCTS_AGENT"]
    else:
      return self.agentsPool["APPROXIMATING_AGENT"]

  def getMCTSAgent(self):
    return self.agentsPool["MCTS_AGENT"]

  def chooseAction(self, gameState):
    global AGENT_1_POSITION, AGENT_2_POSITION, ANTICIPATER
    global AGENT_1_MODE, AGENT_2_MODE, DEPTH_LIMIT, TOTAL_FOOD_COLLECTED
    global POWERED, POWERED_TIMER
    global DEPTH_LIMIT_ON, DEPTH_LIMIT
    if PDDL_DEBUG: print(f"======MODE: #{self.index} - {self.AGENT_MODE}======")
    start = time.time()
    try:
      with time_limit(self.TIME_LIMIT):
        myCurrPos = gameState.getAgentPosition(self.index)
        myState = gameState.getAgentState(self.index)
        scared = True if myState.scaredTimer > 5 else False
        foods = self.getFood(gameState).asList()
        numGhosts = self.numberOfGhosts(gameState)
        # decide if the map has no capsules left
        self.isNoCapsuleMap = True if len(self.getCapsules(gameState)) == 0 else False

        # Bayesian Inference
        if self.index == 0 or self.index == 1 or len(ANTICIPATER) == 0:
          ANTICIPATER = self.getAnticipatedGhosts(gameState)
          if PDDL_DEBUG: print(f"Ghosts Anticipated: {[ghostPos for ghost, ghostPos in ANTICIPATER]}")

        # Update Agent Powered Status
        self.updatePoweredStatus(myCurrPos, gameState, self.getPreviousObservation())

        # sync agents mode
        self.syncAgentMode()

        # no capsules - no depth limit
        if len(self.masterCapsules) == 0:
          if PDDL_DEBUG: print("No capsules - no depth limit")
          DEPTH_LIMIT = -1
          DEPTH_LIMIT_ON = False

        # init agents
        if self.index == 0 or self.index == 1:
          self.initAgent1(myCurrPos, foods)
        else:
          self.initAgent2(myCurrPos)

        # check if agent is stuck
        # Both Agent 1 and attacking Agent 2
        if (self.index == 0 or self.index == 1):
          self.checkIfStuck(myCurrPos)
        elif (self.index == 2 or self.index == 3) and self.AGENT_MODE == "ATTACK":
          self.checkIfStuck(myCurrPos)

        # For Agent 1 and Agent 2 in attack mode
        if self.approximationMode and not self.areWeWinning(gameState):
          self.makeApproximationAgent()

        if self.AGENT_MODE == "APPROXIMATE":
          self.handleApproximationFlow(myCurrPos, gameState)

        # clean agents if dead
        if myCurrPos == self.start: self.cleanAgent(gameState)

        # if agent is at boundary
        if myCurrPos[0] == self.getBoundaryX(gameState):
          self.atBoundaryLogic(myCurrPos, gameState)

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

        # no capsule map - try mcts
        if self.isNoCapsuleMap and self.AGENT_MODE != "APPROXIMATE":
          if MCTS_DEBUG: print("No Capsule Map Now - Try MCTS")
          returnedAction = self.getMCTSAgent().chooseAction(gameState.deepCopy())
          if MCTS_DEBUG: print(f"MCTS Action: {returnedAction}")
          if returnedAction is not None:
            if MCTS_DEBUG: print('Eval time for agent %d: %.4f seconds' % (self.index, time.time() - start))
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

        if PDDL_DEBUG: print('Eval time for agent %d: %.4f seconds' % (self.index, time.time() - start))

        return nextAction

    except TimeoutException as e:
      if PDDL_DEBUG: print('==== Time Limit Exceeded: #%d: %.4f seconds ====' % (self.index, time.time() - start))
      newAction = self.agentsTimeoutFlow(gameState)
      if PDDL_DEBUG: print('Eval time for agent %d: %.4f seconds' % (self.index, time.time() - start))
      return newAction

    except Exception as e:
      if PDDL_DEBUG: print(e)
      newAction = self.agentsTimeoutFlow(gameState)
      if PDDL_DEBUG: print('Eval time for agent %d: %.4f seconds' % (self.index, time.time() - start))
      return newAction

  def checkAndGetBlockingDefensiveAgent(self, gameState):
    global DEFENSIVE_BLOCKING_MODE
    bestBottleNeckPos, bestBottleNeckCount = self.bottleNeckCount[0]
    cFood = len(self.masterFoods) - len(list(set(self.masterFoods) - set(self.bottleNeckPos[bestBottleNeckPos])))
    cCapsule = len([cap for cap in self.masterCapsules if cap in self.bottleNeckPos[bestBottleNeckPos]])
    bottleNeckFood = cFood / len(self.masterFoods)
    # it covers all capsules
    if cCapsule == len(self.masterCapsules):
      # 70% food blocked - No Capsule
      # 60% food blocked - Capsule Map
      if (len(self.masterCapsules) == 0
          and bottleNeckFood > self.foodBottleNeckThresNoCapsuleMap) \
        or (len(self.masterCapsules) > 0
            and bottleNeckFood > self.foodBottleNeckThresOnCapsuleMap):
        # then tell defender to go there and stop
        if PDDL_DEBUG: print(f"Turning defensive agent into blocking agent - {bottleNeckFood * 100}% covered")
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
      if PDDL_DEBUG: print("Died being blocker - switch back")
      self.agent = self.getDefendModeAgent(gameState)
      DEFENSIVE_BLOCKING_MODE = False
    else:
      bestBottleNeckPos, bestBottleNeckCount = self.bottleNeckCount[0]
      if myCurrPos != bestBottleNeckPos:
        return self.agent.chooseAction(gameState, {
          "BlockingPos": bestBottleNeckPos
        })
      else:
        if PDDL_DEBUG: print("STOP HERE")
        return "Stop"

  def makeApproximationAgent(self):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN
    global AGENT_1_AREA, AGENT_2_AREA
    self.movesApproximating += 1
    if PDDL_DEBUG: print(f"Approximate Move: {self.movesApproximating}")
    if self.AGENT_MODE != "APPROXIMATE" and self.AGENT_MODE == "ATTACK":
      # initilize ApproximateQLearning
      if PDDL_DEBUG: print("Turning Attack PDDL into Approximating Agent")
      if self.index == 0 or self.index == 1:
        self.lastFoodEaten = AGENT_1_FOOD_EATEN
        AGENT_1_AREA = None
      else:
        self.lastFoodEaten = AGENT_2_FOOD_EATEN
        AGENT_2_AREA = None
      self.agent = self.getApproximateModeAgent(isMcts=False)

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
      if PDDL_DEBUG: print("Food Eaten/Threshold Reached -> switching back to Attack PDDL")
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      self.agent = self.getAttackModeAgent(gameState)
    elif self.red and myCurrPos[0] <= self.getBoundaryX(gameState) - 3:
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      if PDDL_DEBUG: print("back home - attack PDDL ON - stop approximating")
      self.agent = self.getAttackModeAgent(gameState)
    elif not self.red and myCurrPos[0] >= self.getBoundaryX(gameState) + 3:
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      if PDDL_DEBUG: print("back home - attack PDDL ON - stop approximating")
      self.agent = self.getAttackModeAgent(gameState)

  def atBoundaryLogic(self, myCurrPos, gameState):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN, TOTAL_FOODS
    global TOTAL_FOOD_COLLECTED, DEPTH_LIMIT, DEPTH_LIMIT_ON
    if PDDL_DEBUG: print("At Boundary - Food Collected/Resetting")
    newScore = self.getScore(gameState)
    if self.index == 0 or self.index == 1:
      if newScore > self.currScore:
        TOTAL_FOODS -= AGENT_1_FOOD_EATEN
        self.currScore = newScore
        TOTAL_FOOD_COLLECTED += AGENT_1_FOOD_EATEN
      AGENT_1_FOOD_EATEN = 0
    else:
      if newScore > self.currScore:
        TOTAL_FOODS -= AGENT_2_FOOD_EATEN
        self.currScore = newScore
        TOTAL_FOOD_COLLECTED += AGENT_2_FOOD_EATEN
      AGENT_2_FOOD_EATEN = 0
    if TOTAL_FOOD_COLLECTED >= self.depthFoodCounts - 3:
      if PDDL_DEBUG: print("[FailSafe] Turning off depth limit")
      DEPTH_LIMIT_ON = False
      DEPTH_LIMIT = -1

  def setAgent1ClosestFood(self, myCurrPos, foods):
    global AGENT_1_CLOSEST_FOOD
    foodDists = [(food, self.getMazeDistance(myCurrPos, food)) for food in foods]
    if len(foodDists) > 0:
      minFoodPos = min(foodDists, key=lambda t: t[1])[0]
      AGENT_1_CLOSEST_FOOD = minFoodPos
    else:
      AGENT_1_CLOSEST_FOOD = None

  def agent1AttackingFlow(self, myCurrPos, foods, gameState):
    global DEPTH_LIMIT_ON
    if self.numberOfInvaders(gameState) == 2 and not DEPTH_LIMIT_ON:
      if self.red:
        if myCurrPos[0] <= self.agent.getBoundaryX(gameState):
          if PDDL_DEBUG: print("back home - defensive mode ON - heavy invaders")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if PDDL_DEBUG: print("Heavy Invaders - decrease threshold")
          return self.agent.chooseAction(gameState, {
            "threshold": 0.30
          })
      else:
        if myCurrPos[0] >= self.agent.getBoundaryX(gameState):
          if PDDL_DEBUG: print("back home - defensive mode ON - heavy invaders")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if PDDL_DEBUG: print("Heavy Invaders - decrease threshold")
          return self.agent.chooseAction(gameState, {
            "threshold": 0.30
          })

  def agent2AttackingFlow(self, myCurrPos, foods, gameState):
    if len(foods) <= 2:
      if PDDL_DEBUG: print("len(foods) <= defensive mode ON")
      self.agent = self.getDefendModeAgent(gameState)
    # come back home
    elif self.isEnemyEnteredTerritory(gameState):
      if self.red:
        if myCurrPos[0] <= self.agent.getBoundaryX(gameState):
          if PDDL_DEBUG: print("back home - defensive mode ON")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if PDDL_DEBUG: print("stay offensive - go back home")
          return self.agent.chooseAction(gameState, {
            "problemObjective": "COME_BACK_HOME"
          })
      else:
        if myCurrPos[0] >= self.agent.getBoundaryX(gameState):
          if PDDL_DEBUG: print("back home - defensive mode ON")
          self.agent = self.getDefendModeAgent(gameState)
        else:
          if PDDL_DEBUG: print("stay offensive - go back home")
          return self.agent.chooseAction(gameState, {
            "problemObjective": "COME_BACK_HOME"
          })

  def agent1DefendingFlow(self, scared, gameState):
    if scared:
      if PDDL_DEBUG: print("Turn into offensive")
      self.agent = self.getAttackModeAgent(gameState)
    elif self.numberOfInvaders(gameState) < 2:
      if PDDL_DEBUG: print("Invaders reduced - switching back to attack mode")
      self.agent = self.getAttackModeAgent(gameState)

  def agent2DefendingFlow(self, myCurrPos, gameState, nextAction):
    if self.allInvadersKilled(myCurrPos, gameState, nextAction) \
        or not self.isEnemyEnteredTerritory(gameState):
      if PDDL_DEBUG: print("EATEN ALL INVADERS | No enemy")
      # turn it into offensive
      self.agent = self.getAttackModeAgent(gameState)

  def agentsTimeoutFlow(self, gameState):
    newAction = None
    if ((self.index == 0 or self.index == 1) and self.AGENT_MODE == "ATTACK") \
        or ((self.index == 2 or self.index == 3) and self.AGENT_MODE == "ATTACK"):
      if PDDL_DEBUG: print(f"=== Get Approximate Action: #{self.index} ====")
      agent = self.agentsPool["APPROXIMATING_AGENT"]
      newAction = agent.chooseAction(gameState)
    else:
      if PDDL_DEBUG: print(f"==== Get Random Action: #{self.index} ====")
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
    self.initialDivergingStrategy = True if len(self.offensiveCapsules) == 0 else False
    AGENT_1_AREA = None
    AGENT_2_AREA = None
    # failsafe
    if self.AGENT_MODE == "APPROXIMATE":
      self.history = Queue()
      self.approximationMode = False
      self.movesApproximating = 0
      if PDDL_DEBUG: print("Turning Approximate agent into attacking agent")
      self.agent = self.getAttackModeAgent(gameState)

  def initAgent1(self, myCurrPos, foods):
    AGENT_1_POSITION = myCurrPos
    self.setAgent1ClosestFood(myCurrPos, foods)

  def initAgent2(self, myCurrPos):
    AGENT_2_POSITION = myCurrPos

###################
# Offensive Agent #
###################

class OffensivePDDLAgent(BaseAgent):
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
    self.hardFoodCarryingThreshold = 0.75

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
    food_len = len(self.masterFoods)

    # Create Object PDDl line definition of objects.
    objects = list()
    cells = [f'cell{pos[0]}_{pos[1]}' for pos in allPos]
    cells.append("- cells\n")
    foods = [f'food{i + 1}' for i in range(food_len)]
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
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0] + 1}_{pos[1]})\n')
      if (pos[0] - 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0] - 1}_{pos[1]})\n')
      if (pos[0], pos[1] + 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1] + 1})\n')
      if (pos[0], pos[1] - 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1] - 1})\n')

    return "".join(connected)

  def generatePddlFluent(self, gameState, features):
    """
    Function for creating PDDL fluents for the problem file.
    """
    global FOOD_DEPTHS, DEPTH_LIMIT, DEPTH_LIMIT_ON
    global POWERED, POWERED_TIMER
    # Set Pacman Position
    at_food = None
    pacmanPos = gameState.getAgentPosition(self.index)
    at_pacman = f'\t\t(at-pacman cell{pacmanPos[0]}_{pacmanPos[1]})\n'
    has_capsule = list()

    # Set Ghost(s) positions
    has_ghost = list()
    if len(ANTICIPATER) == 0:
      # won't work - fallback logic
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

      for ghost in ghosts:
        ghostPos = ghost.getPosition()
        if ghost.scaredTimer <= 3:
          has_ghost.append(f'\t\t(has-ghost cell{int(ghostPos[0])}_{int(ghostPos[1])})\n')
    else:
      if PDDL_DEBUG: print("Adding anticipater logic")
      for ghostState, ghostPos in ANTICIPATER:
        if ghostPos:
          if not ghostState.isPacman and ghostState.scaredTimer <= 3:
            has_ghost.append(f'\t\t(has-ghost cell{int(ghostPos[0])}_{int(ghostPos[1])})\n')

    # Set Food Position
    foods = self.getFood(gameState).asList()
    if DEPTH_LIMIT_ON and DEPTH_LIMIT >= 0:
      if PDDL_DEBUG: print(f"DEPTH_LIMIT: {DEPTH_LIMIT}")
      currFoods = self.getFood(gameState).asList()
      at_food = list()
      for i, foodPos in enumerate(FOOD_DEPTHS.keys()):
        foodDepth = FOOD_DEPTHS[foodPos]
        if foodPos in currFoods:
          if foodDepth <= DEPTH_LIMIT:
            if AGENT_1_CLOSEST_FOOD and self.index == 2 or self.index == 3:
              if foodPos != AGENT_1_CLOSEST_FOOD:
                at_food.append(f'\t\t(at-food food{i + 1} cell{foodPos[0]}_{foodPos[1]})\n')
            else:
              at_food.append(f'\t\t(at-food food{i + 1} cell{foodPos[0]}_{foodPos[1]})\n')
    else:
      if len(foods) != 0:
        if AGENT_1_CLOSEST_FOOD and self.index == 2 or self.index == 3:
          if PDDL_DEBUG: print(f"Avoid Food: {AGENT_1_CLOSEST_FOOD}")
          at_food = [f'\t\t(at-food food{i + 1} cell{food[0]}_{food[1]})\n'
                     for i, food in enumerate(foods)
                     if food != AGENT_1_CLOSEST_FOOD]
        else:
          at_food = [f'\t\t(at-food food{i + 1} cell{food[0]}_{food[1]})\n' for i, food in enumerate(foods)]

    # Set Capsule Position
    capsules = self.getCapsules(gameState)
    if not POWERED or (POWERED and POWERED_TIMER < 3):
      has_capsule = [f'\t\t(has-capsule cell{capsule[0]}_{capsule[1]})\n' for capsule in capsules]
    else:
      for capsule in capsules:
        features["blindSpots"].append(capsule)
      if PDDL_DEBUG: print("Powered -> Skipping other capsules")

    # add ghosts in blind spot
    if len(features["blindSpots"]) > 0:
      for blindSpot in features["blindSpots"]:
        has_ghost.append(f'\t\t(has-ghost cell{int(blindSpot[0])}_{int(blindSpot[1])})\n')

    fluents = list()
    fluents.append("\t(:init \n")
    fluents.append(at_pacman)
    if at_food:
      fluents.append("".join(at_food))
    fluents.append("".join(has_ghost))
    fluents.append("".join(has_capsule))
    if features["problemObjective"] == "DIE":
      if PDDL_DEBUG: print("WANT_TO_DIE")
      fluents.append(f"\t\t(want-to-die)\n")
    fluents.append(self.pddlFluentGrid)
    fluents.append("\t)\n")

    return "".join(fluents)

  def getFoodFluent(self, index, foodPos, agentPos, gameState):
    global FOOD_DEPTHS
    if foodPos in FOOD_DEPTHS:
      foodDepth = FOOD_DEPTHS[foodPos]
    else:
      # failsafe
      foodDepth = 0
    ghostState, minGhostDistance = self.getMinGhostAnticipatedDistance(agentPos)
    allowFoodAdd = True
    # if we are not winning - be cautious about the depth
    # if we are winning - take risk
    if not self.areWeWinning(gameState):
      if ghostState and minGhostDistance > 3 and ghostState.scaredTimer < 3:
        if foodDepth + 2 < minGhostDistance:
          # if PDDL_DEBUG: print(f"Explore Depth: {foodDepth}, Buffer:{foodDepth+2}, Distance: {minGhostDistance}")
          allowFoodAdd = True
        else:
          # if PDDL_DEBUG: print(f"Do not explore Depth: {foodDepth}, Buffer:{foodDepth+2}, Distance: {minGhostDistance}")
          allowFoodAdd = False
    if allowFoodAdd:
      return f'\t\t(at-food food{index + 1} cell{foodPos[0]}_{foodPos[1]})\n'
    else:
      return None

  def generatePddlGoal(self, gameState, features):
    """
    Function for creating PDDL goals for the problem file.
    """
    if PDDL_DEBUG: print(f'======New Offensive Action: #{self.index}========')

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
    if len(ANTICIPATER) == 0:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
      ghostsState = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    else:
      ghosts = [ghostPos for ghostState, ghostPos in ANTICIPATER
                if ghostPos and not ghostState.isPacman]
      ghostsState = [ghostState for ghostState, ghostPos in ANTICIPATER
                     if ghostPos and not ghostState.isPacman]

    ghostDistance = 999999
    scaredTimer = 99999
    if len(ghosts) > 0:
      ghostDistance, scaredTimer = self.getGhostDistanceAndTimers(pacmanPos, ghosts, ghostsState)
      thres = features["threshold"]

    if PDDL_DEBUG: print(f'Pacman at {pacmanPos}')

    if features["problemObjective"] is None:
      closestHome, closestCap = self.compareCapsuleAndHomeDist(gameState, pacmanPos)
      # gameTimeLeft decrease by 4 for every 1 move - anticipate + come back
      if ((closestHome * 4) + 100) >= gameTimeLeft:
        if PDDL_DEBUG: print(f"Timer Objective #1: {closestHome}")
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

  def getGhostDistanceAndTimers(self, pacmanPos, ghosts, ghostsState):
    dists = [self.getMazeDistance(pacmanPos, ghost) for ghost in ghosts]
    timers = [ghostState.scaredTimer for ghostState in ghostsState]
    ghostDistance = min(dists)
    scaredTimer = min(timers)
    if PDDL_DEBUG: print(f'Ghost Alert with Dist: {ghostDistance} | scaredTimer: {scaredTimer}')
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
    global DEFENSIVE_BLOCKING_MODE, POWERED
    if POWERED:
      if PDDL_DEBUG: print("Agent powered - no threshold")
      return False
    totalFoodEaten = AGENT_1_FOOD_EATEN + AGENT_2_FOOD_EATEN
    if self.currScore < 0 and totalFoodEaten + self.currScore > 0 \
        and DEFENSIVE_BLOCKING_MODE:
      if PDDL_DEBUG: print("Defensive Blocking: BACK HOME")
      return True
    try:
      foodEatenPer = totalFoodEaten / TOTAL_FOODS
    except:
      if PDDL_DEBUG: print("Re-syncing total foods")
      TOTAL_FOODS = len(self.getFood(gameState).asList())
      foodEatenPer = 0
    if PDDL_DEBUG: print(f"Relative Food Eaten: {round(foodEatenPer, 2) * 100}%")
    foodDists = [self.getMazeDistance(pacmanPos, food) for food in foods]
    if len(foodDists) > 0:
      minDistance = min([self.getMazeDistance(pacmanPos, food) for food in foods])
    else:
      minDistance = 99999
    # so close to food - eat and then run back
    if self.areWeWinning(gameState):
      if PDDL_DEBUG: print(f"Game Winning - Be bit greedy")
      flag = True if (foodEatenPer > threshold and minDistance > 1) else False
    else:
      if PDDL_DEBUG: print(f"Game is at Tie/Not Winning - Hard Threshold: {threshold * 100}%")
      flag = True if (foodEatenPer > threshold) else False
    return flag

  def addEatCapsuleObjective(self, goal):
    if PDDL_DEBUG: print('Objective #2')
    goal.append(f'\t\t(capsule-eaten)\n')
    return "EAT_CAPSULE"

  def goBackStartObjective(self, goal):
    if PDDL_DEBUG: print('Objective #3')
    goal.append(f'\t\t(at-pacman cell{self.start[0]}_{self.start[1]})\n')
    return "GO_START"

  def goBackHomeHardObjective(self, gameState, goal, pacmanPos):
    if PDDL_DEBUG: print('Objective #4')
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
    if PDDL_DEBUG: print('Objective #5')
    goal.append(f'\t\t(carrying-food)\n')
    return "EAT_FOOD"

  def goToBoundaryPoint(self, goal, features, gameState, pacmanPos):
    if PDDL_DEBUG: print('Objective goToBoundaryPoint')
    boundaryPoint = features["reachBoundaryPoint"]
    goal.append(f'\t\t(at-pacman cell{boundaryPoint[0]}_{boundaryPoint[1]})\n')
    return "GO_TO_BOUNDARY_POINT"

  def tryFallBackGoals(self, goal, features, gameState, pacmanPos):
    if features["problemObjective"] == "COME_BACK_HOME":
      if PDDL_DEBUG: print('Objective #6 [FALLBACK]')
      return self.goBackHomeHardObjective(gameState, goal, pacmanPos)
    elif features["problemObjective"] == "DIE":
      if PDDL_DEBUG: print('Objective #7 [FALLBACK]')
      goal.append(f'\t\t(die)\n')
      return "DIE"

  def goBackHomeWithFoodObjective(self, gameState, goal, pacmanPos):
    if PDDL_DEBUG: print('Objective #7')
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
    global ANTICIPATER
    features = {"problemObjective": None,
                "threshold": 0.65,
                "generateGrid": False,
                "blindSpots": [],
                "reachBoundaryPoint": None}

    if overridefeatures:
      if "problemObjective" in overridefeatures:
        if PDDL_DEBUG: print("Overriding problemObjective")
        features["problemObjective"] = overridefeatures["problemObjective"]
      if "threshold" in overridefeatures:
        if PDDL_DEBUG: print("Overriding threshold")
        features["threshold"] = overridefeatures["threshold"]
      if "reachBoundaryPoint" in overridefeatures:
        if PDDL_DEBUG: print("Overriding reachBoundaryPoint")
        features["reachBoundaryPoint"] = overridefeatures["reachBoundaryPoint"]
        features["problemObjective"] = "GO_TO_BOUNDARY_POINT"

    if DEFENSIVE_BLOCKING_MODE:
      if PDDL_DEBUG: print("Defensive blocking - Threshold down to 50%")
      features["threshold"] = 0.50

    agentPosition = gameState.getAgentPosition(self.index)
    if agentPosition == self.start:
      if self.index == 0 or self.index == 1:
        AGENT_1_FOOD_EATEN = 0
      else:
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
    if PDDL_DEBUG: print(f'Action Planner: {action}')

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
      if PDDL_DEBUG: print("No plan found for Objective #1")
      # try coming back home
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "EAT_FOOD":
      if PDDL_DEBUG: print("No plan found for Objective #2")
      # try coming back home
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "GO_START":
      if PDDL_DEBUG: print("No plan found for Objective #3")
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "GO_TO_BOUNDARY_POINT":
      if PDDL_DEBUG: print("No plan found for GO_TO_BOUNDARY_POINT - Normal Flow")
      features["problemObjective"] = None
      return self.getPlan(gameState, features)
    elif problemObjective == "DIE" or problemObjective == "COME_BACK_HOME":
      if PDDL_DEBUG: print("No plan found for Objective #4")
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
      if minGhostDistance == 2:
        if PDDL_DEBUG: print("!! Blind Spot - anticipate ghosts positions !!")
        ghostX, ghostY = minGhostPos
        if (ghostX + 1, ghostY) in allPos and (ghostX + 1, ghostY) not in capsules \
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

class ApproxQLearningOffense(BaseAgent):

  def registerInitialState(self, gameState):
    self.epsilon = 0.05
    self.alpha = 0.2
    self.discount = 0.9
    self.episodesSoFar = 0

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
    if PDDL_DEBUG: print(f'======New Offensive Approx Action: #{self.index}========')
    legalActions = gameState.getLegalActions(self.index)
    agentPosition = gameState.getAgentPosition(self.index)

    if PDDL_DEBUG: print(f'Pacman at {agentPosition}')

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
    if not util.flipCoin(self.epsilon):
      if PDDL_DEBUG: print("Try Exploiting")
      action = self.getPolicy(gameState)
    else:
      if PDDL_DEBUG: print("Try Exploring")
      action = random.choice(legalActions)
    if PDDL_DEBUG: print(f"Approx Action: {action}")

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

    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    agentPosition = gameState.getAgentPosition(self.agentInstance.index)
    x, y = agentPosition
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0

    dist = self.closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height)
    features.divideAll(10.0)
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
            if PDDL_DEBUG: print('Already in goal')
        else:
          if PDDL_DEBUG: print('No plan!')
    except:
      if PDDL_DEBUG: print('Something wrong happened with PDDL parsing')

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
      if PDDL_DEBUG: print('should never have ocurred!')
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
      if PDDL_DEBUG: print('Planner Returned Nothing.....')
      return "Stop"

  def get_coor_from_loc(self, loc):
    return loc.split("cell")[1].split("_")


###################
# Defensive Agent #
###################

class DefensivePDDLAgent(BaseAgent):
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

  def generatePddlObject(self, gameState, features=None):
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
      invaders = [f'invader{i + 1}' for i in range(invader_len)]
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
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0] + 1}_{pos[1]})\n')
      if (pos[0] - 1, pos[1]) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0] - 1}_{pos[1]})\n')
      if (pos[0], pos[1] + 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1] + 1})\n')
      if (pos[0], pos[1] - 1) in allPos:
        connected.append(f'\t\t(connected cell{pos[0]}_{pos[1]} cell{pos[0]}_{pos[1] - 1})\n')

    return "".join(connected)

  def generatePddlFluent(self, gameState, features=None):
    """
    Function for creating PDDL fluents for the problem file.
    """
    global ANTICIPATER
    # Set Self Position
    pacmanPos = gameState.getAgentPosition(self.index)
    at_ghost = f'\t\t(at-ghost cell{pacmanPos[0]}_{pacmanPos[1]})\n'

    # Set Invader(s) positions
    has_invaders = list()

    if features is None:
      if len(ANTICIPATER) == 0:
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        for i, invader in enumerate(invaders):
          invaderPos = invader.getPosition()
          has_invaders.append(f'\t\t(at-invader invader{i + 1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')
      else:
        if PDDL_DEBUG: print("Adding anticipater invader logic")
        for i, invaderTup in enumerate(ANTICIPATER):
          invaderState, invaderPos = invaderTup
          if invaderPos:
            if invaderState.isPacman:
              has_invaders.append(f'\t\t(at-invader invader{i + 1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')
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

  def generatePddlGoal(self, gameState, features=None):
    """
    Function for creating PDDL goals for the problem file.
    """
    global ANTICIPATER
    if PDDL_DEBUG: print(f'======New Defensive Action: #{self.index}========')
    goal = list()
    goal.append('\t(:goal (and\n')

    myPos = gameState.getAgentPosition(self.index)
    if PDDL_DEBUG: print(f'Ghost at {myPos}')
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

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    if features is None:
      if len(ANTICIPATER) == 0:
        # fallback logic
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        for i, invader in enumerate(invaders):
          invaderPos = invader.getPosition()
          goal.append(f'\t\t(not (at-invader invader{i + 1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')
      else:
        if PDDL_DEBUG: print("Adding anticipater invader logic")
        for i, invaderTup in enumerate(ANTICIPATER):
          invaderState, invaderPos = invaderTup
          if invaderPos:
            if invaderState.isPacman:
              goal.append(f'\t\t(not (at-invader invader{i + 1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')
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
      if PDDL_DEBUG: print(f'Handling #1')

    # If No Invaders are detected (Seen 5 steps)
    if not invaders:
      # If Food has not been eaten, Guard the Capsules or Foods
      if not Eaten:
        if myPos not in self.boundaryPos and len(enemyHere) == 0:
          if PDDL_DEBUG: print(f'Going to #1')
          goal.extend(self.generateRedundantGoal(self.boundaryPos, myPos))
        elif myPos not in self.masterCapsules and len(self.getCapsulesYouAreDefending(gameState)) > 0:
          if PDDL_DEBUG: print(f'Going to #2')
          capsules = self.getCapsulesYouAreDefending(gameState)
          goal.extend(self.shufflePddlGoal(capsules, myPos))
        else:
          if PDDL_DEBUG: print(f'Going to #3')
          goal.extend(self.generateRedundantGoal(foods, myPos))
      # If Food have been eaten Rush to the food location.
      else:
        if PDDL_DEBUG: print(f'Going to #4')
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

  def generatePddlProblem(self, gameState, features=None):
    """
    Generates a file for Creating PDDL problem file for current state.
    """
    problem = list()
    problem.append(f'(define (problem p{self.index}-ghost)\n')
    problem.append('\t(:domain ghost)\n')
    problem.append(self.generatePddlObject(gameState, features))
    problem.append(self.generatePddlFluent(gameState, features))
    problem.append(self.generatePddlGoal(gameState, features))
    problem.append(')')

    problem_file = open(f"{CD}/ghost-problem-{self.index}.pddl", "w")
    problem_statement = "".join(problem)
    problem_file.write(problem_statement)
    problem_file.close()
    return f"ghost-problem-{self.index}.pddl"

  def chooseAction(self, gameState, features=None):
    agentPosition = gameState.getAgentPosition(self.index)
    problem_file = self.generatePddlProblem(gameState, features)
    planner = PlannerFF(GHOST_DOMAIN_FILE, problem_file)
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    action = planner.get_legal_action(agentPosition, plannerPosition)
    if PDDL_DEBUG: print(f'Action Planner: {action}')
    return action

################################# MCTS AGENT ##############################

###########################################
# MCTS created Helper Classes & Functions #
###########################################

MANHATTAN_VISION_DISTANCE = 5


def nearByEnemies(our_agent_index: int, gameState: GameState):
  '''
  list the (enemy_index, position) of enemies that are nearby
  '''
  enemy_indexes = gameState.getBlueTeamIndices() if gameState.isOnRedTeam(
    our_agent_index) else gameState.getRedTeamIndices()
  enemy_states = [(gameState.getAgentState(enemy_index), enemy_index) for enemy_index in enemy_indexes]
  enemy_know_positions = [(enemy_state.getPosition(), enemy_index) for enemy_state, enemy_index in enemy_states \
                          if enemy_state.getPosition() is not None]
  my_pos = gameState.getAgentPosition(our_agent_index)
  close_enemies = [(enemy_pos, enemy_index) for enemy_pos, enemy_index in enemy_know_positions \
                   if util.manhattanDistance(enemy_pos, my_pos) <= MANHATTAN_VISION_DISTANCE]
  return close_enemies


def dangerousEnemies(our_agent_index: int, gameState: GameState):
  ''' collect enemies that are capable of eating out pacman'''

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
  if not gameState.getAgentState(our_agent_index).isPacman:
    return gameState

  dangerous_enemies = dangerousEnemies(our_agent_index, gameState)
  my_pos = gameState.getAgentPosition(our_agent_index)
  close_enemies = [(enemy_pos, enemy_index) for enemy_pos, enemy_index in dangerous_enemies if
                   util.manhattanDistance(enemy_pos, my_pos) <= 15]

  # update close note scare enemies to chase pacman
  enemy_best_move = {}  # key: enemy_index value: best_action
  for enemy_pos, enemy_index in close_enemies:
    enemy_actions = gameState.getLegalActions(enemy_index)
    _, best_action = min(
      [(distancer.getDistance(Actions.getSuccessor(enemy_pos, action), my_pos), action) for action in enemy_actions])

    enemy_best_move[enemy_index] = best_action

  # apply action to the state
  for enemy_index, best_action in enemy_best_move.items():
    gameState = gameState.generateSuccessor(enemy_index, best_action)

  return gameState


def simulateEnemyPacman(our_agent_index: int, gameState: GameState, distancer: Distancer,
                        enemy_escape_positions: t.List[t.Tuple[int, int]]):
  '''
  Takes a game state and an agent.
  If there are enemys pacmans that our_agent can see, and they are not super (our pacman is not scared):

  Enemy behaviour:
  If they can move to their boundary, and don't get closed to our agent, do so
  else: Do nothing

  Note: the reward given need to reflect the behaviour here
  e.g if we punish our agent for a enemy getting to the other side, our agent may run away so the enemy is out of screen, and the update doesn't occur

  Args:
      enemy_escape_positions: a list of positions/ goals the enemy pacman may be running toward
          this should be obtained with BaseAgent.getBoundaryPos(gameState, span = 1, defMode = False)
  Returns:
      the updated gameState
  '''
  # if we are in scared state, can't assume they will run away
  if gameState.getAgentState(our_agent_index).scaredTimer != 0:
    return gameState

  blue_team_index = gameState.getBlueTeamIndices()
  if our_agent_index in blue_team_index:
    enemy_indexes = gameState.getRedTeamIndices()
  else:
    enemy_indexes = gameState.getBlueTeamIndices()

  # find enemy Pacman
  enemy_states = [(gameState.getAgentState(enemy_index), enemy_index) for enemy_index in enemy_indexes]
  enemy_pacmans = [(enemy_state.configuration.getPosition(), enemy_index) for enemy_state, enemy_index in enemy_states
                   if (enemy_state.isPacman) and (enemy_state.configuration is not None)]
  my_pos = gameState.getAgentPosition(our_agent_index)
  close_enemy_pacmans = [(enemy_pos, enemy_index) for enemy_pos, enemy_index in enemy_pacmans if
                         enemy_pos is not None and distancer.getDistance(enemy_pos, my_pos) <= 50]

  enemy_next_action = {}
  for enemy_pos, enemy_index in close_enemy_pacmans:
    ## find all the action that minimise distance to an escape point
    enemy_actions = gameState.getLegalActions(enemy_index)  # this includes stop
    enemy_actions.remove(Directions.STOP)
    distanceToAgent = distancer.getDistance(enemy_pos, my_pos)

    min_escape_dist = {}
    for enemy_action in enemy_actions:
      next_enemy_pos = Actions.getSuccessor(enemy_pos, enemy_action)
      if distanceToAgent > 4:
        # distance to each of the escapes for new position
        new_distances = [distancer.getDistance(next_enemy_pos, escape_pos) for escape_pos in enemy_escape_positions]
        min_escape_dist[enemy_action] = min(new_distances)
      else:
        min_escape_dist[enemy_action] = -1 * distancer.getDistance(next_enemy_pos, my_pos)

    min_val = min(min_escape_dist.values())
    best_actions = [k for k, v in min_escape_dist.items() if v == min_val]

    # If there are tie between the min values then choose the one closest to the boundary
    if len(best_actions) > 1:
      action_dist = {}
      for action in best_actions:
        pos = Actions.getSuccessor(enemy_pos, action)
        action_dist[action] = min([distancer.getDistance(pos, escape_pos) for escape_pos in enemy_escape_positions])

      best_action = min(action_dist, key=action_dist.get)
    else:
      best_action = random.choice(best_actions) if best_actions else None

    if best_action is not None and best_action != Directions.STOP:
      enemy_next_action[enemy_index] = best_action

  for enemy_index, action in enemy_next_action.items():
    gameState = gameState.generateSuccessor(enemy_index, action)

  return gameState


def simulatedEnemyPacmanConsiderBoth(enemy_agent_index: int, gameState: GameState, distancer: Distancer,
                                     enemy_escape_positions: t.List[t.Tuple[int, int]]):
  if not gameState.getAgentState(enemy_agent_index).isPacman:
    return gameState

  # TODO: this doesn't acount for enemy pacmans at the border that could become ghosts
  dangerous_our_team = dangerousEnemies(enemy_agent_index, gameState)  # our agents
  enemy_pos = gameState.getAgentPosition(enemy_agent_index)
  enemy_actions = gameState.getLegalActions(enemy_agent_index)
  enemy_actions.remove(Directions.STOP)

  # find the closest enemy
  if dangerous_our_team != []:
    distances = [(distancer.getDistance(enemy_pos, pos), pos) for pos, index in dangerous_our_team]
    closest_our_team_dist, closest_our_team_pos = min(distances)

  # run away only if they are close
  if dangerous_our_team != [] and closest_our_team_dist < 5:

    dist_and_new_pos_action = []
    for enemy_action in enemy_actions:
      new_pos = Actions.getSuccessor(enemy_pos, enemy_action)
      dist = distancer.getDistance(closest_our_team_pos, new_pos)
      dist_and_new_pos_action.append((dist, new_pos, enemy_action))

    max_dist, _, _ = max(dist_and_new_pos_action)

    best = []
    for item in dist_and_new_pos_action:
      if item[0] == max_dist:
        best.append(item)

    # TODO: should add additional info here to consider other enemy, or route home etc
    best_action = random.choice(best)[2]

  else:
    min_escape_dist = collections.Counter()
    for enemy_action in enemy_actions:
      next_enemy_pos = Actions.getSuccessor(enemy_pos, enemy_action)

      new_distances = [distancer.getDistance(next_enemy_pos, escape_pos) for escape_pos in enemy_escape_positions]
      # update action only if this is the best we have seen so far
      if min(new_distances) < min_escape_dist.get(enemy_action, 10000):
        min_escape_dist[enemy_action] = min(new_distances)

    best_action, _ = min_escape_dist.most_common()[-1]

  newGameState = gameState.generateSuccessor(enemy_agent_index, best_action)

  return newGameState

def getTeamMateIndexes(current_agent_index, is_red, gameState):
  '''
  current_agent_index: CpatureAgent.index should be able to provide that info
  is_red: .isRed on CaptureAgent should be able to provide this info
  Return:
      a list of indexes of current_agents_index team mates
  '''
  if is_red:
    red_team = gameState.getRedTeamIndices()
    red_team.remove(current_agent_index)
    team = red_team
  else:
    blue_team = gameState.getBlueTeamIndices()
    blue_team.remove(current_agent_index)
    team = blue_team
  return team


# Tree debug
def takeActionInTree(node, actions: list):
  for action in actions:
    child_node = node.childrenNodes[action]
    node = child_node

  return node

class MctsAgent(BaseAgent):
  # class variables
  # this is where agents can store information to comunicate with each other
  agent0 = {}
  agent1 = {}
  agent2 = {}
  agent3 = {}

  def registerInitialState(self, gameState: GameState):
    self.start = gameState.getAgentPosition(self.index)
    # the following initialises self.red and self.distancer
    CaptureAgent.registerInitialState(self, gameState)


  def chooseAction(self, gameState: GameState):
    # if there are anticipated possitions update map to contain those locations
    try:
      # MCTS has not been set up to handle being scarred, pass control to PDDL
      if gameState.getAgentState(self.index).scaredTimer != 0:
        return None
      # don't need to collect the last 2 food
      if len(self.getFood(gameState).asList())< 3 :
        return None
      # check if game time running out
      our_escape_positions = self.getBoundaryPos(gameState, span=1, defMode=True)
      my_pos = gameState.getAgentPosition(self.index)
      min_escape_distance = min([self.distancer.getDistance(my_pos, escape_pos) for escape_pos in our_escape_positions])
      if gameState.data.timeleft/4 < min_escape_distance + 10:
        return None

      ### MCTS starts here
      if ANTICIPATER:
        enemy_indexes = self.getOpponents(gameState)
        # anticipator returns result ascending by agent index
        for enemy_index, (_, pos) in zip(enemy_indexes, ANTICIPATER):
          enemy_state = gameState.getAgentState(enemy_index)
          # how do we tell the index the anticipator was for
          if enemy_state.configuration is None:
            enemy_state.configuration = Configuration(pos, Directions.NORTH)

      problem = MctsProblem(gameState=gameState,
                            captureAgent=self,
                            discount=.9)

      mcts = Mcts(
        problem,
        tree_policy=UctTreePolicy(problem),
        simulation_policy=MonteCarloRandomSimulation(problem, depth=0),
        learning_rate=1,
        existing_tree=None  
      )

      if mcts.mctsIntialNode.num_times_visited > 300:
        # tree is already very big (and has been checked that it is relevant to the current state) don't grow it
        # could use this time for other activities
        pass
      else:
        mcts.iterate(time_limit_seconds=.7, number_of_itr=300)
      if MCTS_DEBUG: print(f"index {self.index}, depth {mcts.mctsIntialNode.num_times_visited}")

      ##### decide if use mcts tree #######
      # if far away from reward/punishment MCTS will no be helpful. Use helper to git it to a better location
      if self._useful_tree(self.index, mcts.mctsIntialNode):
        action = mcts.getAction()
      else:
        return None

      # always want to keep this so MCTS can try to pick up where it left off and find a better tree
      self.expected_next_tree = mcts.mctsIntialNode.childrenNodes[action]

      # store the explected path
      expected_actions = getBestPathFromNodeActionPosition(self.expected_next_tree, self.index)
      self_shared_info = getattr(MctsAgent, f"agent{self.index}")
      self_shared_info["expected_actions"] = expected_actions

      if MCTS_DEBUG: print(f"index {self.index}, action {action}, depth {mcts.mctsIntialNode.num_times_visited}")
      return action
    except TimeoutException as e:
      raise
    except Exception as e:
      print(f"----------------------\n----- MCTS ERROR --------\n{e}\n---------------------")
      return None

  def _useful_tree(self, agent_index, root_node):
    '''
    Takes a root node for a tree, and determines if it should be used to determine the action
    Args:
    root_node: MctsNode
    Result:
        Bool: True if the tree is deemed useful
    '''
    # if root positive, reward has been found. - can use tree
    # if root 0 - potentially nothing was found, potentially, there are only negative rewards around us that need to be avoided
    #   this is because the best value is propergated up the tree, which will often be zero if there are a lot of negative rewards
    if root_node.v <= 0:
      return False

    return True

  def _alternate_action(self, root_node):
    '''
    Return: a legal 'action' the agent should play when the tree is not useful
    '''
    # This is only used if MTCS is not called with other agents available
    gameState = root_node.state["gameState"]

    actions = gameState.getLegalActions(self.index)
    actions.remove("Stop")
    return random.choice(actions)


#############################
##########   Probem  ########
#############################
# abstract problem class defining the expected methods
class MDPproblem:
  '''
      MDP: < S, s_0, A(s), P_a(s' | s), r(s,a,s'), gamma >

      Model free techniques:
          P_a(s' | s) : not provided/required (# TODO: how does this work for back propergation)
          r(s,a,s')   : provided by simulator
  '''

  def __init__(self, discount):
    self.discount = discount  # discount factor

  def getStartState(self):
    util.raiseNotDefined()

  def getPossibleActions(self, state):
    # in actual Pacman implementation will likely want to remove the action STOP
    util.raiseNotDefined()

  def generateSuccessor(self, state, action, numberOfAgentTurns=None):  # P_a(s'|s)
    """
    This was getTransitionStatesAndProbs, but as pacman is deterministic we have simplifed to remove transition prbs

    Note that in Q-Learning and reinforcment
    learning in general, we do not know these
    probabilities nor do we directly model them.
    """
    util.raiseNotDefined()

  def getReward(self, state, action, nextState):  # r(s,a,s')
    """
    Get the reward for the state, action, nextState transition.

    Not available in reinforcement learning.
    """
    util.raiseNotDefined()

  def isTerminal(self, state):
    """
    Returns true if the current state is a terminal state.  By convention,
    a terminal state has zero future rewards.  Sometimes the terminal state(s)
    may have no possible actions.  It is also common to think of the terminal
    state as having a self-loop action 'pass' with zero reward; the formulations
    are equivalent.
    """
    util.raiseNotDefined()

  # all of the following quick methods are specifically for simulation policy.
  # these should likely just be seperated into a specific problem that is passed on SimulationPolicy only

  def convertToQuickState(self, state):
    '''
    Specifically for use with the Simulation policy
    this turns a regular state into a quick_state to the followin methods can be used on it:
        getPossibleActionQuick
        generateSuccessorQuick

    Converts a regular state into its quick state,
    '''
    util.raiseNotDefined

  def getPossibleActionsQuick(self, quick_state):
    '''
    Specifically for use with the Simulation policy
    This generated the succesor of a quick_state - a light weigh state that must be very quick
    Args:
        quick_state: a representation of the state.
            may be a light weight version of the real state as needs to be very fast
    Return:
        list of possible actions available
    '''
    util.raiseNotDefined()

  def generateSuccessorQuick(self, quick_state, action):
    '''
    Specifically for use with the Simulation policy
    This generated the succesor of a quick_state - a light weigh state that must be very quick
    Args:
        action: the action to be applice
        quick_state: a representation of the state.
            may be a light weight version of the real state as needs to be very fast
    Return:
        new_quick_state, representing the new quick_state caused by the action
    '''
    util.raiseNotDefined()

  def getRewardQuick(self, state_quick, action_quick, nextState_quick):  # r(s,a,s')
    """
    Specifically for use with the Simulation policy
    should find the reward specific to quick states
    """
    util.raiseNotDefined()

  def isTerminalQuick(self, state):
    """
    Specifically for use with the Simulation policy
    Returns true if the current state is a terminal state.  By convention,
    a terminal state has zero future rewards.  Sometimes the terminal state(s)
    may have no possible actions.  It is also common to think of the terminal
    state as having a self-loop action 'pass' with zero reward; the formulations
    are equivalent.
    """
    util.raiseNotDefined()

  def compareStatesEq(self, state1, state2):
    """
    Determines how the states should be compared for equality, As may not wish to consider all parts of the state object
    """
    util.raiseNotDefined()


class MctsProblem(MDPproblem):

  def __init__(self, gameState, captureAgent, discount=.9):

    self.init_gameState = gameState
    # this is not part of the state, as it doesn't change for the entire simulation
    self.agent = captureAgent
    # discount factor
    self.discount = discount

  def getStartState(self):
    return {"gameState": self.init_gameState,
            "our_agent_was_eaten": False,
            "enemy_agent_eaten": False,
            "enemy_agent_eaten_food_dropped": 0,
            "agent_food_eaten": 0,
            "score_change": 0}

  def getPossibleActions(self, state):
    # this will need to be fixed so the whole state is passed around
    gameState = state["gameState"]

    actions = gameState.getLegalActions(self.agent.index)
    actions.remove(Directions.STOP)
    return actions

  def generateSuccessor(self, state, action, numberOfAgentTurns=None):
    gameState = state["gameState"]
    our_agent_index = self.agent.index
    our_agent_red = self.agent.red
    enemies_index = self.agent.getOpponents(gameState)
    first_enemy_index = (our_agent_index + 1) % 4  # the first enemy is the one that will move directly after our turn
    second_enemy_index = (our_agent_index + 3) % 4
    original_enemies_state = [gameState.getAgentState(i) for i in enemies_index]

    # Track original state of game
    original_score = gameState.getScore()
    original_agent_state = gameState.getAgentState(our_agent_index)
    original_my_pos = original_agent_state.getPosition()
    original_enemy_pos = [enemy.getPosition() for enemy in original_enemies_state]
    original_food_carrying = original_agent_state.numCarrying

    ### Try to get our team mates expected move at this point
    # the first action in their list is the one they next expect to play
    # they will do this after our turn, update should be made after our agent performs its action
    team_mate_action = None
    if numberOfAgentTurns is not None:
      # get shared info about team mates NOTE: currently only supports 1 team mate
      team_index = getTeamMateIndexes(our_agent_index, our_agent_red, gameState)[0]
      team_info = getattr(MctsAgent, f"agent{team_index}")
      expected_action = team_info.get("expected_actions", [])

      # first tuen numberOfAgentTurns (which always relates to out agent will be 0)
      # after we make an action we want to applly our teams next turn ( the 0 item in our teams expected turns)
      if numberOfAgentTurns <= len(expected_action) - 1:
        # can now safely find the action we need
        team_mate_action, team_position_action_applied_from = expected_action[numberOfAgentTurns]

    ##### make gameState updates #####
    #####    My actions   ############
    succ_state = gameState.generateSuccessor(
      our_agent_index, action)

    ########## first_enemy_index############
    enemy_escape_positions = self.agent.getBoundaryPos(succ_state, span=1, defMode=False)
    # should assume worst case, that it is being chased both enemy's
    succ_state = simulateEnemyChase(
      our_agent_index=our_agent_index,
      gameState=succ_state,
      distancer=self.agent.distancer)

    succ_state = simulatedEnemyPacmanConsiderBoth(
      enemy_agent_index=first_enemy_index,
      gameState=succ_state,
      distancer=self.agent.distancer,
      enemy_escape_positions=enemy_escape_positions
    )

    ######## My team action ##########

    # Need to generate team mate action so the next state reflects what they will do
    # This goes at the reward/punishment our team mate collect should not be reward/punishment we recieve
    team_mate_score_change = 0
    if team_mate_action:
      # safety that team mate in in the same possition they expected to make that move from
      if succ_state.getAgentPosition(team_index) == team_position_action_applied_from:
        # if our team mate is too far away, stop considering their action
        #  as it makes the sate space much larger, and is less likely to effect us
        if self.agent.distancer.getDistance(team_position_action_applied_from, original_my_pos) < 7:
          succ_state = succ_state.generateSuccessor(team_index, team_mate_action)
          # metrics we need to track
          team_mate_score_change = succ_state.data.scoreChange

    ########## second_enemy_index############
    succ_state = simulatedEnemyPacmanConsiderBoth(
      enemy_agent_index=second_enemy_index,
      gameState=succ_state,
      distancer=self.agent.distancer,
      enemy_escape_positions=enemy_escape_positions
    )

    ################ record metrics
    new_enemies_state = [succ_state.getAgentState(i) for i in enemies_index]
    new_enemy_pos = [enemy.getPosition() for enemy in new_enemies_state]

    eaten_enemies = []
    for old_pos, new_pos in zip(original_enemy_pos, new_enemy_pos):
      if old_pos is None or new_pos is None:
        if MCTS_DEBUG: print("ERROR: a position had a None value")
        eaten_enemies.append(False)
      elif self.agent.distancer.getDistance(old_pos, new_pos) > 1:
        eaten_enemies.append(True)
      else:
        eaten_enemies.append(False)

    #### all gameStateupdates have been made, now track changes ######
    new_state = {"gameState": succ_state}
    new_agent_state = succ_state.getAgentState(our_agent_index)
    new_my_pos = new_agent_state.getPosition()
    new_food_carrying = new_agent_state.numCarrying
    new_score = succ_state.getScore()

    # track the score change
    # need to remove team mates effect on score
    new_state["score_change"] = new_score - original_score - team_mate_score_change

    # increase by 1 if eaten food (). max used as can decrease if eaten or return
    food_eaten = max(new_food_carrying - original_food_carrying, 0)
    new_state["agent_food_eaten"] = food_eaten

    # Compromise. Reward both agents when an enemy is eaten - even if only 1 was involved
    # eating enemies is expected to be uncommon, and well worth the trade off of eating an enemy
    # the down side is it will mess up the agents tree for that turn by giving a reward in a random place
    enemy_agent_eaten_food_dropped = 0
    enemy_agent_eaten = False
    for index, eaten in zip(enemies_index, eaten_enemies):
      if eaten:
        enemy_agent_eaten = True
        enemy_agent_eaten_food_dropped += gameState.getAgentState(index).numCarrying

    new_state["enemy_agent_eaten"] = enemy_agent_eaten
    new_state["enemy_agent_eaten_food_dropped"] = enemy_agent_eaten_food_dropped

    # Our agent always decides to move one step
    if self.agent.distancer.getDistance(new_my_pos, original_my_pos) > 1:
      new_state["our_agent_was_eaten"] = True
      new_state["our_agent_food_lost"] = original_food_carrying
    else:
      new_state["our_agent_was_eaten"] = False
      new_state["our_agent_food_lost"] = 0

    return new_state

  def getReward(self, state, action, nextState):  # r(s,a,s')
    """
    Get the reward for the state, action, nextState transition.

    - Distance to food ->
        - distance to the closest food in next state
    """
    reward = self.rewardLogic(state, action, nextState)

    return reward

  # Note: Under currect implementation Action and nextState need to be able to except None values
  def rewardLogic(self, state, action, nextState):
    # NOTE: the size of reward here affect if we do exploration or exploitation in UCB. Adjust carefully
    reward = 0
    gameState = state["gameState"]
    score_change = state["score_change"]

    # If red team, want scores to go up
    if score_change != 0:
      if self.agent.red == True:
        return score_change
      else:
        # want to give a positive reward when the score goes doesn't
        return -1 * score_change

    if state["our_agent_was_eaten"]:
      return -.1 + -1 * state["our_agent_food_lost"]

    if state["enemy_agent_eaten"]:
      return .1 + state["enemy_agent_eaten_food_dropped"]

    if state["agent_food_eaten"]:
      return .5

    return reward

  def isTerminal(self, state):
    # Kept for future extensibility
    return False

  def convertToQuickState(self, state):
    # Kept for future extensibility
    return state

  def getPossibleActionsQuick(self, quick_state):
    # Kept for future extensibility
    return self.getPossibleActions(quick_state)

  def generateSuccessorQuick(self, quick_state, action):
    # Kept for future extensibility
    return self.generateSuccessor(quick_state, action)

  def getRewardQuick(self, state_quick, action_quick, nextState_quick):
    return self.getReward(state_quick, action_quick, nextState_quick)

  def isTerminalQuick(self, state_quick):
    # Kept for future extensibility
    return self.isTerminal(state_quick)

  def compareStatesEq(self, state1, state2, consider_team=False):
    '''
    Defines hows the states should be checked for equality
    Args
    consider_team: if your team members states should also be considered
    '''
    state1 = state1["gameState"]
    state2 = state2["gameState"]
    for i in range(state1.getNumAgents()):
      config = state1.getAgentState(i).configuration
      if config is not None:
        config.direction = "North"

    for i in range(state2.getNumAgents()):
      config = state2.getAgentState(i).configuration
      if config is not None:
        config.direction = "North"

    return self._eq_gameState_helper(state1, state2, consider_team)

  def _eq_gameState_helper(self, gameState1, gameState2, consider_team):
    gameState_data1 = gameState1.data
    gameState_data2 = gameState2.data
    if not gameState_data1.food == gameState_data2.food: return False
    if not gameState_data1.capsules == gameState_data2.capsules: return False
    if not gameState_data1.score == gameState_data2.score: return False

    # remove agents that are on our team
    our_team_indexes = self.agent.getTeam(gameState1)
    our_team_indexes.remove(self.agent.index)
    team_mate_index = our_team_indexes[0]
    agentStates = list(zip(gameState_data1.agentStates, gameState_data2.agentStates))
    # comment the bellow if want to consider enemies possition (beyond their effect of the board e.g eating food)
    if not consider_team:
      del agentStates[team_mate_index]

    # agentStates no longer contains our team mate
    for agentState1, agentState2 in agentStates:
      if not self._eq_agentState_helper(agentState1, agentState2): return False

    return True

  def _eq_agentState_helper(self, agentState1, agentState2):
    if not agentState1.scaredTimer == agentState2.scaredTimer: return False
    if not self._eq_configuration_helper(agentState1.configuration, agentState2.configuration): return False
    return True

  def _eq_configuration_helper(self, config1, config2):
    if config1 is None or config2 is None:
      if config1 is None and config2 is None:
        return True
      return False

    return config1.pos == config2.pos


##############################
#########   Solver  ##########
##############################

##### helpers #####
class MctsNode:
  """
  MctsNode a node in the tree, with added properties for MCTS
  """

  def __init__(self, state, parentNode, action, available_actions):
    '''
    Args:
    state: object defined by the Problem class being used:
        could contain gameState, walls, score etc
    parentNode MctsNode: the parent of this current node/state
        this will be None for the root of the tree only
    action:
        the action performed on the parent node that gets to this node
        this will only be None for the root node
    Available_actions:
        This is only ever added to here - these values may be prunned if it is decided these states should not be considered

    '''
    self.num_times_visited = 0  # number of node visits (n)
    # the expected value of the best action from this state v(s) = max_a Q(s,a)
    self.v = 0

    # these to thing combine give us Q(s,a) which is what each node represents
    self.state = state
    self.action = action  # action used to arrive at the current node

    self.parentNode = parentNode  # parent to current node
    self.available_actions = available_actions
    self.dead = False

    # dictionary: key: action, value: childnode.
    # As pacman is deterministic, can assume only one child node per actions
    self.childrenNodes = {}

  def is_root(self):
    """ Check whether node is a root node or not """
    return self.parentNode == None

  # helper methods, not stricly required
  def getPerformedAction(self):
    return list(self.childrenNodes.keys())

  def createChild(self, action, state, child_available_actions):
    '''
    Takes an action and the state generated and uses the to initalise a child node
    '''
    child_node = MctsNode(state, parentNode=self, action=action, available_actions=child_available_actions)
    self.childrenNodes[action] = child_node

  def __str__(self):
    # TODO: this is not generic and to fixed
    agent_pos = self.state['gameState'].getAgentState(3).getPosition()
    return f"pos: {agent_pos}. Action: {self.action}. dead: {self.dead}. n {self.num_times_visited}. v: {round(self.v, 4)}"


def pprint_tree(node, agent_index, file=None, _prefix="", _last=True):
  agent_pos = node.state['gameState'].getAgentState(agent_index).getPosition()
  msg = f"pos: {agent_pos}. Action: {node.action}. dead: {node.dead}. n {node.num_times_visited}. v: {round(node.v, 4)}"
  node_children = list(node.childrenNodes.values())
  if MCTS_DEBUG: print(_prefix, "`- " if _last else "|- ", node, sep="", file=file)
  _prefix += "   " if _last else "|  "
  child_count = len(node_children)
  for i, child in enumerate(node_children):
    _last = i == (child_count - 1)
    pprint_tree(child, agent_index, file, _prefix, _last)


##### tree orchestrator #####
class Mcts():
  ''' executes a tree search for the problem provided.

      MTCS step:          Related policy:
      selection           tree policy
      expansion           tree policy
      simulation          simulation policy
      backprop

  '''

  def __init__(self, problem: MctsProblem, tree_policy, simulation_policy, learning_rate=1,
               existing_tree: t.Optional[MctsNode] = None):
    '''
    Args:
    existing_tree: an existing tree (e.g from a previously calculated iteration of Mcts).
        if the root node passed is equivalent to the stateState defined by the problem: exisiting_tree will be used
        else: a new tree will be initialised
    '''
    startState = problem.getStartState()
    available_action = problem.getPossibleActions(startState)

    if existing_tree is not None and problem.compareStatesEq(startState, existing_tree.state, consider_team=True):
      self.mctsIntialNode = existing_tree
    else:
      self.mctsIntialNode = self.initalise_tree(startState, available_action)

    self.tree_policy = tree_policy
    self.simulation_policy = simulation_policy
    self.problem = problem
    self.learning_rate = learning_rate

  def initalise_tree(self, state, available_actions):
    '''
    turn the intial state into a MCTS node
    Args:
        state: as defined by the problem class
    Return:
        MctsNode
    '''
    return MctsNode(state, parentNode=None, action=None, available_actions=available_actions)

  def iterate(self, time_limit_seconds: float, number_of_itr: int):
    """
    Run the MTCS to expand the tree
    Args:
        time_limit_seconds: how long can iterate for
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=time_limit_seconds)
    itr = 1

    while (datetime.now() - start_time < time_limit) and itr <= number_of_itr:

      node_to_simulate = self.tree_policy.select_node_prune(root_node=self.mctsIntialNode)

      simulation_reward = self.simulation_policy.estimate_value(node=node_to_simulate)

      node_to_simulate.v = simulation_reward
      self.backPropergate(node_to_simulate)
      itr += 1


  def backPropergate(self, simulated_node: MctsNode):
    """
    once we have a reward, need to back properagate up the tree until we reach the root node

    Standard equation:
    v(s) = max_a Q(s,a)
    where:
        Q(s,a) =  SUM Pa(s′|s)[r(s,a,s′)+γV(s′)]
        V(s') = node.v

    As the Pacman game has deterministic transitions, this simplifies to:

    v(s) = max_a Q(s,a)
    where:
        Q(s,a) =  r(s,a,s′) + γV(s′)
        V(s') = node.v

    """
    curr_node = simulated_node
    while not (curr_node.is_root()):  # this condition means we have reached the parent
      parent_node = curr_node.parentNode
      best_child_node = Mcts.childWithBestReward(parent_node)

      best_action = best_child_node.action
      best_child_state = best_child_node.state
      best_child_reward = best_child_node.v

      reward = self.problem.getReward(parent_node.state, best_action, best_child_state)
      parent_node.v = (1 - self.learning_rate) * parent_node.v \
                      + self.learning_rate * (reward + self.problem.discount * best_child_reward)

      curr_node = parent_node

  @staticmethod
  def childWithBestReward(node):
    child_reward_value = util.Counter()
    for action, childNode in node.childrenNodes.items():
      # shouldn't use dead nodes as they are repeats of existing nodes in the path
      if not childNode.dead:
        child_reward_value[childNode] = childNode.v

    return child_reward_value.argMax()

  def getAction(self):
    ''' from the tree structure get the next action '''

    best_child_node = Mcts.childWithBestReward(self.mctsIntialNode)
    return best_child_node.action

  @staticmethod
  def getBestPathFromNode(node: MctsNode):
    '''
    Find the best reward path in the MCTS tree
    '''
    actions = []
    while list(node.childrenNodes.keys()) != [] and not node.dead:
      best_child_node = Mcts.childWithBestReward(node)
      # when no child could be selected because all dead nodes
      # this happens when tree hasn't had enough time to propergate this up
      if best_child_node == None:
        break
      actions.append(best_child_node.action)
      node = best_child_node
    return actions


# methods here are specific to pacman trees, not added to the general mcts
def getBestPathFromNodeActionPosition(node: MctsNode, agent_index):
  '''
  Find the best reward path in the MCTS tree
  args:
  agent_index: the
  return:
  list: (action, position_action_applied_from)
  '''
  actions_and_result_pos = []
  while list(node.childrenNodes.keys()) != [] and not node.dead:
    best_child_node = Mcts.childWithBestReward(node)
    # when no child could be selected because all dead nodes
    # this happens when tree hasn't had enough time to propergate this up
    if best_child_node == None:
      break
    pos = node.state["gameState"].getAgentPosition(agent_index)
    actions_and_result_pos.append((best_child_node.action, pos))
    node = best_child_node
  return actions_and_result_pos


######################################
########### Tree Policy ##############
######################################


class TreePolicy:
  '''responsible for determining the next node to be selected for simulation from a MctsNode tree

      This should be overridden and the choose_child method updated
  '''

  def __init__(self, problem: MDPproblem):
    self.problem = problem  # required for getPossibleActions

  def select_node(self, root_node: MctsNode):
    '''
    stating from the root node, move through the tree to a node that needs to be simulated.
    updates the counts of nodes visited as it proceeds

    expected flow is as follows:
        Select a node in tree that has not had all it's available actions expanded
        Expand this node  - apply available one action that has not yet been seen,
            Add all children into the tree
            Select one of the children generated to be returned for simulation

    return this child
    (it will need to be simulated to get idea about the reward)
    '''
    node = root_node
    node.num_times_visited += 1
    while self.fully_expanded(node):
      child_node = self.choose_child(node)
      child_node.num_times_visited += 1
      node = child_node

    # choose an unexplored action to explore
    possible_actions = self.problem.getPossibleActions(node.state)
    actions_explored = node.getPerformedAction()
    unexplored_actions = list(
      set(possible_actions) - set(actions_explored))
    action_choosen = random.choice(unexplored_actions)

    # update the tree
    child_state = self.problem.generateSuccessor(
      node.state, action_choosen)
    node.createChild(action_choosen, child_state)
    child_node_to_simulate = node.childrenNodes[action_choosen]
    child_node_to_simulate.num_times_visited += 1
    return child_node_to_simulate

  def select_node_prune(self, root_node: MctsNode):
    '''
    Similar to the above method, but checks if the current state has already been visited in this path
    If this is the case, exploration of this node stops, as we know we have completed a loop to end up in an identical state
    '''
    # This needs to keep running until a suitable node for simulation is found
    while (True):
      node = root_node
      depth = 0
      node.num_times_visited += 1
      while self.fully_expanded(node):
        # check if not living children
        if not self.has_living_children(node):
          # if no living children we have hit a dead end and should restart
          if node.is_root():  # if this is the root node we have a problem
            raise RuntimeError("root node is dead")  # This required error handling be caller
          node.dead = True
          node = root_node  # restart search
          depth = 0
        else:
          # have found a suitable child to explore
          child_node = self.choose_child(node)
          child_node.num_times_visited += 1
          node = child_node
          depth += 1

      # check to ensure this node does become fully expanded while trying to find suitable child
      # if it does we need to start the seach again
      while not self.fully_expanded(node):
        # choose an unexplored action to explore
        possible_actions = node.available_actions
        simulation_candidate_nodes = []
        # generate all children
        for action_choosen in possible_actions:
          
          child_state = self.problem.generateSuccessor(node.state, action_choosen, numberOfAgentTurns=depth)
          child_available_actions = self.problem.getPossibleActions(child_state)
          node.createChild(action_choosen, child_state, child_available_actions)
          child_node = node.childrenNodes[action_choosen]
          path = self.getPathToNode(child_node)
          if self.check_state_in_path(path[:-1], child_state):
            child_node.dead = True
          else:
            # child could be simulated
            simulation_candidate_nodes.append(child_node)

        if simulation_candidate_nodes:
          # choose action to simulate
          child_to_simulate = random.choice(simulation_candidate_nodes)
          child_to_simulate.num_times_visited += 1
          return child_to_simulate

      # if it became fully expanded because all remaining unexplored actions are dead child nodes
      # in this chase node is dead ONLY if all children are dead (there may be some already expanded living ones)
      if not self.has_living_children(node):
        node.dead = True

  def choose_child(self, node):
    '''
    Uses some MAB technique to select the child node the most warrents exploring
    Override this with MultiArmBandit
    '''
    util.raiseNotDefined()

  def fully_expanded(self, node):
    """
    determine if all available actions have been applied to a node

    Likely steps:
    Takes a node and checks the actions it performs.
    Compares that with the available actions.
    If the lists are same then returns true, otherwise returns false.
    """
    possible_actions = node.available_actions
    performed_actions = node.getPerformedAction()
    return set(possible_actions) == set(performed_actions)

  def getPathToNode(self, node: MctsNode):
    '''
    Helper method for select_node_prune.
    return:
        list, starting root node, all the way to current node
    '''
    nodes = []
    nodes.append(node)
    while node.parentNode:
      nodes.append(node.parentNode)
      node = node.parentNode

    return list(reversed(nodes))

  def check_state_in_path(self, path: t.List[MctsNode], state):
    """
    Helper method for select_node_prune.
    return:
        boolean
    """
    for node in path:
      if self.problem.compareStatesEq(node.state, state):
        return True
    return False

  def has_living_children(self, node):
    '''
    Helper method for select_node_prune.
    '''
    for action, child in node.childrenNodes.items():
      if child.dead == False:
        return True
    return False


class UctTreePolicy(TreePolicy):

  def choose_child(self, node):
    Cp = 1 / math.sqrt(2)
    values_and_nodes = collections.Counter()

    for childNode in node.childrenNodes.values():
      # due to checks prior to this step, there should alway be a child that is not dead
      if childNode.dead:
        continue

      # to deal with the divide by zero error
      num_time_child_visited = childNode.num_times_visited if childNode.num_times_visited else .00001
      childValue = childNode.v + (Cp * math.sqrt((2 * math.log(node.num_times_visited)) / num_time_child_visited))
      values_and_nodes[childNode] = childValue

    best_child = values_and_nodes.most_common(1)
    best_child = best_child[0][0] if best_child != [] else None
    return best_child


########### simulation Policy ###############


class SimulationPolicy:
  '''
  Abstract class defining the required methods simulation is expected to provide
  '''

  # TODO: this needs to be very fast. Do not store any of the states we generate
  def __init__(self, problem):
    # this is required to generate the next state, and find reward
    self.problem = problem

  def estimate_value(self, node):
    '''
    Method for esitmating the V(s') where s' is the child node:
    This should be overridden for different techniques:
        e.g Monte Carlo simulation
            Q-approximation

    Args:
        state
    Return:
        float. extimate of the expected reward from this state

    Functions likely required:
        self.problem.getTransitionStatesAndProbs
        self.problem.getreward
        self.problem.isTerminal
    '''
    util.raiseNotDefined()


class MonteCarloRandomSimulation(SimulationPolicy):
  '''
  Use monte carlo simulation to gain a sample of the reward for a state
  '''

  def __init__(self, problem: MctsProblem, depth=None):
    '''
    Args:
        depth: The maximum depth to run a simulation to if havn't reached a terminal state
    '''
    self.problem = problem
    self.discount = problem.discount  # the discount factor
    self.depth = depth if depth is not None else 100

  def estimate_value(self, node: MctsNode):
    '''
    Args:
        state
    return:
        the simulation reward
    '''

    cumulative_reward = 0

    simulation_node = node
    simulation_node_quick_state = self.problem.convertToQuickState(
      simulation_node.state)
    from_quick_state = simulation_node_quick_state

    if self.depth == 0:
      # Do not get the rewards in future states, but still need the reward from the current state we are in
      cumulative_reward += self.problem.getRewardQuick(from_quick_state, None, None)
    else:
      for depth in range(self.depth):
        actions = self.problem.getPossibleActionsQuick(from_quick_state)
        
        next_action = random.choice(actions)
        next_state = self.problem.generateSuccessorQuick(
          from_quick_state, next_action)

        from_quick_state = next_state

        reward = self.problem.getRewardQuick(
          from_quick_state, next_action, next_state)
        # Note: no discounting should occurs here
        # when a reward is encountered on a series of random actions, depth f this occurance does not reflect distance of reward to pacman
        cumulative_reward += reward
        if self.problem.isTerminalQuick(from_quick_state):
          break

    return cumulative_reward