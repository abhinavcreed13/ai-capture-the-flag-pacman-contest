'''
####################################
Basic idea:
####################################

There are 3 key step:
- Select a node in investigate/ update:
    Determine which node to expand (s),
    the action to apply (a)
     and the child state we are interested in (s')

- Get a score for a Q(s,a):
    expanded_node: s
    action_applied: a
    child_node: s'

    Expected value of an action is:
    Q(s,a):=  SUM Pa(s′|s)[r(s,a,s′)+γV(s′)]

    In this case we are intestested only in a specific s' (chosen above). That is this equations
    Pa(s′|s)[r(s,a,s′)+γV(s′)]

    V(s') is unknown and need to be found/approximated.
        options available:
            - Monte Carlo full search
            - Q learning/ Sarsa
            - Q- approximation (linear or otherwise)
            - Reward shapping
            - n-step learning
    -> can use any method that gives us some indication of the reward we could expect from this state

- backpropergate the reward found above:
    expanded node: s
    action applied: a
    child node: s'

    Q(s,a):=  SUM Pa(s′|s)[r(s,a,s′)+γV(s′)]
    v(s) = max Q(s,a)

    then apply to parent of expanded node
    parent node: s
    action applied: a
    child node: s' <- s (old expanded node)

    Do this all the way up the

###################################################
How the following functions relateds to the ideas outlined above::
###################################################

Mcts:
executes the MonteCarlo search. it combines all the above step of finding a node, get a score, back propergate
To do this it uses the following helper classes:

    TreePolicy: (find a node step)
        Abstact class to inherit from. Takes the root node, and returns a node that need to be simulate

    SimulationPolicy: (get a score)
        Abstact class to inherit from. Takes a node, and returns the expected score/reward for that node

    MDPProblem:
        Abstact class to inherit from. Defines all the important aspects of the problem
            possible actions
            transition
            etc.

MCTSNode:
Stores the state, plus any additional information required be the MCTS algothim to operate

############ Some likely concrete classes ###################
MonteCarloSimulation(SimulationPolicy):
    simulates as dicussed in class

UCBTreePolicy(TreePolicy):
uses UCB to select nodes.

'''

# Genetric module
import typing as t
import collections
import random
import time
import math
from datetime import timedelta, datetime

# Pacman imports
import util
from capture import GameState
from distanceCalculator import Distancer
from game import Directions, Actions, Configuration
from captureAgents import CaptureAgent


def createTeam(firstIndex, secondIndex, isRed,
               first='mctsTeamWorkAgent', second='mctsTeamWorkAgent'):
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


# INTERGRATION
ANTICIPATER = []


#######################################
# Existing Helper Classes & Functions #
#######################################

# Implementation of Ford-Fulkerson algorithm
# taken from https://github.com/bigbighd604/Python/blob/master/graph/Ford-Fulkerson.py

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


class EdgeDict(dict):
  '''
  Keeps a list of undirected edges. Doesn't matter what order you add them in.
  '''

  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)

  def __getitem__(self, key):
    return dict.__getitem__(self, tuple(sorted(key)))

  def __setitem__(self, key, val):
    return dict.__setitem__(self, tuple(sorted(key)), val)

  def __contains__(self, key):
    return dict.__contains__(self, tuple(sorted(key)))

  def getAdjacentPositions(self, key):
    edgesContainingKey = [edge for edge in self if key in edge]
    adjacentPositions = [[position for position in edge if position != key][0] for edge in edgesContainingKey]
    return adjacentPositions


##############
# Base Agent #
##############
# Base agent specfic imports
DEBUG = True
from util import Queue
from collections import Counter


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

    # Display The Distribution On the board
    # self.displayDistributionsOverPositions(self.obs.values())

    for enemy in self.getOpponents(gameState):
      anticipatedPos = self.approxPos(enemy)
      enemyGameState = gameState.getAgentState(enemy) if anticipatedPos else None
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
    dists = [self.getMazeDistance(pacmanPos, ghostPos) for ghostState, ghostPos in ANTICIPATER]
    minDis = min(dists) if dists else 999999
    return minDis

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

  def updatePoweredStatus(self, agentPos, gameState, prevgameState):
    global POWERED, POWERED_TIMER
    global DEPTH_LIMIT, DEPTH_LIMIT_ON
    capsules = self.getCapsules(prevgameState) if prevgameState else self.getCapsules(gameState)
    if agentPos in capsules:
      if DEBUG: print("PacMan Powered - No depth limit now")
      POWERED_TIMER = 40 * 2
      POWERED = True
      DEPTH_LIMIT = -1
      DEPTH_LIMIT_ON = False
    else:
      if POWERED_TIMER > 0:
        currMinDis = self.getMinGhostAnticipatedDistance(agentPos)
        spawnedGhostPos = self.isGhostSpawnedBack()
        if spawnedGhostPos:
          POWERED = True
          distance = self.getMazeDistance(agentPos, spawnedGhostPos)
          if DEBUG: print(f'Ghost Spawned: {spawnedGhostPos}, {distance}')
          if distance < POWERED_TIMER / 2:
            POWERED_TIMER -= distance + 2
          else:
            POWERED_TIMER -= 1
        else:
          POWERED = True
          POWERED_TIMER -= 1
      else:
        POWERED = False
        POWERED_TIMER = 0
        # DEPTH_LIMIT = 4

    if POWERED:
      print(f'POWERED ============== {POWERED_TIMER}')


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
  # NOTE: this is likely to be a very expensive method as it currently works with an entire game state object
  # calc is only applicable when we are a pacman
  if not gameState.getAgentState(our_agent_index).isPacman:
    return gameState

  dangerous_enemies = dangerousEnemies(our_agent_index, gameState)
  my_pos = gameState.getAgentPosition(our_agent_index)
  close_enemies = [(enemy_pos, enemy_index) for enemy_pos, enemy_index in dangerous_enemies if
                   util.manhattanDistance(enemy_pos, my_pos) <= MANHATTAN_VISION_DISTANCE]

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
                         enemy_pos is not None and distancer.getDistance(enemy_pos, my_pos) <= 6]

  enemy_next_action = {}
  for enemy_pos, enemy_index in close_enemy_pacmans:
    # find all the action that minimise distance to an escape point
    enemy_actions = gameState.getLegalActions(enemy_index)  # this includes stop

    min_escape_dist = collections.Counter()
    for enemy_action in enemy_actions:
      next_enemy_pos = Actions.getSuccessor(enemy_pos, enemy_action)
      # distance to each of the escapes for new position
      new_distances = [distancer.getDistance(next_enemy_pos, escape_pos) for escape_pos in enemy_escape_positions]
      min_escape_dist[enemy_action] = min(new_distances)

    best_action, _ = min_escape_dist.most_common()[-1] if min_escape_dist.most_common() != [] else (None, None)

    if best_action is not None and best_action != Directions.STOP:
      enemy_next_action[enemy_index] = best_action

  for enemy_index, action in enemy_next_action.items():
    gameState = gameState.generateSuccessor(enemy_index, action)

  return gameState


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


class mctsSoloAgent(BaseAgent):
  # class variables - these musst remain for other method - but will be empty
  agent0 = {}
  agent1 = {}
  agent2 = {}
  agent3 = {}

  # INTERGRATION - remove this whole method
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

    self.legalPositions = gameState.getWalls().asList(False)
    self.obs = {}
    for enemy in self.getOpponents(gameState):
      self.enemyStartPositions.append(gameState.getInitialAgentPosition(enemy))
      self.initalize(enemy, gameState.getInitialAgentPosition(enemy))

  def chooseAction(self, gameState: GameState):
    # Note: using the anticipator provides a good idea of where the enemies are
    # The down side is the tree can rarely be carried forward and used in the next round
    ANTICIPATER = self.getAnticipatedGhosts(gameState)

    # # if there are anticipated possitions update map to contain those locations
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
      existing_tree=getattr(self, "expected_next_tree", None)
    )

    if mcts.mctsIntialNode.num_times_visited > 500:
      # tree is already very big (and has been checked that it is relevant to the current state) don't grow it
      # could use this time for other activities
      pass
    else:
      mcts.iterate(time_limit_seconds=.8, number_of_itr=200)
      print(f"index {self.index}, depth {mcts.mctsIntialNode.num_times_visited}")

    ##### decide if use mcts tree #######
    # if far away from reward/punishment MCTS will no be helpful. Use helper to git it to a better location
    if self._useful_tree(self.index, mcts.mctsIntialNode):
      action = mcts.getAction()
      print("Tree used")
    else:
      action = self._alternate_action(mcts.mctsIntialNode)
      print("Tree NOT used")

    # always want to keep this so MCTS can try to pick up where it left off and find a better tree
    self.expected_next_tree = mcts.mctsIntialNode.childrenNodes[action]

    if action != None:
      return action
    else:
      return random.choice(gameState.getLegalActions(self.index))

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
    if root_node.v == 0:
      # if there are n
      if not nearByEnemies(agent_index, root_node.state["gameState"]):
        return False

    return True

  def _alternate_action(self, root_node):
    '''
    Return: a legal 'action' the agent should play when the tree is not useful
    '''
    # TODO: make this more helpful. Could potentially call one of the other fast agents
    gameState = root_node.state["gameState"]
    actions = gameState.getLegalActions(self.index)
    actions.remove("Stop")
    return random.choice(actions)


class mctsTeamWorkAgent(BaseAgent):
  # class variables
  # this is where agents can store information to comunicate with each other
  agent0 = {}
  agent1 = {}
  agent2 = {}
  agent3 = {}

  # INTERGRATION - remove this whole method
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
      
      self.legalPositions = gameState.getWalls().asList(False)
      self.obs = {}
      for enemy in self.getOpponents(gameState):
          self.enemyStartPositions.append(gameState.getInitialAgentPosition(enemy))
          self.initalize(enemy, gameState.getInitialAgentPosition(enemy))

  def chooseAction(self, gameState: GameState):
    ANTICIPATER = self.getAnticipatedGhosts(gameState)

    # if there are anticipated possitions update map to contain those locations
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
      mcts.iterate(time_limit_seconds=.8, number_of_itr=120)

    ##### decide if use mcts tree #######
    # if far away from reward/punishment MCTS will no be helpful. Use helper to get it to a better location
    if self._useful_tree(self.index, mcts.mctsIntialNode):
      action = mcts.getAction()
    else:
      action = self._alternate_action(mcts.mctsIntialNode)

    # always want to keep this so MCTS can try to pick up where it left off and find a better tree
    self.expected_next_tree = mcts.mctsIntialNode.childrenNodes[action]

    # store the explected path - this is crucial for team work
    expected_actions = getBestPathFromNodeActionPosition(self.expected_next_tree, self.index)
    self_shared_info = getattr(mctsTeamWorkAgent, f"agent{self.index}")
    self_shared_info["expected_actions"] = expected_actions

    print(f"index {self.index}, action {action}, depth {mcts.mctsIntialNode.num_times_visited}")
    if action != None:
      return action
    else:
      return random.choice(gameState.getLegalActions(self.index))

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
    #   this is because the best value is propergated up the tree, which will often be zero if there are a lot of negative rewards        if root_node.v == 0:
    if not nearByEnemies(agent_index, root_node.state["gameState"]):
        return False

    return True

  def _alternate_action(self, root_node):
    '''
    Return: a legal 'action' the agent should play when the tree is not useful
    '''
    # TODO: make this more helpful. Could potentially call one of the other fast agents
    gameState = root_node.state["gameState"]
    actions = gameState.getLegalActions(self.index)
    actions.remove("Stop")
    return random.choice(actions)


def pprint_tree2(node, agent_index, file=None, _prefix="", _last=True):
  agent_pos = node.state['gameState'].getAgentState(agent_index).getPosition()
  msg = f"pos: {agent_pos}. Action: {node.action}. dead: {node.dead}. n {node.num_times_visited}. v: {round(node.v, 4)}"
  node_children = list(node.childrenNodes.values())
  print(_prefix, "`- " if _last else "|- ", node, sep="", file=file)
  _prefix += "   " if _last else "|  "
  child_count = len(node_children)
  for i, child in enumerate(node_children):
    _last = i == (child_count - 1)
    pprint_tree(child, file, _prefix, _last)


#############################
##########   Probem  ########
#############################
# abstract problem class defining the expected methods
class MDPproblem:
  '''
      MDP: < S, s_0, A(s), P_a(s' | s), r(s,a,s'), gamma >

      Model free techniques:
          P_a(s' | s) : not provided/required 
          r(s,a,s')   : provided by simulator
  '''

  def __init__(self, discount):
    self.discount = discount  # discount factor

  def getStartState(self):
    util.raiseNotDefined()

  def getPossibleActions(self, state):
    # in actual Pacman implementation will likely want to remove the action STOP
    util.raiseNotDefined()

  def generateSuccessor(self, state, action):  # P_a(s'|s)
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

  def generateSuccessorQuick(self, quick_state, action, numberOfAgentTurns=None):
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

    # self.startState = gameState.getPacmanPosition()
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

    # Track original state of game
    original_score = gameState.getScore()
    original_agent_state = gameState.getAgentState(our_agent_index)
    original_my_pos = original_agent_state.getPosition()
    original_food_carrying = original_agent_state.numCarrying

    ### Try to get our team mates expected move at this point
    # the first action in their list is the one they next expect to play
    # they will do this after our turn, update should be made after our agent performs its action
    team_mate_action = None
    if numberOfAgentTurns is not None:
      # get shared info about team mates
      team_index = getTeamMateIndexes(our_agent_index, our_agent_red, gameState)[0]
      # get class info object
      team_info = getattr(mctsTeamWorkAgent, f"agent{team_index}")
      # expected action may or may not be there
      expected_action = team_info.get("expected_actions", [])

      # first tuen numberOfAgentTurns (which always relates to out agent will be 0)
      # after we make an action we want to applly our teams next turn ( the 0 item in our teams expected turns)
      if numberOfAgentTurns <= len(expected_action) - 1:
        # can now safely find the action we need
        team_mate_action, team_position_action_applied_from = expected_action[numberOfAgentTurns]

    ##### make gameState updates #####
    original_edible_enemies = edibleEnemies(our_agent_index, gameState)
    succ_state = gameState.generateSuccessor(
      our_agent_index, action)
    new_edible_enemies = edibleEnemies(our_agent_index, succ_state)

    succ_state = simulateEnemyChase(
      our_agent_index=our_agent_index,
      gameState=succ_state,
      distancer=self.agent.distancer)

    #### all gameStateupdates have been made, now track changes ######
    new_state = {"gameState": succ_state}
    new_agent_state = succ_state.getAgentState(our_agent_index)
    new_my_pos = new_agent_state.getPosition()
    new_food_carrying = new_agent_state.numCarrying
    new_score = succ_state.getScore()

    # track the score change
    new_state["score_change"] = new_score - original_score

    # increase by 1 if eaten food (). max used as can decrease if eaten or return
    food_eaten = max(new_food_carrying - original_food_carrying, 0)
    new_state["agent_food_eaten"] = food_eaten

    enemies_eaten = set(original_edible_enemies) - set(new_edible_enemies)
    enemy_agent_eaten_food_dropped = 0
    if enemies_eaten:
      enemy_agent_eaten = True
      for _, index in enemies_eaten:
        enemy_agent_eaten_food_dropped += gameState.getAgentState(index).numCarrying
    else:
      enemy_agent_eaten = False
    new_state["enemy_agent_eaten"] = enemy_agent_eaten
    new_state["enemy_agent_eaten_food_dropped"] = enemy_agent_eaten_food_dropped

    # Our agent always decides to move one step
    if self.agent.distancer.getDistance(new_my_pos, original_my_pos) > 1:
      new_state["our_agent_was_eaten"] = True
      new_state["our_agent_food_lost"] = original_food_carrying
    else:
      new_state["our_agent_was_eaten"] = False
      new_state["our_agent_food_lost"] = 0

    # Need to generate team mate action so the next state reflects what they will do
    # This goes at the reward/punishment our team mate collect should not be reward/punishment we recieve
    if team_mate_action:
      # safety that team mate in in the same possition they expected to make that move from
      if succ_state.getAgentPosition(team_index) == team_position_action_applied_from:
        # if our team mate is too far away, stop considering their action
        #  as it makes the sate space much larger, and is less likely to effect us
        if self.agent.distancer.getDistance(team_position_action_applied_from, original_my_pos) < 7:
          new_state["gameState"] = succ_state.generateSuccessor(team_index, team_mate_action)

    return new_state

  def getReward(self, state, action, nextState):  # r(s,a,s')
    """
    Get the reward for the state, action, nextState transition.

    - Distance to food ->
        - distance to the closest food in next state
    """
    # This function with use reward to be features * weights
    # reward = self.evaluateReward(state, action, nextState)
    reward = self.rewardLogic(state, action, nextState)

    return reward

  # Note: Under currect implementation Action and nextState need to be able to except None values
  def rewardLogic(self, state, action, nextState):
    # NOTE: the size of reward here affect if we do exploration or exploitation in UCB. Adjust carefully
    reward = 0
    gameState = state["gameState"]
    score_change = state["score_change"]

    # If red team, want scores to go up
    # Use max so that if the enemy makes the score go down (though us updating them) - we don't punish the agent
    if score_change != 0:
      if self.agent.red == True:
        return max(score_change, 0)
      else:
        # want to give a positive reward when the score goes doesn't
        return max(-1 * score_change, 0)

    if state["our_agent_was_eaten"]:
      return -.1 + -1 * state["our_agent_food_lost"]

    if state["enemy_agent_eaten"]:
      return .1 + state["enemy_agent_eaten_food_dropped"]

    if state["agent_food_eaten"]:
      return 1

    return reward

  def isTerminal(self, state):
    # NOTE: this is currently a place holder for future improvements
    return False

  def convertToQuickState(self, state):
    # NOTE: this is currently a place holder for future improvements
    return state

  def getPossibleActionsQuick(self, quick_state):
    # NOTE: this is currently a place holder for future improvements
    return self.getPossibleActions(quick_state)

  def generateSuccessorQuick(self, quick_state, action):
    # NOTE: this is currently a place holder for future improvements
    return self.generateSuccessor(quick_state, action)

  def getRewardQuick(self, state_quick, action_quick, nextState_quick):
    return self.getReward(state_quick, action_quick, nextState_quick)

  def isTerminalQuick(self, state_quick):
    # NOTE: this is currently a place holder for future improvements
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


class MctsRewardShappingProblem(MctsProblem):
  def rewardLogic(self, gameState, action, nextGameState):
    """
    Get features used for state evaluation.
    List of features considered or to be considered
        - Score in the successor state
        - Distance to the nearest enemy food
        - Distance to closest enemy ghost
        - Compute if agent is pacman
        - Distance to boundary position agent is in pacman state
        - Eating enemy in my side - check its food carrying
        - How much food am I carrying
    """
    features = util.Counter()
    weights = util.Counter()

    features = self.getFeatures(gameState, action, nextGameState)
    weights = self.getWeights(gameState, action, nextGameState)

    return features * weights

  def getFeatures(self, gameState, action, nextGameState):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    curr_state = gameState['gameState']
    successor = nextGameState['gameState']

    myPos = successor.getAgentState(self.agent.index).getPosition()

    foodList = self.agent.getFood(successor).asList()
    capsuleList = self.agent.getCapsules(successor)
    foodCarrying = successor.getAgentState(self.agent.index).numCarrying
    gameScore = self.agent.getScore(successor)
    (boundaryPositions, _) = getViableBoundaryPositions(self.agent, successor)
    initialAgentPosition = curr_state.getInitialAgentPosition(self.agent.index)
    # bottleNeckPositions = self.agent.bottleNeckPositions()

    # Move is reverse
    rev = Directions.REVERSE[curr_state.getAgentState(self.agent.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    # Next position leads to death
    if myPos is initialAgentPosition:
      features['death'] = 1

    # Distance to closest Capsule
    if len(capsuleList) > 0:
      features['closestCapsulePositionDistance'] = closestPositionDistance(self.agent, myPos, capsuleList)

    # Score in the successor state
    features['successorScore'] = gameScore

    # Amount of food agent is carrying
    if successor.getAgentState(self.agent.index).numCarrying > 0:
      features['foodCarrying'] = foodCarrying

      # Distance back to entry points
      features['closestBoundaryPositionDistance'] = closestPositionDistance(self.agent, myPos, boundaryPositions)

    # Distance to the nearest enemy food
    if len(foodList) > 0:
      features['distanceToClosestFood'] = closestPositionDistance(self.agent, myPos, foodList)

    # Distance to closest enemy ghost
    enemies = [successor.getAgentState(
      i) for i in self.agent.getOpponents(successor)]

    enemyDistances = [self.agent.getMazeDistance(myPos, enemy.getPosition())
                      for enemy in enemies if (not enemy.isPacman and enemy.getPosition() != None)]

    if len(enemyDistances) > 0:
      closestEnemyDistance = min(enemyDistances)
      if closestEnemyDistance <= 5:
        features['distanceToGhost'] = closestEnemyDistance

    return features

  def getWeights(self, gameState, action, nextGameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    successor = nextGameState['gameState']
    enemies = [successor.getAgentState(
      i) for i in self.agent.getOpponents(successor)]
    ghostScaredTime = min([enemy.scaredTimer for enemy in enemies])

    weights = util.Counter()

    weights['reverse']: -3
    weights['death']: -200
    weights['closestCapsulePositionDistance'] = 50
    weights['successorScore'] = 200
    weights['foodCarrying'] = 20
    weights['closestBoundaryPositionDistance'] = -2
    weights['distanceToClosestFood'] = -5
    weights['distanceToGhost'] = -10
    # weights['closestBottleneckPositionDistance'] = -15

    if ghostScaredTime > 5:
      weights['distanceToGhost'] = 0
      weights['closestCapsulePositionDistance'] = 0
      weights['death'] = 0
    return weights


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


def pprint_tree(node, file=None, _prefix="", _last=True):
  node_children = list(node.childrenNodes.values())
  print(_prefix, "`- " if _last else "|- ", node, sep="", file=file)
  _prefix += "   " if _last else "|  "
  child_count = len(node_children)
  for i, child in enumerate(node_children):
    _last = i == (child_count - 1)
    pprint_tree(child, file, _prefix, _last)


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

    if existing_tree is not None and problem.compareStatesEq(startState, existing_tree.state, consider_team=False):
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

  # this method could be part of the problem statement,
  # but the tree already contains all this information, so for now seem to make sense to have it here.
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
            raise RuntimeError("root node is dead")
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

  # NOTE: this needs to be very fast. Do not store any of the states we generate
  def __init__(self, problem):
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
        # Note: no discounting occurs here
        # when a reward is encountered on a series of random actions, depth f this occurance does not reflect distance of reward to pacman
        cumulative_reward += reward
        if self.problem.isTerminalQuick(from_quick_state):
          break

    return cumulative_reward


# This is a place holder for heuristic simulation development
class MonteCarloHeuristicSimulation(SimulationPolicy):
  '''
  Use monte carlo simulation with a heuristic to simulate the reward
  Note: this will generate more states
  '''

  def __init__(self, problem: MctsProblem, depth=None):
    '''
    Args:
        depth: The maximum depth to run a simulation to if havn't reached a terminal state
    '''
    self.problem = problem
    self.discount = problem.discount  # the discount factor
    self.depth = depth if depth else 100

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
    for depth in range(self.depth):

      actions = self.problem.getPossibleActionsQuick(from_quick_state)
      # ************ Exploring Heuristic Options ********
      next_states = [self.problem.generateSuccessorQuick(
        from_quick_state, next_action) for next_action in actions]

      idx = self.choose_successor(next_states)
      next_state = next_states[idx]
      next_action = actions[idx]

      # ************ Exploring Heuristic Options ********
      reward = self.problem.getRewardQuick(
        from_quick_state, next_action, next_state)

      cumulative_reward += (self.discount ** depth) * reward
      if self.problem.isTerminalQuick(from_quick_state):
        break

      from_quick_state = next_state

    return cumulative_reward

  def choose_successor(self, states: list):
    '''
    The may just be a random selections, or some heurisic to help it deal with sparse rewards
    NOTE: this method will likely depend of the shape of quick_state
    Args:
        state: as defined in the 'problem'
    return:
        the index of the state in the list that should be choosen
    '''

    return util.raiseNotDefined()


class MonteCarloModelSimulation(MonteCarloRandomSimulation):
  '''
  If we have a model that can predict value of a position we don't need to simulate the reward
  This is a place holher for that
  '''

  def estimate_value(self, node: MctsNode):
    '''
    Args:
        state
    return:
        the simulation reward
    '''
    util.raiseNotDefined()


class MonteCarloRewardBasedHeuristicSimulation(MonteCarloRandomSimulation):
  '''
  Use monte carlo simulation with a heuristic to simulate the reward
  Note: this will generate more states
  '''

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
    for depth in range(self.depth):

      actions = self.problem.getPossibleActionsQuick(from_quick_state)
      (next_state, reward) = self.choose_successor(from_quick_state, actions)

      cumulative_reward += (self.discount ** depth) * reward
      if self.problem.isTerminalQuick(from_quick_state):
        break

      from_quick_state = next_state

    return cumulative_reward

  # this method has been overriden to return reward as well as the state, as it is needed in the following step
  def choose_successor(self, from_quick_state, actions: list):
    '''
    The may just be a random selections, or some heurisic to help it deal with sparse rewards
    NOTE: this method will likely depend of the shape of quick_state
    Args:
        state: as defined in the 'problem'
    return:
        (the_next_state, heuristic_values_for_that_state)
    '''
    heuristic_value = util.Counter()

    next_states = [self.problem.generateSuccessorQuick(
      from_quick_state, next_action) for next_action in actions]

    for idx, next_state in enumerate(next_states):
      heuristic_value[idx] = self.problem.getRewardQuick(from_quick_state, actions[idx], next_state)

    idx_for_best = heuristic_value.argMax()

    return (next_states[idx_for_best], heuristic_value[idx_for_best]) 