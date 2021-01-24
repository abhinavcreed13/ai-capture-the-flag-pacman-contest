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
import random, time, util
from game import Directions
import game
import sys, os, platform
import re
import subprocess
import math
from util import nearestPoint, Queue
from collections import Counter

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

AGENT_1_CLOSEST_FOOD = None
AGENT_2_CLOSEST_FOOD = None

ANTICIPATER = []

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


##########
# Agents #
##########

# 0 - OffensiveAgent
# 1 - OffensiveAgent
# 2 - DefensiveAgent
# 3 - DefensiveAgent

class MasterPDDLAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    global TOTAL_FOODS
    CaptureAgent.registerInitialState(self, gameState)
    self.AGENT_MODE = None
    self.agent = self.getAgent(gameState)
    self.start = gameState.getAgentPosition(self.index)
    TOTAL_FOODS = len(self.getFood(gameState).asList())

    # Bayesian Inference
    self.legalPositions = gameState.getWalls().asList(False)
    self.obs = {}
    for enemy in self.getOpponents(gameState):
      self.initalize(enemy, gameState.getInitialAgentPosition(enemy))

  """
      Bayesian Inference Functions Starts.
  """

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

  """
  Bayesian Inference Functions Ends.
  """

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
      agent = offensivePDDLAgent(self.index)
      agent.registerInitialState(gameState)
      agent.observationHistory = self.observationHistory
      return agent
    else:
      self.AGENT_MODE = "ATTACK"
      if gameState.isOnRedTeam(self.index):
        RED_ATTACKERS += 1
      else:
        BLUE_ATTACKERS += 1
      agent = offensivePDDLAgent(self.index)
      agent.registerInitialState(gameState)
      agent.observationHistory = self.observationHistory
      return agent

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
      agent = defensivePDDLAgent(self.index)
      agent.registerInitialState(gameState)
      agent.observationHistory = self.observationHistory
      return agent
    else:
      self.AGENT_MODE = "DEFEND"
      if gameState.isOnRedTeam(self.index):
        RED_DEFENDERS += 1
      else:
        BLUE_DEFENDERS += 1
      agent = defensivePDDLAgent(self.index)
      agent.registerInitialState(gameState)
      agent.observationHistory = self.observationHistory
      return agent

  def isEnemyEnteredTerritory(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    return len(enemyHere) > 0

  def numberOfInvaders(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    return len(enemyHere)

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

  def chooseAction(self, gameState):
    print(f"======MODE: {self.AGENT_MODE}======")
    start = time.time()
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN, TOTAL_FOODS
    global AGENT_1_CLOSEST_FOOD, AGENT_2_CLOSEST_FOOD
    myCurrPos = gameState.getAgentPosition(self.index)
    myState = gameState.getAgentState(self.index)
    scared = True if myState.scaredTimer > 5 else False
    foods = self.getFood(gameState).asList()

    # Agent 1
    # if (self.index == 0 or self.index == 1):
    #   # update ANTICIPATER
    #   ANTICIPATER = self.getAnticipatedGhosts(gameState)

    if myCurrPos[0] == self.getBoundaryX(gameState):
      print("Food Collected - Resetting")
      if self.index == 0 or self.index == 1:
        TOTAL_FOODS -= AGENT_1_FOOD_EATEN
        AGENT_1_FOOD_EATEN = 0
      else:
        TOTAL_FOODS -= AGENT_2_FOOD_EATEN
        AGENT_2_FOOD_EATEN = 0

    # Agent 1
    if self.AGENT_MODE == "ATTACK" \
        and (self.index == 0 or self.index == 1) and not scared:
      foodDists = [(food, self.getMazeDistance(myCurrPos, food)) for food in foods]
      minFoodPos = min(foodDists, key=lambda t: t[1])[0]
      AGENT_1_CLOSEST_FOOD = minFoodPos
      if self.numberOfInvaders(gameState) == 2:
        if self.red:
          if myCurrPos[0] <= self.agent.getBoundaryX(gameState):
            print("back home - defensive mode ON - heavy invaders")
            self.agent = self.getDefendModeAgent(gameState)
          else:
            print("Heavy Invaders - decrease threshold")
            return self.agent.chooseAction(gameState, {
              "threshold": 0.30
            })
        else:
          if myCurrPos[0] >= self.agent.getBoundaryX(gameState):
            print("back home - defensive mode ON - heavy invaders")
            self.agent = self.getDefendModeAgent(gameState)
          else:
            print("Heavy Invaders - decrease threshold")
            return self.agent.chooseAction(gameState, {
              "threshold": 0.30
            })

    # Agent 2
    if self.AGENT_MODE == "ATTACK" \
        and (self.index == 2 or self.index == 3):
      if myCurrPos == self.start:
        print("I died - create new offensive agent")
        self.agent = self.getAttackModeAgent(gameState)
      elif len(foods) <= 2:
        print("len(foods) <= defensive mode ON")
        self.agent = self.getDefendModeAgent(gameState)
      # come back home
      elif self.isEnemyEnteredTerritory(gameState):
        if self.red:
          if myCurrPos[0] <= self.agent.getBoundaryX(gameState):
            print("back home - defensive mode ON")
            self.agent = self.getDefendModeAgent(gameState)
          else:
            print("stay offensive - go back home")
            return self.agent.chooseAction(gameState, {
              "problemObjective": "COME_BACK_HOME"
            })
        else:
          if myCurrPos[0] >= self.agent.getBoundaryX(gameState):
            print("back home - defensive mode ON")
            self.agent = self.getDefendModeAgent(gameState)
          else:
            print("stay offensive - go back home")
            return self.agent.chooseAction(gameState, {
              "problemObjective": "COME_BACK_HOME"
            })

    nextAction = self.agent.chooseAction(gameState)

    # Agent 1
    if self.AGENT_MODE == "DEFEND" and (self.index == 0 or self.index == 1):
      if scared:
        print("Turn into offensive")
        self.agent = self.getAttackModeAgent(gameState)
      elif self.numberOfInvaders(gameState) < 2:
        print("Invaders reduced - switching back to attack mode")
        self.agent = self.getAttackModeAgent(gameState)

    # Agent 2
    if self.AGENT_MODE == "DEFEND" and (self.index == 2 or self.index == 3):
      # if scared:
      #   print("turn it into offensive")
      #   self.agent = self.getAttackModeAgent(gameState)
      if self.allInvadersKilled(myCurrPos, gameState, nextAction) \
          or not self.isEnemyEnteredTerritory(gameState):
        print("EATEN ALL INVADERS | No enemy")
        # turn it into offensive
        self.agent = self.getAttackModeAgent(gameState)

    print('Eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return nextAction

  def getBoundaryX(self, gameState):
    return gameState.data.layout.width / 2 - 1 if self.red else gameState.data.layout.width / 2

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
      #   print(f'Enemy is Pacman at {anticipatedPos}')
      # else:
      #   print(f'Enemy is Ghost at {anticipatedPos}')
      
    # print('===========Anticipator==============')

    return anticipatedGhosts
    # =============================================================

###################
# Offensive Agent #
###################

class offensivePDDLAgent(CaptureAgent):
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

  def isSurrounded(self, gameState):
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

  def generatePddlObject(self, gameState):
    """
    Function for creating PDDL objects for the problem file.
    """

    # Get Cell Locations without walls and Food count for object setup.
    allPos = gameState.getWalls().asList(False)
    food_len = len(self.getFood(gameState).asList())

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
    foods = self.getFood(gameState).asList()
    if len(foods) != 0:
      if AGENT_1_CLOSEST_FOOD and self.index == 2 or self.index == 3:
        print(f"Avoid Food: {AGENT_1_CLOSEST_FOOD}")
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
    print(f'======New Offensive Action: #{self.index}========')

    problemObjective = None
    gameTimeLeft = gameState.data.timeleft
    pacmanPos = gameState.getAgentPosition(self.index)
    foods = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)
    thres = features["threshold"]

    # Get History of locations
    if len(self.history.list) < 8:
      self.history.push(pacmanPos)
    elif len(self.history.list) == 8:
      # print(self.history.list)
      count = Counter(self.history.list).most_common()
      try:
        self.stuck = True if count and count[0][1] >= 3 and count[0][1] == count[1][1] else False
      except:
        print("!! STUCK PROBLEM !!")
        self.stuck = False
      if self.stuck: print('I am Stuck! Moving Out of Sight!')
      self.history.pop()
      self.history.push(pacmanPos)

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
      # print(f"Ghosts: {thres}")
    # else:
    #   print("Ghosts: 1.0")
    #   thres = 1

    print(f'Pacman at {pacmanPos}')

    if features["problemObjective"] is None:
      closestHome, closestCap = self.compareCapsuleAndHomeDist(gameState, pacmanPos)
      # gameTimeLeft decrease by 4 for every 1 move - anticipate + come back
      if ((closestHome * 4) + 50) >= gameTimeLeft:
        print("Objective #1")
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
    print(f'Ghost Alert with Dist: {ghostDistance} | scaredTimer: {scaredTimer}')
    return (ghostDistance, scaredTimer)

  def getBoundaryX(self, gameState):
    return gameState.data.layout.width / 2 - 1 if self.red else gameState.data.layout.width / 2

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
    totalFoodEaten = AGENT_1_FOOD_EATEN + AGENT_2_FOOD_EATEN
    foodEatenPer = totalFoodEaten/TOTAL_FOODS
    print(f"Relative Food Eaten: {round(foodEatenPer,2) * 100}%")
    # foodLeft = len(self.masterFoods) - self.foodEaten
    # foodCaryingPer = (foodLeft - len(foods)) / foodLeft
    minDistance = min([self.getMazeDistance(pacmanPos, food) for food in foods])
    # so close to food - eat and then run back
    flag = True if foodEatenPer > threshold and minDistance > 1 else False
    return flag

  def addEatCapsuleObjective(self, goal):
    print('Objective #2')
    goal.append(f'\t\t(capsule-eaten)\n')
    return "EAT_CAPSULE"

  def goBackStartObjective(self, goal):
    print('Objective #3')
    goal.append(f'\t\t(at-pacman cell{self.start[0]}_{self.start[1]})\n')
    return "GO_START"

  def goBackHomeHardObjective(self, gameState, goal, pacmanPos):
    print('Objective #4')
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
    print('Objective #5')
    goal.append(f'\t\t(carrying-food)\n')
    return "EAT_FOOD"

  def tryFallBackGoals(self, goal, features, gameState, pacmanPos):
    if features["problemObjective"] == "COME_BACK_HOME":
      print('Objective #6 [FALLBACK]')
      return self.goBackHomeHardObjective(gameState, goal, pacmanPos)
    elif features["problemObjective"] == "DIE":
      print('Objective #7 [FALLBACK]')
      goal.append(f'\t\t(die)\n')
      return "DIE"

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

  def chooseAction(self, gameState, overridefeatures = None):
    global AGENT_1_FOOD_EATEN, AGENT_2_FOOD_EATEN
    # global ANTICIPATER
    features = {"problemObjective": None,
                "threshold": 0.65,
                "generateGrid": False,
                "blindSpots":[]}

    if overridefeatures:
      if "problemObjective" in overridefeatures:
        print("Overriding problemObjective")
        features["problemObjective"] = overridefeatures["problemObjective"]
      if "threshold" in overridefeatures:
        print("Overriding threshold")
        features["threshold"] = overridefeatures["threshold"]

    agentPosition = gameState.getAgentPosition(self.index)
    if agentPosition == self.start:
      # I died
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
      # fallback failed -> die
      if plan is None:
        problemObjective = "DIE"
        plannerPosition, plan, \
        problemObjective, planner = self.tryFallbackPlans(gameState, features,
                                                          problemObjective)
    action = planner.get_legal_action(agentPosition, plannerPosition)
    print(f'Action Planner: {action}')

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

  def tryFallbackPlans(self, gameState, features, problemObjective):
    # no plan found for eating capsule
    if problemObjective == "EAT_CAPSULE":
      print("No plan found for Objective #1")
      # try coming back home
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "EAT_FOOD":
      print("No plan found for Objective #2")
      # try coming back home
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "GO_START":
      print("No plan found for Objective #3")
      features["problemObjective"] = "COME_BACK_HOME"
      return self.getPlan(gameState, features)
    elif problemObjective == "DIE" or problemObjective == "COME_BACK_HOME":
      print("No plan found for Objective #4")
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
    walls = gameState.getWalls().asList()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghostsPos = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghostsPos) > 0:
      ghostsDist = [(ghost, self.getMazeDistance(agentPosition, ghost)) for ghost in ghostsPos]
      minGhostPos, minGhostDistance = min(ghostsDist, key=lambda t: t[1])
      minGhostEucDistance = self.euclideanDistance(agentPosition, minGhostPos)
      # print(minGhostDistance, minGhostEucDistance)
      if minGhostDistance == 2:
        # if minGhostEucDistance == round(math.sqrt(2), 5):
        print("!! Blind Spot - anticipate ghosts positions !!")
        ghostX, ghostY = minGhostPos
        if (ghostX+1, ghostY) not in walls and (ghostX+1, ghostY) not in ghostsPos:
          features["blindSpots"].append((ghostX+1, ghostY))
        if (ghostX-1, ghostY) not in walls and (ghostX-1, ghostY) not in ghostsPos:
          features["blindSpots"].append((ghostX+1, ghostY))
        if (ghostX, ghostY-1) not in walls and (ghostX, ghostY-1) not in ghostsPos:
          features["blindSpots"].append((ghostX, ghostY-1))
        if (ghostX, ghostY+1) not in walls and (ghostX, ghostY+1) not in ghostsPos:
          features["blindSpots"].append((ghostX, ghostY+1))

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

###################
# Defensive Agent #
###################

class defensivePDDLAgent(CaptureAgent):
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
    self.pddlObject = self.generatePddlObject(gameState)
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
    enemy = 1 if self.red else -1
    xSpan = [x - i for i in range(span)] if self.red else [x + i for i in range(span)]
    walls = gameState.getWalls().asList()
    homeBound = list()
    for x in xSpan:
      pos = [(int(x), y) for y in range(layout.height) if (x, y) not in walls and (x+enemy, y) not in walls]
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

    # if len(ANTICIPATER) == 0:
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    for i, invader in enumerate(invaders):
      invaderPos = invader.getPosition()
      has_invaders.append(f'\t\t(at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')
    # else:
    #   for i, invaderVal in enumerate(ANTICIPATER):
    #     invaderState, invaderPos = invaderVal
    #     if invaderState.isPacman:
    #       has_invaders.append(f'\t\t(at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])})\n')

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
    print(f'======New Defensive Action{self.index}========')
    goal = list()
    goal.append('\t(:goal (and\n')

    myPos = gameState.getAgentPosition(self.index)
    print(f'Ghost at {myPos}')
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
    # if len(ANTICIPATER) == 0:
      # Find Invaders and set their location.
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    for i, invader in enumerate(invaders):
      invaderPos = invader.getPosition()
      goal.append(f'\t\t(not (at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')
    # else:
    #   for i, invaderVal in enumerate(ANTICIPATER):
    #     invaderState, invaderPos = invaderVal
    #     if invaderState.isPacman:
    #       invaders.append(invaderPos)
    #       goal.append(f'\t\t(not (at-invader invader{i+1} cell{int(invaderPos[0])}_{int(invaderPos[1])}))\n')

    if len(foods) < self.numFoodDef:
      Eaten = True
      targetFood = list(set(prevFoods) - set(foods))
      if targetFood:
        self.target = targetFood
    elif self.numFoodDef == len(foods):
      Eaten = False
      self.target = list()
      print(f'Handling #1')

    # If No Invaders are detected (Seen 5 steps)
    if not invaders:
      # If Food has not been eaten, Guard the Capsules or Foods
      if not Eaten:
        if myPos not in self.boundaryPos and len(enemyHere) == 0:
          print(f'Going to #1')
          goal.extend(self.generateRedundantGoal(self.boundaryPos, myPos))
        elif myPos not in self.masterCapsules and len(self.getCapsulesYouAreDefending(gameState)) > 0:
          print(f'Going to #2')
          capsules = self.getCapsulesYouAreDefending(gameState)
          goal.extend(self.shufflePddlGoal(capsules, myPos))
        else:
          print(f'Going to #3')
          goal.extend(self.generateRedundantGoal(foods, myPos))
      # If Food have been eaten Rush to the food location.
      else:
        print(f'Going to #4')
        if myPos in self.target:
          self.target.remove(myPos)
        goal.extend(self.shufflePddlGoal(self.target, myPos))


    goal.append('\t))\n')
    return "".join(goal)

  def generateRedundantGoal(self,compare,myPos):
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

  def generatePddlProblem(self, gameState):
    """
    Generates a file for Creating PDDL problem file for current state.
    """
    problem = list()
    problem.append(f'(define (problem p{self.index}-ghost)\n')
    problem.append('\t(:domain ghost)\n')
    # problem.append(self.pddlObject)
    problem.append(self.generatePddlObject(gameState))
    problem.append(self.generatePddlFluent(gameState))
    problem.append(self.generatePddlGoal(gameState))
    problem.append(')')

    problem_file = open(f"{CD}/ghost-problem-{self.index}.pddl", "w")
    problem_statement = "".join(problem)
    problem_file.write(problem_statement)
    problem_file.close()
    return f"ghost-problem-{self.index}.pddl"

  def chooseAction(self, gameState):
    # global ANTICIPATER
    agentPosition = gameState.getAgentPosition(self.index)
    problem_file = self.generatePddlProblem(gameState)
    planner = PlannerFF(GHOST_DOMAIN_FILE, problem_file)
    output = planner.run_planner()
    plannerPosition, plan = planner.parse_solution(output)
    action = planner.get_legal_action(agentPosition, plannerPosition)
    print(f'Action Planner: {action}')
    # actions = gameState.getLegalActions(self.index)
    return action
