# ########################    Over View   ############################
#
# Classical Planning follow the following pattern:
#
# Problem -> Solver -> Plan
#
# Problem = <S, s_0, S_g, A(s), T(a,s) -> s', C(a,s) -> int>
# Solver = explores the problem state space, finding a plan from the initial state to goal
# Plan = The list of actions to perform
#
# Blind and heuristic search:
# Problem:
# Achieved by subclassing the SearchProblem class
#     init:
#     getStartState:      <-->         s_0
#     isGoalState:        <-->         S_g
#         needs to recognise any goal state
#     getSuccessors:      <-->         ( A(s), T(a,s) -> s', C(a,s) -> int )
#
#         generates the next states from the current state
#         This is mainly the GameState object which tracks most of the information.
#         Plus any additional infomation isGoalState requires to recognise a goal
#
#         The following util provide are the work horse
#
#             Utilities:
#             gameState.getLegalActions(self.captureAgent.index)              <-->        A(s)
#             gameState.generateSuccessor(self.captureAgent.index, action)    <-->        T(a,s) ->s'
#
#     Note: by setting to correct goal, and use of a comprehensive world representation (through .generateSuccessor)
#           minimal manual rule encoding is required. It is best to keep the goal as close to the real goal as possible,
#           letting the algorithem do the work
#
# Solvers:
# All design to use the Search Problem interface.
# They have varying performance and optimality.
# - this has a performance on the time component portion, but will not improve the decisioning of the agent
#     - for this need to re-define your goal
#
#     Blind:
#         Take just a problem
#     Heuristic:
#         Take a problem and a search heuristic
#
# Plan:
# Is the set of action to reach the goal


########################### Guide for practicle implementation   ###############################
# Define the problem:
#    isGoalState: define goal state
#    getSuccesor: define node expansion
#    getStartState: define the inital staticmethod
#    init: store any info requred to set up the problem
#
#    Note: the state is likely a combination of GameState and ay other information the algorithem needs to track
#          As described above:
#          by setting to correct goal, and use of a comprehensive world representation (through .generateSuccessor)
#          minimal manual rule encoding is required. It is best to keep the goal as close to the real goal as possible,
#          letting the algorithem do the work

# Create an agents:
#   Create a class inheritting from CaptureAgent
#   register this class in 'createTeam'
#   define 'chooseAction'
#       - This must create the problem object above, and chose a solver to solve the problem
#       - Option include: Blind searches, Heuristic searches etc
#   use the plan returned by the solver to determine the agents next action

#########################  Notes on performance so far ###################################
# Pros:
# Avoids the ghosts actual position at it recognises this results in beng returned to the beginning - a longer path to acieving its goal
#

# Cons:
# Does not recognise the risk of being near a ghost states
# - this in an underlying issues of this family of techniques, as it can only recognise success state and failed state (binary). it is not suitable assigning continuous values to possitions
#       -  Requires explict programming of the goal that capute the risk of being near a ghost
#           - e.g add a flag to the state indicating if the pacman has been withing 1 unit of a ghost and remove
#               - this has it's own problems (e.g when pacman is being chased), and makes the solution less pure
#       - this could be better handled by solutions that automatically account for the expected reward in each state - which captures the risk of being near a ghost
#
# ~ TODO: does this one count:
#   limited to explict information it its state space

import util
from contextlib import contextmanager
import signal
import typing as t

# Standard imports
from captureAgents import CaptureAgent
import distanceCalculator
import random
from game import Directions, Actions  # basically a class to store data
import game

# My imports
from capture import GameState
import logging


#logging.basicConfig(filename="agent_runtime_log.log", level = logging.DEBUG)


# this is the entry points to instanciate you agents
def createTeam(firstIndex: int, secondIndex: int, isRed: bool,
               first: str = 'offensiveAgent', second: str = 'defensiveAgent') -> t.List[CaptureAgent]:

    # capture agents must be instanciated with an index
    # time to compute id 1 second in real game
    return [eval(first)(firstIndex, timeForComputing=1), eval(second)(secondIndex, timeForComputing=1)]


class agentBase(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState: GameState) -> None:
        """
        Required.

        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        self.start = gameState.getAgentPosition(self.index)
        # the following initialises self.red and self.distancer
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState: GameState) -> Directions:
        """
        Required.

        This is called each turn to get an agent to choose and action

        Return:
        This find the directions by going through gameState.getLegalAction
        - don't try and generate this by manually
        """
        actions = gameState.getLegalActions(self.index)

        return random.choice(actions)


class offensiveAgent(agentBase):

    def chooseAction(self, gameState: GameState) -> Directions:
        # steps:
        # Build/define problem
        # Used solver to find the solution/path in the problem~
        # Use the plan from the solver, return the required action

        problem = FoodOffenseWithAgentAwareness(
            startingGameState=gameState, captureAgent=self)
        try:
            with time_limit(1):
                actions = aStarSearch(problem, heuristic=offensiveHeuristic)
            # this can occure if start in the goal state. In this case do not want to perform any action.
            if actions == []:
                actions == ["Stop"]

        except TimeoutException as e:
            print("TimeoutException")
            actions = [random.choice(gameState.getLegalActions(self.index))]
            #logging.exception("SolutiionNotFound: "f"initial state:\n{str(gameState)}"f"problem info: expanded: {problem.expanded}. MINIMUM_IMPROVEMENT: {problem.MINIMUM_IMPROVEMENT} ")

        except SolutionNotFound as e:
            print("NotSolutionFound")
            actions = [random.choice(gameState.getLegalActions(self.index))]
            #logging.exception("SolutiionNotFound: "f"initial state:\n{str(gameState)}"f"problem info: expanded: {problem.expanded}. MINIMUM_IMPROVEMENT: {problem.MINIMUM_IMPROVEMENT} ")

        # logging.info(actions)
        #logging.info(f"Agent position: {gameState.getAgentPosition(self.index)}" f"Number of expanded states {problem.expanded}")
        return actions[0]

        return actions[0]


#################  problems and heuristics  ####################

def uniform_agent_direction(gameState):
    '''
    the agent direction is considered when checking for equality of game state.
    This is not important to us and creates more states than required, so set them all to be constant
    '''
    default_direction = Directions.NORTH

    for agent_state in gameState.data.agentStates:
        if agent_state.configuration:
            agent_state.configuration.direction = default_direction
        else:
            pass  # this happens when non enemy agent is visible - not required to do anything here

    return gameState


class FoodOffenseWithAgentAwareness():
    '''
    This problem extends FoodOffense by updateing the enemy ghost to move to our pacman if they are adjacent (basic Goal Recognition techniques).
    This conveys to our pacman the likely effect of moving next to an enemy ghost - but doesn't prohibit it from doing so (e.g if Pacman has been trapped)

    Note: This is a SearchProblem class. It could inherit from search.Search problem (mainly for conceptual clarity).
    '''

    def __init__(self, startingGameState: GameState, captureAgent: CaptureAgent):
        """
        Your goal checking for the CapsuleSearchProblem goes here.
        """
        self.expanded = 0
        self.startingGameState = uniform_agent_direction(startingGameState)
        # Need to ignore previous score change, as everything should be considered relative to this state
        self.startingGameState.data.scoreChange = 0
        self.MINIMUM_IMPROVEMENT = 1
        self.DEPTH_CUTOFF = 1
        # WARNING: Capture agent doesn't update with new state, this should only be used for non state dependant utils (e.g distancer)
        self.captureAgent: CaptureAgent = captureAgent
        self.goal_state_found = None

    def getStartState(self):
        # This needs to return the state information to being with
        return (self.startingGameState, self.startingGameState.getScore())

    def isGoalState(self, state: t.Tuple[GameState]) -> bool:
        """
        Your goal checking for the CapsuleSearchProblem goes here.
        """
        # Goal state when:
        # - Pacman is in our territory
        # - has eaten x food: This comes from the score changing
        # these are both captured by the score changing by a certain amount

        # Note: can't use CaptureAgent, at it doesn't update with game state
        gameState = state[0]

        # If red team, want scores to go up
        if self.captureAgent.red == True:
            if gameState.data.scoreChange >= self.MINIMUM_IMPROVEMENT:
                self.goal_state_found = state
                return True
            else:
                False
        # If blue team, want scores to go down
        else:
            if gameState.data.scoreChange <= -self.MINIMUM_IMPROVEMENT:
                self.goal_state_found = state
                return True
            else:
                False

    def getSuccessors(self, state: t.Tuple[GameState], node_info: t.Optional[dict] = None) -> t.List[t.Tuple[t.Tuple[GameState], Directions, int]]:
        """
        Your getSuccessors function for the CapsuleSearchProblem goes here.
        Args:
          state: a tuple combineing all the state information required
        Return:
          the states accessable by expanding the state provide
        """
        # TODO: it looks like gameState does not know the behaviour of when to end the game
        # - gameState.data.timeleft records the total number of turns left in the game (each time a player nodes turn decreases, so should decriment by 4)
        # - capture.CaptureRule.process handles actually ending the game, using 'game' and 'gameState' object
        # As these rules are not capture (enforced) within out gameState object, we need capture it outselves
        # - Option 1: track time left explictly
        # - Option 2: when updating the gameState, add additional information that generateSuccesor doesn't collect
        #           e.g set gameState.data._win to true. If goal then check gameState.isOver() is not true
        gameState = state[0]

        actions: t.List[Directions] = gameState.getLegalActions(
            self.captureAgent.index)
        # not interested in exploring the stop action as the state will be the same as out current one.
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        next_game_states = [gameState.generateSuccessor(
            self.captureAgent.index, action) for action in actions]

        # if planning close to agent, include expected ghost activity
        current_depth_of_search = len(node_info["action_from_init"])
        # we are only concerned about being eaten when we are pacman
        if current_depth_of_search <= self.DEPTH_CUTOFF and gameState.getAgentState(self.captureAgent.index).isPacman:
            self.expanded += 1  # track number of states expanded

            # make any nearby enemy ghosts take a step toward you if legal
            for i, next_game_state in enumerate(next_game_states):
                # get enemys
                current_agent_index = self.captureAgent.index
                enemy_indexes = next_game_state.getBlueTeamIndices() if next_game_state.isOnRedTeam(
                    current_agent_index) else next_game_state.getRedTeamIndices()

                # keep only enemies that are close enough to catch pacman.
                close_enemy_indexes = [
                    enemy_index for enemy_index in enemy_indexes if next_game_state.getAgentPosition(enemy_index) is not None]
                distancer = self.captureAgent.distancer
                my_pos = next_game_state.getAgentState(
                    current_agent_index).getPosition()
                adjacent_enemy_indexs = list(filter(lambda x: distancer.getDistance(
                    my_pos, next_game_state.getAgentState(x).getPosition()) <= 1, close_enemy_indexes))

                # check in enemies are in the right state
                adjacent_ghost_indexs = list(filter(lambda x: (not next_game_state.getAgentState(
                    x).isPacman) and (next_game_state.getAgentState(x).scaredTimer <= 0), adjacent_enemy_indexs))

                # move enemies to the pacman position
                ghost_kill_directions = []
                for index in adjacent_ghost_indexs:
                    position = next_game_state.getAgentState(
                        index).getPosition()
                    for action in Actions._directions.keys():
                        new_pos = Actions.getSuccessor(position, action)
                        if new_pos == my_pos:
                            ghost_kill_directions.append(action)
                            break

                # update state:
                for enemy_index, direction in zip(adjacent_ghost_indexs, ghost_kill_directions):
                    self.expanded += 1
                    next_game_state = next_game_state.generateSuccessor(
                        enemy_index, direction)

                # make the update
                next_game_states[i] = next_game_state
                # if they are next to pacman, move ghost to pacman possiton

        # As per the following discussion, this is a very expensive method, may need to fully control our own state if it proves to be an issue
        # https://github.com/COMP90054/Documentation/blob/master/FAQ-Pacman.md#i-have-performance-problem-with-generatesuccessor-in-my-search-implementation-why
        # ball park: 100s  of generated states okay, 1000's not
        successors = [((uniform_agent_direction(next_game_state),), action, 1)
                      for action, next_game_state in zip(actions, next_game_states)]

        return successors


# helpers
direction_map = {Directions.NORTH: (0, 1),
                 Directions.SOUTH: (0, -1),
                 Directions.EAST:  (1, 0),
                 Directions.WEST:  (-1, 0),
                 Directions.STOP:  (0, 0)}


def offensiveHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  
    """
    captureAgent = problem.captureAgent
    index = captureAgent.index
    gameState = state[0]

    # check if we have reached a goal state and explicitly return 0
    if captureAgent.red == True:
        if gameState.data.scoreChange >= problem.MINIMUM_IMPROVEMENT:
            return 0
    # If blue team, want scores to go down
    else:
        if gameState.data.scoreChange <= - problem.MINIMUM_IMPROVEMENT:
            return 0

    # continue with normal logc

    agent_state = gameState.getAgentState(index)
    food_carrying = agent_state.numCarrying

    myPos = gameState.getAgentState(index).getPosition()
    distancer = captureAgent.distancer

    # this will be updated to be closest food location if not collect enough food
    return_home_from = myPos

    # still need to collect food
    dist_to_food = 0
    if food_carrying < problem.MINIMUM_IMPROVEMENT:
        # distance to the closest food
        food_list = getFood(captureAgent, gameState).asList()

        min_pos = None
        min_dist = 99999999
        for food in food_list:
            dist = distancer.getDistance(myPos, food)
            if dist < min_dist:
                min_pos = food
                min_dist = dist

        dist_to_food = min_dist
        return_home_from = min_pos
        return dist_to_food

    # Returning Home
    # WARNING: this assumes the maps are always semetrical, territory is divided in half, red on right, blue on left
    walls = list(gameState.getWalls())
    y_len = len(walls[0])
    x_len = len(walls)
    mid_point_index = int(x_len/2)
    if captureAgent.red:
        mid_point_index -= 1

    # find all the entries and find distance to closest
    entry_coords = []
    for i, row in enumerate(walls[mid_point_index]):
        if row is False:  # there is not a wall
            entry_coords.append((int(mid_point_index), int(i)))

    minDistance = min([distancer.getDistance(return_home_from, entry)
                       for entry in entry_coords])
    return dist_to_food + minDistance


# methods required for above heuristic
def getFood(agent, gameState):
    """
    Returns the food you're meant to eat. This is in the form of a matrix
    where m[x][y]=true if there is food you can eat (based on your team) in that square.
    """
    if agent.red:
        return gameState.getBlueFood()
    else:
        return gameState.getRedFood()


################# Defensive problems and heuristics  ####################


class defensiveAgent(agentBase):

    prevMissingFoodLocation = None
    enemyEntered = False
    boundaryGoalPosition = None

    def chooseAction(self, gameState: GameState):

        problem = defendTerritoryProblem(
            startingGameState=gameState, captureAgent=self)

        # actions = search.breadthFirstSearch(problem)
        # actions = aStarSearch(problem, heuristic=defensiveHeuristic)
        actions = aStarSearchDefensive(problem, heuristic=defensiveHeuristic)
        aStarSearchDefensive
        if len(actions) != 0:
            return actions[0]
        else:
            return random.choice(gameState.getLegalActions(self.index))
            # return 'Stop'
        # return actions[0]


class defendTerritoryProblem():
    def __init__(self, startingGameState: GameState, captureAgent: CaptureAgent):
        self.expanded = 0
        self.startingGameState = startingGameState
        self.captureAgent: CaptureAgent = captureAgent
        self.enemies = self.captureAgent.getOpponents(startingGameState)
        self.walls = startingGameState.getWalls()
        self.intialPosition = self.startingGameState.getAgentPosition(
            self.captureAgent.index)
        self.gridWidth = self.captureAgent.getFood(startingGameState).width
        self.gridHeight = self.captureAgent.getFood(startingGameState).height
        if self.captureAgent.red:
            self.boundary = int(self.gridWidth / 2) - 1
            self.myPreciousFood = self.startingGameState.getRedFood()
        else:
            self.boundary = int(self.gridWidth / 2)
            self.myPreciousFood = self.startingGameState.getBlueFood()

        (self.viableBoundaryPositions,
         self.possibleEnemyEntryPositions) = self.getViableBoundaryPositions()

        self.GOAL_POSITION = self.getGoalPosition()
        self.goalDistance = self.captureAgent.getMazeDistance(
            self.GOAL_POSITION, self.intialPosition)

    def getViableBoundaryPositions(self):
        myPos = self.startingGameState.getAgentPosition(
            self.captureAgent.index)
        b = self.boundary
        boundaryPositions = []
        enemyEntryPositions = []

        for h in range(0, self.gridHeight):
            if self.captureAgent.red:
                if not(self.walls[b][h]) and not(self.walls[b+1][h]):
                    if (b, h) != myPos:
                        boundaryPositions.append((b, h))
                    enemyEntryPositions.append((b+1, h))

            else:
                if not(self.walls[b][h]) and not(self.walls[b-1][h]):
                    if (b, h) != myPos:
                        boundaryPositions.append((b, h))
                    enemyEntryPositions.append((b-1, h))

        return (boundaryPositions, enemyEntryPositions)

    def getGoalPosition(self):
        isPacman = self.startingGameState.getAgentState(
            self.captureAgent.index).isPacman

        isScared = self.startingGameState.getAgentState(
            self.captureAgent.index).scaredTimer > 0

        if isScared:
            boundaryGoalPositions = self.closestPosition(
                self.intialPosition, self.viableBoundaryPositions)
            if self.captureAgent.boundaryGoalPosition == None:
                boundaryGoalPosition = boundaryGoalPositions.pop()
                self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
            else:
                if self.captureAgent.boundaryGoalPosition == self.intialPosition:
                    boundaryGoalPosition = boundaryGoalPositions.pop()
                    self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
                else:
                    boundaryGoalPosition = self.captureAgent.boundaryGoalPosition
            return boundaryGoalPosition

        missingFoodPosition = self.getMissingFoodPosition()

        if missingFoodPosition != None:
            self.captureAgent.prevMissingFoodLocation = missingFoodPosition
            return missingFoodPosition

        for enemy in self.enemies:
            if self.startingGameState.getAgentState(enemy).isPacman:
                self.captureAgent.enemyEntered = True
                if self.startingGameState.getAgentPosition(enemy) != None:
                    return self.startingGameState.getAgentPosition(enemy)
                else:
                    return self.getProbableEnemyEntryPointBasedOnFood()
                    # return self.getProbableEnemyEntryPoint()
            else:
                self.captureAgent.enemyEntered = False

        if self.captureAgent.prevMissingFoodLocation != None and self.captureAgent.enemyEntered:
            return self.captureAgent.prevMissingFoodLocation

        boundaryGoalPositions = self.closestPosition(
            self.intialPosition, self.viableBoundaryPositions)

        if self.captureAgent.boundaryGoalPosition == None:
            boundaryGoalPosition = boundaryGoalPositions.pop()
            self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
        else:
            if self.captureAgent.boundaryGoalPosition == self.intialPosition:
                boundaryGoalPosition = boundaryGoalPositions.pop()
                self.captureAgent.boundaryGoalPosition = boundaryGoalPosition
            else:
                boundaryGoalPosition = self.captureAgent.boundaryGoalPosition

        return boundaryGoalPosition

    def closestPosition(self, fromPos, positions):
        positionsSorted = util.PriorityQueue()
        for toPos in positions:
            positionsSorted.push(
                toPos, self.captureAgent.getMazeDistance(toPos, fromPos))
        return positionsSorted

    def getProbableEnemyEntryPoint(self):
        positionsSorted = util.PriorityQueue()
        positionsSorted = self.closestPosition(
            self.intialPosition, self.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.distancer.getDistanceOnGrid(self.intialPosition, possibleEntry) > 5:
                return possibleEntry
        return random.choice(self.possibleEnemyEntryPositions)

    def getProbableEnemyEntryPointBasedOnFood(self):
        positionsSorted = util.PriorityQueue()
        bestEnemyPosition = util.PriorityQueue()
        positionsSorted = self.closestPosition(
            self.intialPosition, self.possibleEnemyEntryPositions)

        while not(positionsSorted.isEmpty()):
            possibleEntry = positionsSorted.pop()
            if self.captureAgent.distancer.getDistanceOnGrid(self.intialPosition, possibleEntry) > 5:
                closestFoodPosition = self.closestPosition(
                    possibleEntry, self.myPreciousFood.asList()).pop()
                distancetoToClosestFoodFromPosition = self.captureAgent.getMazeDistance(
                    possibleEntry, closestFoodPosition)
                bestEnemyPosition.push(
                    possibleEntry, distancetoToClosestFoodFromPosition)

        bestEnemyEntryPosition = bestEnemyPosition.pop()

        if bestEnemyEntryPosition:
            return bestEnemyEntryPosition
        else:
            return random.choice(self.possibleEnemyEntryPositions)

    def getMissingFoodPosition(self):

        prevFood = self.captureAgent.getFoodYouAreDefending(self.captureAgent.getPreviousObservation()).asList() \
            if self.captureAgent.getPreviousObservation() is not None else list()
        currFood = self.captureAgent.getFoodYouAreDefending(
            self.startingGameState).asList()

        if prevFood:
            if len(prevFood) > len(currFood):
                foodEaten = list(set(prevFood) - set(currFood))
                if foodEaten:
                    return foodEaten[0]
        return None

    def getStartState(self):
        return (self.startingGameState, self.goalDistance)

    def isGoalState(self, state: (GameState, int)):

        gameState = state[0]

        (x, y) = myPos = gameState.getAgentPosition(self.captureAgent.index)

        if myPos == self.GOAL_POSITION:
            return True
        else:
            return False

    def getSuccessors(self, state: (GameState, int), node_info=None):
        self.expanded += 1

        gameState = state[0]

        actions: t.List[Directions] = gameState.getLegalActions(
            self.captureAgent.index)

        goalDistance = self.captureAgent.getMazeDistance(
            self.GOAL_POSITION, gameState.getAgentPosition(self.captureAgent.index))

        successors_all = [((gameState.generateSuccessor(
            self.captureAgent.index, action), goalDistance), action, 1) for action in actions]

        successors = []

        for successor in successors_all:
            (xs, ys) = successor[0][0].getAgentPosition(
                self.captureAgent.index)
            if self.captureAgent.red:
                if xs <= self.boundary:
                    successors.append(successor)
            else:
                if xs >= self.boundary:
                    successors.append(successor)

        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        util.raiseNotDefined()


def defensiveHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    gameState = state[0]
    currGoalDistance = state[1]

    succGoalDistance = problem.captureAgent.getMazeDistance(
        problem.GOAL_POSITION, gameState.getAgentPosition(problem.captureAgent.index))

    if succGoalDistance < currGoalDistance:
        return 0
    else:
        return float('inf')


#################### Utils #############################
# utils for controlling execution time
# Credit to: https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python


class TimeoutException(Exception):
    pass

# TODO: this only runs in Unix environment (the competition is unix). If developing on windows can update this.


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# How to use it
# try:
#     with time_limit(10):
#         long_function_call()
# except TimeoutException as e:
#     print("Timed out!")


################# Search Algorithems ###################


class SolutionNotFound(Exception):
    pass


class Node():
    def __init__(self, *, name):
        self.name = name

    def add_info(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


def nullHeuristic(state, problem=None):
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE FOR TASK 3 ***"
    node = Node(name="n0").add_info(state=problem.getStartState())
    h_n = heuristic(node.state, problem=problem)
    g_n = 0  # accumulated cost so far
    node.add_info(
        f_n=g_n + h_n,  # f unction to sort the priority queue by
        g_n=g_n,  # accumilated cost so far
        action_from_init=[],
    )

    op = util.PriorityQueue()
    op.push(node, priority=node.f_n)
    close = set()  # state based
    best_g = {}  # key = state, value = g_n

    # total_expanded = 0
    # reopen_count = 0

    while not op.isEmpty():
        count = 1
        node = op.pop()
        if (node.state not in close) or (node.g_n < best_g[node.state]):
            # very useful debugging info - will be left for future users
            # total_expanded += 1
            # print("------------")
            # print(node.state[0])
            # print(node.action_from_init)
            # print(f"f_n {node.f_n}")
            # print(f"g_n {node.g_n}")
            # print(f"Nodes expanded {total_expanded}")
            # if node.state in close:
            #     print(f"node reopened")
            #     reopen_count +=1
            #     print(f"total reopens {reopen_count}")
            #     print("previous g_n improved on {best_g[node.state]}")
            # print("------------")
            close.add(node.state)
            best_g[node.state] = node.g_n
            if problem.isGoalState(node.state):
                break
            else:
                for related_node in problem.getSuccessors(node.state, node_info={"action_from_init": [*node.action_from_init]}):
                    new_state, action, step_cost = related_node[0], related_node[1], related_node[2]
                    g_n = node.g_n + step_cost
                    h_n = heuristic(new_state, problem=problem)
                    if h_n < float('inf'):  # solution is possible
                        new_node = Node(name=f"n{count}").add_info(
                            state=new_state,
                            f_n=g_n + h_n,
                            g_n=g_n,
                            action_from_init=[
                                *node.action_from_init] + [action],
                        )
                        # checking if goal here improves performance
                        # also protects agains bad heuristics that would send a goal to the end of the heap
                        if problem.isGoalState(node.state):
                            node = new_node
                            break
                        count += 1
                        op.update(new_node, new_node.f_n)
    else:
        raise SolutionNotFound({"start_state": problem.getStartState})

    # debugging info will be left for future users
    # print("--------------- FINAL RESULTS -----------------")
    # print(node.action_from_init)
    # print(len(node.action_from_init))
    # print(f"node reopened {reopen_count}")
    return node.action_from_init
    gameState = state[0]
    currGoalDistance = state[1]

    succGoalDistance = problem.captureAgent.getMazeDistance(
        problem.GOAL_POSITION, gameState.getAgentPosition(problem.captureAgent.index))
    # print(gameState.getAgentPosition(problem.captureAgent.index))
    if succGoalDistance < currGoalDistance:
        return 0
    else:
        return 9999999

    return 0


def aStarSearchDefensive(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE FOR TASK 3 ***"

    explore = util.PriorityQueue()

    initial_state = problem.getStartState()
    h = heuristic(initial_state, problem)
    g = 0

    explore.update((initial_state, [], g), g + h)

    visited_states = []
    best_g = 0

    while not explore.isEmpty():
        state, action, g = explore.pop()

        if state not in visited_states or g < best_g:
            visited_states.append(state)

            if g < best_g:
                best_g = g

            if problem.isGoalState(state):
                return action

            for child_state, child_action, child_cost in problem.getSuccessors(state):
                child_heuristic = heuristic(child_state, problem)

                if child_heuristic < float("inf"):
                    explore.update(
                        (child_state, action + [child_action], g + child_cost),
                        g + child_cost + child_heuristic,
                    )

    return []
    util.raiseNotDefined()
