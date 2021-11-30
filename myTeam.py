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
from game import Directions, Actions
import game
import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
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
    #return [eval(first)(firstIndex), eval(second)(secondIndex)]
    return [trackingAgent(firstIndex), trackingAgent(secondIndex)]


##########
# Agents #
##########

globalBeliefs = {}

class trackingAgent(CaptureAgent):
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

        '''
        Your initialization code goes here, if you need any.
        '''
        self.legalPositions = []
        for x in range(self.getFood(gameState).width):
            for y in range(self.getFood(gameState).height):
                if not gameState.hasWall(x, y):
                    self.legalPositions.append((x, y))
        for opponent in self.getOpponents(gameState):
            self.initializeAgentBeliefs(gameState, opponent)
        for teammate in self.getTeam(gameState):
            globalBeliefs[teammate] = util.Counter()
        self.teammate = [i for i in self.getTeam(gameState) if i != self.index][0]

    def initializeAgentBeliefs(self, gameState, agent):
        agentBeliefs = util.Counter()
        agentBeliefs[gameState.getInitialAgentPosition(agent)] = 1.0
        globalBeliefs[agent] = agentBeliefs

    def initializeUniformly(self, gameState, agent):
        globalBeliefs[agent] = util.Counter()
        for p in self.legalPositions:
            globalBeliefs[agent][p] = 1.0
        globalBeliefs[agent].normalize()

    def updateBeliefs(self, currentState):
        currentDists = currentState.getAgentDistances()
        currentPos = currentState.getAgentPosition(self.index)
        for agent in self.getOpponents(currentState):
            obs = currentState.getAgentPosition(agent)
            if obs == None:  # if not directly observable
                if globalBeliefs[agent].totalCount() == 0:
                    self.initializeUniformly(currentState, agent)
                updatedBeliefs = util.Counter()
                for p in self.legalPositions:
                    trueDistance = util.manhattanDistance(p, currentPos)
                    updatedBeliefs[p] = currentState.getDistanceProb(trueDistance, currentDists[agent]) * \
                                        globalBeliefs[agent][p]
                updatedBeliefs.normalize()
            else:
                updatedBeliefs = util.Counter()
                updatedBeliefs[obs] = 1.0
            globalBeliefs[agent] = updatedBeliefs


    def isGhost(self, gameState, agent, position):
        halfLine = gameState.getWalls().width / 2
        if gameState.isOnRedTeam(agent):
            if position[0] <= halfLine:
                return True
        else:
            if position[0] > halfLine:
                return True
        return False


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        self.updateBeliefs(self.getCurrentObservation())

        actions = gameState.getLegalActions(self.index)
        currentPos = gameState.getAgentPosition(self.index)
        targetIndex = (self.index + 1) % 4  # each of my agents targets one of the other team's agents

        # target is at the "center of mass" of the probabilities
        bestGuess = 0
        targetPosGuesses = []
        xPosSum = 0
        yPosSum = 0
        for p in globalBeliefs[targetIndex]:
            xVal, yVal = p
            xPosSum += xVal * globalBeliefs[targetIndex][p]
            yPosSum += yVal * globalBeliefs[targetIndex][p]
        targetX = xPosSum / max(globalBeliefs[targetIndex].totalCount(),1)
        targetY = yPosSum / max(globalBeliefs[targetIndex].totalCount(),1)

        #calculating mirror position
        mirrorPosX = min(max(gameState.getWalls().width-1 - targetX,1),gameState.getWalls().width)
        mirrorPosY = min(max(gameState.getWalls().height-1 - targetY,1),gameState.getWalls().height)
        mirrorPos = int(mirrorPosX), int(mirrorPosY)
        if mirrorPos not in self.legalPositions:  # if not a legal position (e.g. wall), choose a legal neighbor
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (int(mirrorPosX+i), int(mirrorPosY+j)) in self.legalPositions:
                        neighbors.append((int(mirrorPosX+i), int(mirrorPosY+j)))
            mirrorPos = random.choice(neighbors)

        # calculate maze distance to mirroring opponent for each possible move
        distances = []
        for action in actions:
            dX, dY = Actions.directionToVector(action)
            nextX, nextY = currentPos[0] + math.ceil(dX), currentPos[1] + math.ceil(dY)
            if (nextX, nextY) in self.legalPositions:
                distances.append(self.distancer.getDistance((int(nextX), int(nextY)), mirrorPos))
            else:
                distances.append(999999)

        # choose move that minimizes distance to mirrored position
        selected = actions[distances.index(min(distances))]

        dX, dY = Actions.directionToVector(selected)
        newPos = (currentPos[0] + math.ceil(dX), currentPos[1] + math.ceil(dY))
        myNewCounter = util.Counter()
        myNewCounter[newPos] = 1.0
        globalBeliefs[self.index] = myNewCounter

        self.displayDistributionsOverPositions([globalBeliefs[agent] for agent in range(gameState.getNumAgents())])
        self.debugDraw(mirrorPos, (1,0,1), True)
        return selected





class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
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

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)
