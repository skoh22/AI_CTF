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
markTarget = {}

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
        markTarget[self.index] = (self.index + 1) % 4  # each of my agents targets one of the other team's agents

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

    def handoff(self, gameState):
        # if agents are nearer to teammate's marks than their own, switch marks
        team = self.getTeam(gameState)
        matchDistance = 0
        switchDistance = 0
        for agent in team:
            matchDistance += self.distancer.getDistance(self.getPosDistCentroid(agent), self.getMirrorPos(gameState, self.getPosDistCentroid(markTarget[agent])))
            switchDistance += self.distancer.getDistance(self.getPosDistCentroid(agent), self.getMirrorPos(gameState, self.getPosDistCentroid((markTarget[agent]+2)%4)))
        if switchDistance < matchDistance:
            for agent in team:
                markTarget[agent] = (markTarget[agent]+2)%4

    def getPosDistCentroid(self, agentIndex):
        # position at the "center of mass" of the probability distribution (or a legal neighbor)
        xPosSum = 0
        yPosSum = 0
        for p in globalBeliefs[agentIndex]:
            xVal, yVal = p
            xPosSum += xVal * globalBeliefs[agentIndex][p]
            yPosSum += yVal * globalBeliefs[agentIndex][p]
        x = int(xPosSum / max(globalBeliefs[agentIndex].totalCount(),1))
        y = int(yPosSum / max(globalBeliefs[agentIndex].totalCount(),1))
        pos = (x, y)

        if pos not in self.legalPositions:  # if not a legal position (e.g. wall), choose a legal neighbor
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (int(x + i), int(y + j)) in self.legalPositions:
                        neighbors.append((int(x + i), int(y + j)))
            pos = random.choice(neighbors)
        return pos

    def getMirrorPos(self, gameState, position):
        # get rotationally symmetric reflection of a position
        x, y = position
        mirrorPosX = min(max(gameState.getWalls().width - 1 - x, 1), gameState.getWalls().width)
        mirrorPosY = min(max(gameState.getWalls().height - 1 - y, 1), gameState.getWalls().height)
        mirrorPos = (int(mirrorPosX), int(mirrorPosY))
        return mirrorPos

    def withinFive(self,gameState):
        if gameState.getAgentPosition((self.index + 1) % 4) is not None or gameState.getAgentPosition((self.index + 3) % 4) is not None:
            return True
        else:
            return False

    def chooseAction(self, gameState):
        self.updateBeliefs(self.getCurrentObservation())
        self.handoff(gameState)

        actions = gameState.getLegalActions(self.index)
        currentPos = gameState.getAgentPosition(self.index)

        if self.withinFive(gameState):
            #print "withinFive "
            #if we know where the bad guys are, run expectimax
            actions = gameState.getLegalActions(self.index)  # pacman actions
            for i in range(len(actions)):
                action = actions[i]
                positions = []
                for j in range(gameState.getNumAgents()):
                    positions.append(self.getPosDistCentroid(j))
                val = self.Expectimax(gameState.generateSuccessor(self.index, action), positions, 0, markTarget[self.index])
                if i is 0:
                    bestAction = actions[0]
                    bestVal = val
                if val > bestVal:
                    bestVal = val
                    bestAction = action

            #tell teammate position
            dX, dY = Actions.directionToVector(bestAction)
            newPos = (currentPos[0] + math.ceil(dX), currentPos[1] + math.ceil(dY))
            myNewCounter = util.Counter()
            myNewCounter[newPos] = 1.0
            globalBeliefs[self.index] = myNewCounter

            return bestAction

        else: #mirroring
            # get mirrored position of marked opponent
            mirrorPos = self.getMirrorPos(gameState, self.getPosDistCentroid(markTarget[self.index]))

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

    def Expectimax(self, gameState, currentPositions, currentDepth, currentAgent):
        #print "Expectimax", "currentDepth: ", currentDepth, "currentAgent: ", currentAgent
        #print "self.index: ", self.index
        if currentDepth >= 3 or gameState.isOver():
            nextActions = []
            posX, posY = self.getPosDistCentroid(currentAgent)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (int(posX + i), int(posY + j)) in self.legalPositions and not (i != 0 and j != 0):
                        nextActions.append(Actions.vectorToDirection((i, j)))
            return self.evaluationFunction(gameState, currentPositions,nextActions)
        if currentAgent is self.index:  # pacman's turn
            nextActions = gameState.getLegalActions(currentAgent)
            values = []
            #setting nextAgent
            nextAgent = (markTarget[self.index] + 2)%4
            nextDepth = currentDepth
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                nextPositions = currentPositions #problem with generateSuccessor
                nextPositions[currentAgent] = Actions.getSuccessor(nextPositions[currentAgent],nextAction)
                values.append(self.Expectimax(gameState, nextPositions ,nextDepth, nextAgent))
                if i is 0:
                    valMax = values[0]
                elif values[i] > valMax:
                    valMax = values[i]
            return valMax
        else:
            #print "currentAgent: ", currentAgent
            nextActions = []
            posX, posY = self.getPosDistCentroid(currentAgent)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (int(posX + i), int(posY + j)) in self.legalPositions and not (i != 0 and j != 0):
                        nextActions.append(Actions.vectorToDirection((i,j)))
            values = []
            arg_expected = 0
            # updating and tracking currentAgent
            nextAgent = self.index
            nextDepth = currentDepth + 1
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                nextPositions = currentPositions  # problem with generateSuccessor
                nextPositions[currentAgent] = Actions.getSuccessor(nextPositions[currentAgent], nextAction)
                values.append(self.Expectimax(gameState, nextPositions, nextDepth, nextAgent))
            avg = 0
            for v in values:
                avg += float(v) / float(len(values))
            return avg

    def evaluationFunction(self, gameState, positions, action):
        #plan:
        # reward for eating ghosts
        # penalty for dying
        score = 0
        if self.isGhost(gameState, self.index, positions[self.index]) and gameState.getAgentState(self.index).scaredTimer == 0:
            if positions[self.index] == positions[markTarget[self.index]] or positions[self.index] == positions[(markTarget[self.index]+2)%4]:
                score += 1000
        else:
            if positions[self.index] == positions[markTarget[self.index]] or positions[self.index] == positions[(markTarget[self.index]+2)%4]:
                score -= 1000

        #print "score: ",score

        return score


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
