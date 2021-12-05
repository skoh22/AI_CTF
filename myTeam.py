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

    def withinFive(self, gameState):
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
            #if we know where the bad guys are, run expectimax
            actions = gameState.getLegalActions(self.index)  # pacman actions
            for i in range(len(actions)):
                action = actions[i]
                positions = []
                for j in range(gameState.getNumAgents()):
                    agentPos = (int(self.getPosDistCentroid(j)[0]), int(self.getPosDistCentroid(j)[1]))
                    positions.append(agentPos)
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

            #self.displayDistributionsOverPositions([globalBeliefs[agent] for agent in range(gameState.getNumAgents())])
            #self.debugDraw(mirrorPos, (1,0,1), True)
            return selected

    def Expectimax(self, gameState, currentPositions, currentDepth, currentAgent):
        def updatePositions(oldPositions, agent, action):
            newPositions = oldPositions
            newAgentPosition = Actions.getSuccessor(oldPositions[agent], action)
            newAgentPosition = (int(newAgentPosition[0]), int(newAgentPosition[1]))
            newPositions[agent] = newAgentPosition
            return newPositions
        def getActions(position):
            x, y = position
            actions = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (int(x+i), int(y+j)) in self.legalPositions and not (i != 0 and j != 0):
                        actions.append(Actions.vectorToDirection((i, j)))
            return actions
        if currentDepth >= 2 or gameState.isOver():
            nextActions = getActions(currentPositions[currentAgent])
            try:
                return max([self.evaluationFunction(gameState, currentPositions, action) for action in nextActions])
            except:
                return 0
        if currentAgent is self.index:  # pacman's turn
            nextActions = getActions(currentPositions[currentAgent])
            values = []
            # setting nextAgent
            nextAgent = markTarget[self.index]
            nextDepth = currentDepth
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                nextPositions = updatePositions(currentPositions, currentAgent, nextAction)
                values.append(self.Expectimax(gameState, nextPositions, nextDepth, nextAgent))
                if i is 0:
                    valMax = values[0]
                elif values[i] > valMax:
                    valMax = values[i]
            return valMax
        else:
            nextActions = getActions(currentPositions[currentAgent])
            '''actionProbs = util.Counter()
            for action in nextActions:  # guess relative probabilities of enemies' moves on greedy decisions
                newPositions = updatePositions(currentPositions, currentAgent, action)
                newPos = newPositions[currentAgent]

                print 'AGENT:', currentAgent
                print 'CURRENT:', newPositions[currentAgent] in self.legalPositions
                print 'ME:', self.index
                print 'SELF:', newPositions[self.index] in self.legalPositions

                if gameState.isOnRedTeam(currentAgent):
                    food = gameState.getRedFood().asList()
                else:
                    food = gameState.getBlueFood().asList()
                try:
                    oldDistToFood = min([self.distancer.getDistance(newPositions[currentAgent], f) for f in food])
                    newDistToFood = min([self.distancer.getDistance(newPos, f) for f in food])
                    actionProbs[action] += 5 if newDistToFood < oldDistToFood else 0
                except:  # no food left
                    pass

                self.debugDraw([newPositions[self.index]], (1,1,1))
                print newPositions[self.index]
                time.sleep(0.7)
                assert newPositions[currentAgent] in self.legalPositions
                assert newPositions[self.index] in self.legalPositions
                oldDistToMe = self.distancer.getDistance(newPositions[currentAgent], newPositions[self.index])
                newDistToMe = self.distancer.getDistance(newPos, newPositions[self.index])
                if self.isGhost(gameState, self.index, newPositions[self.index]) and gameState.getAgentState(self.index).scaredTimer == 0:
                    if newDistToMe > oldDistToMe:
                        actionProbs[action] += 10
                    elif newDistToMe < oldDistToMe:
                        actionProbs[action] -= 10
                else:
                    if newDistToMe > oldDistToMe:
                        actionProbs[action] -= 10
                    elif newDistToMe < oldDistToMe:
                        actionProbs[action] += 10
            if actionProbs.totalCount() == 0:  # prevent issue if all probs are 0
                for action in nextActions:
                    actionProbs[action] = 1
            actionProbs.normalize()'''

            # updating and tracking currentAgent
            if currentAgent == markTarget[self.index]:
                nextAgent = (markTarget[self.index]+2)%4
                nextDepth = currentDepth
            else:
                nextAgent = self.index
                nextDepth = currentDepth + 1
            values = []
            for i in range(len(nextActions)):
                nextAction = nextActions[i]
                nextPositions = updatePositions(currentPositions, currentAgent, nextAction)
                values.append(self.Expectimax(gameState, nextPositions, nextDepth, nextAgent))
            avg = 0
            for v in values:
                #avg += (float(v) *  actionProbs[nextActions[values.index(v)]])/ float(len(values))
                avg += float(v) / float(len(values))
            return avg

    def evaluationFunction(self, gameState, positions, action):
        # reward for eating ghosts
        # penalty for dying
        nextPositions = positions  # problem with generateSuccessor
        nextPositions[self.index] = Actions.getSuccessor(nextPositions[self.index], action)
        nextPositions[self.index] = (int(nextPositions[self.index][0]), int(nextPositions[self.index][1]))
        score = 0
        if self.isGhost(gameState, self.index, nextPositions[self.index]) and gameState.getAgentState(self.index).scaredTimer == 0:
            score -= 1000 * (min(self.distancer.getDistance(nextPositions[self.index], nextPositions[markTarget[self.index]]), self.distancer.getDistance(nextPositions[self.index], nextPositions[(markTarget[self.index]+2)%4])))
            score += 100
        else:
            score += 1000 * (min(self.distancer.getDistance(nextPositions[self.index], nextPositions[markTarget[self.index]]), self.distancer.getDistance(
                nextPositions[self.index], nextPositions[(markTarget[self.index] + 2) % 4])))
            score -= 100

        return score