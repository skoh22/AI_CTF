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
import os, pickle, math


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
    return [FeatureLearningAgent(firstIndex), FeatureLearningAgent(secondIndex)]


##########
# Agents #
##########
globalBeliefs = {}

class FeatureLearningAgent(CaptureAgent):
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
        #self.weights = util.Counter()
        self.epsilon = 0.8
        self.gamma = 0.75
        self.alpha = 0.2

        self.legalPositions = []
        for x in range(self.getFood(gameState).width):
            for y in range(self.getFood(gameState).height):
                if not gameState.hasWall(x, y):
                    self.legalPositions.append((x, y))
        for opponent in self.getOpponents(gameState):
            self.initializeAgentBeliefs(gameState, opponent)
        for teammate in self.getTeam(gameState):
            globalBeliefs[teammate] = util.Counter()
            globalBeliefs[teammate][gameState.getInitialAgentPosition(teammate)] = 1.0
        self.myTeammate = [i for i in self.getTeam(gameState) if i != self.index][0]

        self.weightsFile = 'weights{}.txt'.format(self.index)
        if os.path.exists(self.weightsFile):
            with open(self.weightsFile, 'r') as file:
                self.weights = pickle.load(file)
        else:
            with open(self.weightsFile, 'w') as file:
                self.weights = util.Counter()

        print self.weights

        try:
            print gameState.getAgentState((self.index+1)%4).numCarrying
            print gameState.getAgentState((self.index+1)%4).configuration
        except:
            pass


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


    def updateWeights(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # determine q values and utility for nextState
        nextQValues = [self.getQValue(nextState, a) for a in nextState.getLegalActions(self.index)]
        if len(nextQValues) == 0:
            nextValue = 0
        else:
            nextValue = max(nextQValues)

        # update
        change = reward + (self.gamma * nextValue) - self.getQValue(state, action)
        features = self.getFeatures(state, action)
        for f in features:
            step = self.alpha * change * features[f]
            self.weights[f] += step

        with open(self.weightsFile, 'w') as file:
            pickle.dump(self.weights, file)


    def getFeatures(self, gameState, action):
        # method calls
        currentPos = gameState.getAgentPosition(self.index)
        foodGrid =self.getFood(gameState)
        foodList = foodGrid.asList()
        walls = gameState.getWalls()

        visibleOpponents = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState)]
        dX, dY = Actions.directionToVector(action)
        nextX, nextY = int(currentPos[0] + math.ceil(dX)), int(currentPos[1] + math.ceil(dY))
        nextState = gameState.generateSuccessor(self.index, action)
        newPos = (nextX, nextY)

        features = util.Counter()
        # feature assignments
        features['eatsFood'] = 1.0 if foodGrid[nextX][nextY] else 0.0
        features['eatsCapsule'] = 1.0 if newPos in self.getCapsules(gameState) else 0.0
        features['getsEaten'] = 1.0 if newPos in visibleOpponents and not self.isGhost(gameState, self.index, newPos) else 0.0
        features['eatsOpponent'] = 1.0 if newPos in visibleOpponents and self.isGhost(gameState, self.index, newPos) else 0.0
        features['numFoodLeft'] = len(foodList) - features['eatsFood']
        features['newDistToFood'] = min([self.distancer.getDistance(newPos, food) for food in foodList] if len(foodList) > 0 else [0])
        features['numFoodDefending'] = len(self.getFoodYouAreDefending(gameState).asList())
        features['newMazeDistToTeammate'] = self.distancer.getDistance(newPos, globalBeliefs[self.myTeammate].argMax())
        features['newManhattanDistToTeammate'] = util.manhattanDistance(newPos, globalBeliefs[self.myTeammate].argMax())
        features['score'] = self.getScore(nextState)
        features['scoreChange'] = self.getScore(nextState) - self.getScore(gameState)
        features['numActions'] = len(nextState.getLegalActions(self.index))
        if nextState.isOver():
            if self.getScore(nextState) > 0:
                features['wins'] = 1.0
            elif self.getScore(nextState) < 0:
                features['wins'] = -1.0

        for enemy in range(2):
            enemyIndex = self.getOpponents(gameState)[enemy]
            # assume that opponent is in one of the spaces we are most confident of
            bestGuess = 0
            targetPosGuesses = []
            for p in globalBeliefs[enemyIndex]:
                if globalBeliefs[enemyIndex][p] == bestGuess:
                    targetPosGuesses.append(p)
                elif globalBeliefs[enemyIndex][p] > bestGuess:
                    bestGuess = globalBeliefs[enemyIndex][p]
                    targetPosGuesses = [p]
            enemyPos = random.choice(targetPosGuesses)

            features['mazeDistToEnemy'+str(enemy)] = self.distancer.getDistance(newPos, enemyPos)
            features['deltaToEnemy'+str(enemy)] = features['mazeDistToEnemy'+str(enemy)] - self.distancer.getDistance(currentPos, enemyPos)
            features['enemyIsGhost'+str(enemy)] = self.isGhost(gameState, enemyIndex, enemyPos)
            features['enemyPosConfidence'+str(enemy)] = bestGuess

        # make the distance a number less than one otherwise the update will diverge wildly
        features.divideAll(10000)

        return features


    def getQValue(self, state, action):
        """
          Return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        return self.weights * self.getFeatures(state, action)


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
        """
        actions = state.getLegalActions(self.index)
        if len(actions) == 0:
            return None
        else:
            maxQ = -99999
            maxActionList = []
            for action in actions:
                thisQ = self.getQValue(state, action)
                if thisQ > maxQ:
                    maxQ = thisQ
                    maxActionList = [action]
                elif thisQ == maxQ:
                    maxActionList.append(action)
            return random.choice(maxActionList)


    def chooseAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
        """
        self.updateBeliefs(self.getCurrentObservation())
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(gameState)

        reward = self.getScore(gameState.generateSuccessor(self.index, action))
        try:
            for agent in range(gameState.getNumAgents()):
                for move in gameState.getLegalActions(agent):
                    if gameState.generateSuccessor(agent, move).isOver():
                        if agent in self.getTeam(gameState):
                            reward *= 1000
                        else:
                            reward *=-1000
        except:
            pass
        self.updateWeights(gameState, action, gameState.generateSuccessor(self.index, action), reward)
        #time.sleep(0.3)
        #print self.weights

        currentPos = gameState.getAgentPosition(self.index)
        dX, dY = Actions.directionToVector(action)
        newPos = (currentPos[0] + math.ceil(dX), currentPos[1] + math.ceil(dY))
        myNewCounter = util.Counter()
        myNewCounter[newPos] = 1.0
        globalBeliefs[self.index] = myNewCounter

        self.displayDistributionsOverPositions([globalBeliefs[agent] for agent in range(gameState.getNumAgents())])

        return action


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


