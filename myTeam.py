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
import os, pickle


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
    return [eval(first)(firstIndex), FeatureLearningAgent(secondIndex)]


##########
# Agents #
##########

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


def isPacman(gameState, agent):
    return gameState.data


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
        '''if os.path.exists('weights.txt'):
            with open('weights.txt', 'r') as weightsFile:
                self.weights = pickle.loads(weightsFile.read())
        else:
            with open('weights.txt', 'a') as weightsFile:
                self.weights = util.Counter()'''
        self.weights = util.Counter()
        self.epsilon = 0.05
        self.gamma = 0.8
        self.alpha = 0.2,





    def update(self, state, action, nextState, reward):
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
        change = reward + (self.discount * nextValue) - self.getQValue(state, action)
        features = self.getFeatures(state, action)
        for f in features:
            step = self.alpha * change * features[f]
            self.weights[f] += step


    def getFeatures(self, currentState, action):
        currentPos = currentState.getAgentPosition(self.index)
        foodPositions = self.getFood(currentState).asList()

        walls = currentState.getWalls()

        # numFoodAttacking = len(self.getFood(currentState).asList())
        # distToOwnFood = min([self.distancer.getDistance(currentPos, food) for food in self.getFood(currentState).asList()])
        # numFoodDefending = len(self.getFoodYouAreDefending(currentState).asList())
        # distToTeammate = LOG OWN POS IN GLOBALBELIEFS FOR TEAMMATE (Counter w/ only one val == 1)
        # enemiesWithinView = sum([currentState.getAgentPosition(enemy) for enemy in self.getOpponents(currentState)])

        # compute the location of pacman after he takes the action
        x, y = currentPos
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        features = util.Counter()

        enemyPositions = [currentState.getAgentPosition(i) for i in self.getOpponents(currentState)]
        visibleEnemies = [enemy for enemy in enemyPositions if enemy is not None]
        ghosts = []
        pacs = []
        for enemy in visibleEnemies:
            if self.red:
                if enemy[0] < self.getFood(currentState).width / 2:
                    pacs.append(enemy)
                else:
                    ghosts.append(enemy)
            else:
                if enemy[0] > self.getFood(currentState).width / 2:
                    pacs.append(enemy.getPosition())
                else:
                    ghosts.append(enemy.getPosition())

        print 'GHOSTS', ghosts

        try:
            # count the number of ghosts 1-step away
            features["#-of-ghosts-1-step-away"] = sum(
                (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        except:
            pass

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        #dist = closestFood((next_x, next_y), food, walls)
        nearestFoodDist = min([self.distancer.getDistance(currentPos, food) for food in foodPositions])
        if nearestFoodDist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(nearestFoodDist) / (walls.width * walls.height)
        features.divideAll(10.0)

        #print foodPositions

        score = self.getScore(currentState)

        #features['food'] = nearestFood
        features['score'] = score
        features['enemy1Dist']
        features['enemy2Dist']
        features['teammateDist']
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
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(gameState)

        return action
