# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #I used newScaredTimes for better evaluation function
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        ghostPosition = newGhostStates[0].getPosition()
        distanceToGhost = manhattanDistance(newPos,ghostPosition)


        if distanceToGhost > 0:
            score -= 1.0 / distanceToGhost

        foodList = newFood.asList()
        for x in foodList:
            distancesToFood = [manhattanDistance(newPos,x)]
            if len(distancesToFood) == True:
                score += 1.0 / min(distancesToFood)

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"


        return self.max_value(gameState,1)




    def max_value(self,gameState,depth):
        """
        Beginning of Terminal State
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # End of Terminal State
        v = -99999999999999999999999999999999999
        agentIndex = 0  # agentIndex of pacman is 0
        action = Directions.STOP
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex,a)
            value = max(v, self.min_value(successor,depth,1))
            if value > v:
                v = value
                action = a
        if depth > 1:
            return v
        return action

    def min_value(self,gameState,depth,agentIndex):
        """
        Beginning of Terminal State
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # End of Terminal State
        v = 99999999999999999999999999999999999999999
        ghostNumber = gameState.getNumAgents() - 1
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            successor = [gameState.generateSuccessor(agentIndex,a)]
            if agentIndex != ghostNumber:
                for s in successor:
                    v = min(v, self.min_value(s, depth, agentIndex + 1))
            else:
                #Opponent behaviors
                if depth < self.depth:
                    for s in successor:
                        v = min(v, self.max_value(s, depth + 1))
                else:
                 #The deepest leaf nodes values
                    for s in successor:
                        v = min(v, self.evaluationFunction(s))
        return v




       # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -99999999999999999999999999999999
        beta = 9999999999999999999999999999999999
        return self.max_value(gameState, 1, alpha, beta)

    def max_value(self, gameState, depth, alpha,beta):
        """
        Beginning of Terminal State
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # End of Terminal State
        v = -99999999999999999999999999999999999
        action = Directions.STOP
        actions = gameState.getLegalActions(0)
        for a in actions:
            successor = gameState.generateSuccessor(0, a)
            value = self.min_value(successor, depth, 1,alpha,beta)
            if value > v:
                v = value
                action = a
            if v > beta:
                return v
            alpha = max(alpha,v)
        if depth > 1:
            return v
        return action

    def min_value(self, gameState, depth, agentIndex,alpha,beta):
        """
        Beginning of Terminal State
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # End of Terminal State
        v = 99999999999999999999999999999999999999999
        actions = gameState.getLegalActions(agentIndex)
        GhostNumber = gameState.getNumAgents() - 1
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex,action)
            if agentIndex != GhostNumber:
                value = self.min_value(successor, depth, agentIndex + 1, alpha, beta)
            else:
                if depth < self.depth:
                    value = self.max_value(successor, depth + 1, alpha, beta)
                else:
                    value = self.evaluationFunction(successor)
            if value < v:
                v = value
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):


    def getAction(self, gameState):

        return self.max_value(gameState,1)

    def max_value(self, gameState, depth):
        """
        Beginning of Terminal State
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # End of Terminal State
        v = -99999999999999999999999999999999999
        action = Directions.STOP
        actions = gameState.getLegalActions(0)
        for a in actions:
            successor = gameState.generateSuccessor(0, a)
            value = self.expectedMin(successor, depth, 1)
            if value > v:
                v = value
                action = a
        if depth > 1:
            return v
        return action

    def expectedMin(self, gameState, depth, agentIndex):
        """
        Beginning of Terminal State
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # End of Terminal State
        v = 0
        actions = gameState.getLegalActions(agentIndex)
        GhostNumber = gameState.getNumAgents() - 1
        for a in actions:
            successor = [gameState.generateSuccessor(agentIndex,a)]
            probobalityOfSuccessor = 1.0 / len(actions)
            if agentIndex != GhostNumber:
                for s in successor:
                    v += probobalityOfSuccessor * self.expectedMin(s, depth, agentIndex + 1)
            else:
                if depth < self.depth:
                    for s in successor:
                        v += probobalityOfSuccessor * self.max_value(s, depth + 1)
                else:
                    for s in successor:
                        v += probobalityOfSuccessor * self.evaluationFunction(s)
        return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I used the same method with the first question, then I used newScaredTimes to handle when pacman eats capsules.
      I did not consider the eating capsule behavior and the aftermath in first one. Thus, in this function I created
      an if statement that checks whether the scared timer is bigger than zero or not. If its positive, I let the pacman
      to chase the ghost fast. Therefore, the pacman would have some kind of defense mechanicsm against the ghost and
      become a better evalutation than the first one.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()
    ghostPosition = newGhostStates[0].getPosition()
    distanceToGhost = manhattanDistance(newPos, ghostPosition)


    if distanceToGhost > 0:
        if newScaredTimes > 0:
        #we need to chase the scared ghost fast
            score += 1.0/distanceToGhost
        else:
        #regular ghosts we have to escape from them
            score -= 0.1 / distanceToGhost

    foodList = newFood.asList()
    for x in foodList:
        distancesToFood = [manhattanDistance(newPos, x)]
        if len(distancesToFood) == True:
            score += 0.1 / min(distancesToFood)

    return score



# Abbreviation
better = betterEvaluationFunction

