# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.qValues = util.Counter()


        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        def terminalChecker(state,maxValue):
            if self.mdp.isTerminal(state):
                self.values[state] = 0
            else:
                self.values[state] = maxValue

        allIterations = [range (0,iterations)]

        for allIterations[0] in allIterations[0]:
            dummyValues = self.values.copy()
            states = [mdp.getStates()]
            for state in states[0]:
                maxValue = None
                possAct = [mdp.getPossibleActions(state)]
                for action in possAct[0]:
                    nextValue = 0
                    transitionStatesAndProbs = [mdp.getTransitionStatesAndProbs(state,action)]
                    for nextState, probability in transitionStatesAndProbs[0]:
                        value = [dummyValues[nextState]]
                        reward = [mdp.getReward(state, action, nextState)]
                        """Valuet = Sigma (R(s,a,s') + Discount*Valuet) * (T(s,a,s')  """
                        nextValue += (reward[0] + discount * value[0]) * probability
                    maxValue = max(nextValue, maxValue)
                terminalChecker(state,maxValue)




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        discount = [self.discount]
        transitionStatesAndProbs = [self.mdp.getTransitionStatesAndProbs(state, action)]
        QValueFromValues = 0

        for nextState, probability in transitionStatesAndProbs[0]:
            reward = [self.mdp.getReward(state,action,nextState)]
            value = [self.values[nextState]]
            """Valuet = Sigma (R(s,a,s') + Discount*Valuet-1) * (T(s,a,s')  """
            QValueFromValues = QValueFromValues + (reward[0] + (discount[0] * value[0])) * probability

        return QValueFromValues

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        "*** YOUR CODE HERE ***"

        possibleActions = self.mdp.getPossibleActions(state)
        if len(possibleActions) != 0:
            maxPolicy = max([(self.getQValue(state,a),a) for a in possibleActions])
            return maxPolicy[1]
        else:
            return None



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
