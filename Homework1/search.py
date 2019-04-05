# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def partiallyGenericSearch(problem,utility):
    emptySet = []

    startState = problem.getStartState()
    utility.push((startState, emptySet, 0))
    traced = set()
    while not utility.isEmpty():
        position, path, visited = utility.pop()
        if problem.isGoalState(position) == True:
            return path
        if position not in traced:
            traced.add(position)
            successor = problem.getSuccessors(position)
            for childPosition, childPath, toward in successor:
                if childPosition not in traced:
                    utility.push((childPosition, path + [childPath], visited))
    return emptySet

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    utility = util.Stack()
    return partiallyGenericSearch(problem, utility)

    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    utility = util.Queue()
    return partiallyGenericSearch(problem, utility)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


""" 
    def AnotherParrtiallyGenericSearch(problem, fn , heuristic = nullHeuristic):
    utility = util.PriorityQueue()
    emptySet = []
    startState = problem.getStartState()
    Distinction = 0
    utility.push((startState, emptySet, 0), 0)
    traced = set()
    while not utility.isEmpty():
        position, path, visited = utility.pop()
        if problem.isGoalState(position) == True:
            return path
        if position not in traced:
            traced.add(position)
            successor = problem.getSuccessors(position)
            for side, up, toward in successor:
                forward = toward + visited
                backward = forward + heuristic(side, problem)
                if fn == astar:
                    Distinction = forward + backward
                if fn == ucs:
                    Distinction = forward
                if side not in traced:
                    utility.push((side, path + [up], forward), Distinction)
    return emptySet

    """
def uniformCostSearch(problem):
    """Search the node of least total cost first."""


    "*** YOUR CODE HERE ***"

    UniPriorityQueue = util.PriorityQueue()
    emptySet = []
    startState = problem.getStartState()
    UniPriorityQueue.push((startState, emptySet, 0),0)
    traced = set()
    while not UniPriorityQueue.isEmpty():
        position, path, visited = UniPriorityQueue.pop()
        if problem.isGoalState(position) == True:
            return path
        if position not in traced:
            traced.add(position)
            successor = problem.getSuccessors(position)
            for side, up, toward in successor:
                forward = toward + visited
                if side not in traced:
                    UniPriorityQueue.push((side, path + [up], forward), forward)
    return emptySet




def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    aStarPriorityQueue = util.PriorityQueue()
    emptySet = []
    startState = problem.getStartState()
    aStarPriorityQueue.push((startState, emptySet, 0), 0)
    traced = set()
    while not aStarPriorityQueue.isEmpty():
        position, path, visited = aStarPriorityQueue.pop()
        if problem.isGoalState(position) == True:
            return path
        if position not in traced:
            traced.add(position)
            successor = problem.getSuccessors(position)
            for side, up, toward in successor:
                forward = toward + visited
                backward = forward + heuristic(side, problem)
                if side not in traced:
                    aStarPriorityQueue.push((side, path + [up], forward), backward)
    return emptySet



    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
