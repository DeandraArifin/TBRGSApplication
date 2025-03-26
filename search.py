import sys

from collections import deque
from matplotlib.font_manager import weight_dict
from nbformat import convert
from loadproblem import nodes, edges
from utils import *

#might need this as a global variable to build something like the romania map
route = {}

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        
        #altered to return the state itself as well
        if isinstance(self.goal, list):
            return is_in(state, self.goal), state
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)
  
  
class Graph:


    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)
  
#create the class for the graph problem, initial and goal states will later be initialized when we define the running function to run in main
class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m


#reformats the edges and weights so you can see the children nodes and their path costs per node
def convert_to_adjacency_list(edges):
      adjacency_list = {}
      for(source_node, dest_node), weight in edges.items():
            if source_node not in adjacency_list:
                  adjacency_list[source_node] = {}
            if dest_node not in adjacency_list:
                  adjacency_list[dest_node] = {}
            
            adjacency_list[source_node][dest_node] = weight
      return adjacency_list

# adjacency_list = convert_to_adjacency_list(edges)
# ourGraph = Graph(adjacency_list)
# ourGraph.locations(nodes)
      
def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        #   done to receive both values, first a boolean of whether or not a state is a goal state, the other the state itself
        goal, goal_state = problem.goal_test(node.state)
        #   only checks the returned boolean value
        if goal:
            return node, explored, goal_state
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None

def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    #   done to receive both values, first a boolean of whether or not a state is a goal state, the other the state itself
    goal, goal_state = problem.goal_test(node.state)
    explored = set()
    if goal:
          #returns the node, explored set (empty if the initial = goal state), and state returned if it's a goal state
        return node, explored, goal_state
  
  # debugging checks
#     print(f"node = {node}")
#     print(f"explored = {explored}")
#     print(f"goal state = {goal_state}")
#     print(f"goal = {goal}")
#     breakpoint()

    frontier = deque([node])
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                goal_child, goal_state_child = problem.goal_test(child.state)
                if goal_child:
                    return child, explored, goal_state_child
                frontier.append(child)
    return None

#pass in origin and destination instead of hardcoding it
    
def runOurGraph(ourGraph, origin, destination, search_algo):
    # Let's start with our path search problem
    prob = GraphProblem(origin, destination, ourGraph)
    result, explored, goal_state = search_algo(prob)
    
    # debugging checks
#     print(f"result = {result}")
#     print(f"explored = {explored}")
#     print(f"goal state = {goal_state}")
#     breakpoint()

    path =[]
    pNode = result
    while pNode.parent:
        path.insert(0, pNode.action)
        pNode = pNode.parent

    print(f"Our Path Finding Problem: The path from init to goal according to {search_algo.__name__} is: ", path)
    return path, explored, goal_state
    
def main():
        
        filename = sys.argv[1]
        method = sys.argv[2]

        # So the use of this is as specified in the assignment doc, (python search.py <filename> <method>)
        # search.py PathFinder-test.txt DFS (will currently just return nothing, but useful to see the test results for loading in)
        # Could maybe add some error handling to be more user friendly

        import loadproblem
        origin, destination = loadproblem.loadproblem(filename)


        # The following is the test lines from loadproblem.py, just to make sure it all worked fine with the changes
      #   print("Nodes:", loadproblem.nodes)
      #   print("Edges:", loadproblem.edges)
      #   print("Origin:", origin)
      #   print("Destination:", destination)
        
        #calls the reformatting of the edges
        adjacency_list = convert_to_adjacency_list(loadproblem.edges)
      #   print(adjacency_list)
        
        ourGraph = Graph(adjacency_list)
        ourGraph.locations = loadproblem.nodes
      #   print(ourGraph.locations)
       
        
        if method == 'DFS':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, depth_first_graph_search)
        elif method == 'BFS':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, breadth_first_graph_search)
        else:
              print(f"Method {method} not implemented.")
              result = None
              sys.exit(1)

        if result != None:
            #   goal, nodes_expanded, path = result
            #   logic is that all nodes created will be the initial, goal + any other nodes in the explored set
            #   initializes nodes_expanded with the explored set
              nodes_expanded = explored
            #   result holds the nodes in the resulting path of the search, so we're adding them to the nodes_expanded list
              for node in result:
                    nodes_expanded.add(node)   
                    
            
              print(f"{filename} {method}")
              
              #the len() function returns the number of nodes within the nodes_expanded list
              print(f"goal = {goal_state}, number_of_nodes = {len(nodes_expanded)}")
              print(" -> ".join(map(str,result)))

              # Should also be correct format as specified in doc

        else:
              print("No path found.")


if __name__ == "__main__":
      main()
