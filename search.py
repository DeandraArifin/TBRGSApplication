import sys
from collections import deque
from matplotlib.font_manager import weight_dict
from nbformat import convert
from loadproblem import nodes, edges
from utils import *
import testcasegen

class Problem:
    """The abstract class for a formal problem."""

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
            return state == self.goal, state

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
        # Now expands smaller node ID's first
        return sorted(
            [self.child_node(problem, action)
            for action in problem.actions(self.state)],
            key=lambda node: node.state
        )

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

def UndirectedGraph(graph_dict=None):
    # Build a Graph where every edge (including future ones) goes both ways.
    # We should never need to use this though.. the graphs are directed.
    return Graph(graph_dict=graph_dict, directed=False)

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
    
    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                
                #modified to iterate through a list of goals if there's more than one goal
                if isinstance(self.goal, list):
                    return int(min(distance(locs[node], locs[g]) for g in self.goal))
                
                return int(distance(locs[node], locs[self.goal]))

            #again checks if goal is a list
            if isinstance(self.goal, list):
                return int(min(distance(locs[node.state], locs[g]) for g in self.goal))
            
            return int(distance(locs[node.state], locs[self.goal]))
            
        else:
            return np.inf


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
    print(frontier)

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
        
    return None, explored, None

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
    return None, explored, None

def best_first_graph_search(problem, f, display=True):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        goal, goal_state = problem.goal_test(node.state)
        if goal:
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier, the paths explored are: ", (explored), "what's left in the frontier is: ", [item for priority, item in frontier.heap])
            return node, explored, goal_state
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None, explored, None

def greedy_best_first_graph_search(problem, display=False):
    return best_first_graph_search(problem, lambda n: problem.h(n), display)

def astar_search(problem, h=None, display=False):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

def cus1_search(problem, display=False): # Custom algorithm 1, it is UCS//Uninformed astar.
    return best_first_graph_search(problem, lambda n: n.path_cost, display)

def cus2_search(problem, display=False): # Pretty much making Astar but where each path cost is 1, so using n.depth instead of n.path_cost to count the steps taken not total path cost
    h = memoize(problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.depth + h(n), display)

    
def runOurGraph(ourGraph, origin, destination, search_algo):
    problem = GraphProblem(origin, destination, ourGraph)
    
    result, explored, goal_state = search_algo(problem)
    
    if result == None:
        return None, explored, None
    path =[]
    pNode = result
    
    #adds in the actions taken FROM initial state to goal state (excludes initial state in path)
    while pNode.parent:
        # print(f"Node: {pNode.state}, Parent: {pNode.parent.state if pNode.parent else None}")
        path.insert(0, pNode.action)
        pNode = pNode.parent

    # Uncomment the following line if you want for debugging, should not be printing in final submission
    # print(f"Our Path Finding Problem: The path from init to goal according to {search_algo.__name__} is: ", path)
    return path, explored, goal_state
    
def main():
        #Trying to implement test case generation into here, which will also help with visualisation.
        generate = False
        filename = sys.argv[1]
        if filename.lower() == 'generate':
            generate = True
            i = testcasegen.main()
            os.chdir("./GeneratedPaths")
            filename = f"GenPathFinder{i}.txt"
            # Uncomment the following line if you want for debugging, should not be printing in final submission
            # print(filename)
        else:
            # Uncomment the following line if you want for debugging, should not be printing in final submission
            # print(filename)
            generate = False
            pass
        method = sys.argv[2]
        # breakpoint()

        # So the use of this is as specified in the assignment doc, (python search.py <filename> <method>)
        # search.py PathFinder-test.txt DFS (will currently just return nothing, but useful to see the test results for loading in)
        # Could maybe add some error handling to be more user friendly

        import loadproblem
        origin, destination, edges, nodes = loadproblem.loadproblem(filename)
        # Uncomment the following line if you want for debugging, should not be printing in final submission
        # print(f"Origin: {origin}\nDestination:{destination}")
        if "GenPathFinder" not in filename:
            os.chdir("..")

        adjacency_list = convert_to_adjacency_list(loadproblem.edges)

        # Uncomment the following if you want the adjacency list printed - not needed to be printed unless debugging
        # print("adjacency list = ", adjacency_list)
        
        #reformats edges and nodes into a graph constructed like the romania map
        #initializes it with the adjacency list, containing nodes, child nodes and the weights between parent and child nodes
        ourGraph = Graph(adjacency_list)
        if generate == False:
            newg, pos = loadproblem.todraw(nodes, edges, origin, destination)
        #locations holds the coordinates of the nodes
        ourGraph.locations = loadproblem.nodes       
       
        
        if method == 'DFS':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, depth_first_graph_search)
        elif method == 'BFS':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, breadth_first_graph_search)
        elif method == 'Astar':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, astar_search)
        elif method == 'GBFS':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, greedy_best_first_graph_search)
        elif method == 'CUS1':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, cus1_search)
        elif method == 'CUS2':
              result, explored, goal_state = runOurGraph(ourGraph, origin, destination, cus2_search)
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
            #   print(f"{nodes_expanded}")
              print(origin, "->", " -> ".join(map(str,result)))
              if generate == False:
                # colour_map = ['#91dea8' if node in result else '#2a8d48' if node == origin
                #                else '#f86c62' for node in newg.nodes]
                colour_map = ['#91dea8' if node in result else '#2a8d48' if node == origin
                              else '#f86c62' if node in nodes_expanded else '#D295BF' for node in newg.nodes]
                loadproblem.draw(newg, pos, colour_map, origin, destination)

                


              # Should also be correct format as specified in doc

        else:
              print("No path found.")


if __name__ == "__main__":
      main()