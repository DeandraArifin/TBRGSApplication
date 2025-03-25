import sys

from matplotlib.font_manager import weight_dict
from loadproblem import nodes, edges

def dfs(origin, destination):
    # DFS logic to be implemented here, currently just returns nothing
    return None, 0, []

    # Rest of search logic to be implemented also.

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



def main():
        
        filename = sys.argv[1]
        method = sys.argv[2]

        # So the use of this is as specified in the assignment doc, (python search.py <filename> <method>)
        # search.py PathFinder-test.txt DFS (will currently just return nothing, but useful to see the test results for loading in)
        # Could maybe add some error handling to be more user friendly

        import loadproblem
        origin, destination = loadproblem.loadproblem(filename)


        # The following is the test lines from loadproblem.py, just to make sure it all worked fine with the changes
        print("Nodes:", loadproblem.nodes)
        print("Edges:", loadproblem.edges)
        print("Origin:", origin)
        print("Destination:", destination)
        
        #calls the reformatting of the edges
        adjacency_list = convert_to_adjacency_list(loadproblem.edges)
        print(adjacency_list)

        result = None # Neutral placeholder for result variable to prevent weird errors.
        
       
        
        if method == 'DFS':
              result = dfs(origin, destination)
        elif method == 'BFS':
              # etc, etc
              pass
        else:
              print(f"Method {method} not implemented.")
              sys.exit(1)

        if result:
              goal, nodes_expanded, path = result
              print(f"{filename} {method}")
              print(f"{goal} {nodes_expanded}")
              print(" ->".join(map(str,path)))

              # Should also be correct format as specified in doc

        else:
              print("No path found.")


if __name__ == "__main__":
      main()
