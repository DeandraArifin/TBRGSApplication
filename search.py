import sys
from loadproblem import nodes, edges

def dfs(origin, destination):
    # DFS logic to be implemented here, currently just returns nothing
    return None, 0, []

    # Rest of search logic to be implemented also.

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
