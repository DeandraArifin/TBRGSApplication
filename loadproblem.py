import sys
import re

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

#initialisation
nodes = {}
edges = {}
origin = None
destinations = []

def loadproblem(filename): # Changed this so it will accept filename passed from search.py
    with open(filename) as f:
        mydict = {}
        current_section = None
        for line in f:
            line = line.strip()

            if line.endswith(':'):
                current_section = line[:-1]
                mydict[current_section] = []

            elif current_section == 'Nodes' and re.match(r'^\d+:', line): #r = raw string, \d+ = one or more digits
                match = re.match(r'(\d+): \((\d+),(\d+)\)', line)
                if match:  
                    node_id = int(match.group(1))
                    node_x = int(match.group(2))
                    node_y = int(match.group(3))
                    nodes[node_id] = (node_x, node_y)

            elif current_section == 'Edges' and re.match(r'^\(\d+,\d+\):', line): #matchese the (2,1): mapping a literal , and :
                match = re.match(r'\((\d+),(\d+)\): (\d+)', line) #\( \) = grouping
                if match: 
                    source_node = int(match.group(1))
                    dest_node = int(match.group(2))
                    weight = int(match.group(3))
                    edges[(source_node, dest_node)] = weight #relationship between two nodes, mapping to edge weight

            elif current_section == 'Origin' and line.isdigit():
                origin = int(line)
                
            elif current_section == 'Destinations':
                if ";" in line:
                     destinations = list(map(int, line.split(';'))) #split at the semicolon
                else:
                    destinations = int(line.strip())
               
    return origin, destinations #have to return these because they're local unlike node + edges 

# Commenting the test lines out - now in search.py

# (origin, destinations) = loadproblem()
# print("Nodes: ", nodes)
# print("Edges: ", edges)
# print("Origin: ", origin)
# print("Destination: ", destinations)