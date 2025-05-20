import sys
import re
import networkx as nx
import matplotlib.pyplot as plt

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
                match = re.match(r'(\d+): \((-?\d+),(-?\d+)\)', line) # 4/04 Changed this so we can now handle nodes that are negative, eg at location (-1, 1). Not sure if this matters but tested and shouldn't break anything
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
               
    return origin, destinations, edges, nodes #have to return these because they're local unlike node + edges

def todraw(nodes, edges, origin, destinations):
    newg = nx.DiGraph()
    for node_id, (node_x, node_y) in nodes.items():
        newg.add_node(node_id, pos=(node_x, node_y))
    for edge, weight in edges.items():
        newg.add_edge(*edge, weight=weight) #* to unpack source and dest
    pos = nx.get_node_attributes(newg, 'pos')
    colour_map = False
    draw(newg, pos, colour_map, origin, destinations)
    return newg, pos

def draw(newg, pos, colour_map, origin, destinations):
    #nodes+edges
    if colour_map == False:
        nx.draw_networkx_nodes(newg, pos, node_size=500, node_color="#D295BF")
    else:
        nx.draw_networkx_nodes(newg, pos, node_size=500, node_color=colour_map)
    
    nx.draw_networkx_edges(
        newg, pos, width=4, alpha=1, edge_color="#77507c",
        style="solid", arrowstyle="->", arrowsize=25)
    #labels
    if type(destinations)==int:
        dest = destinations
        destinations = []
        destinations.append(dest)
    labels = {}
    for node in newg.nodes:
        if node == origin:
            labels[node] = f"\n{node}\nOrigin"
        elif node in destinations :
            labels[node] = f"\n{node}\nDest."
        else:
            labels[node] = str(node)

    nx.draw_networkx_labels(newg, pos, font_size=15, font_family="sans-serif", labels=labels)
    edge_labels = nx.get_edge_attributes(newg, "weight")
    nx.draw_networkx_edge_labels(newg, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()