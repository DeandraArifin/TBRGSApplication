import networkx as nx
import random as rnd
import matplotlib.pyplot as plt
import os

def randnodes():
    maxnode = rnd.randint(6, 12)
    li = list(range(1, maxnode + 1))
    return li

def randedges(G):
    edges = []

    for node in G.nodes():
        connected = [dest for (source, dest) in G.edges(node)]
        unconnected = [n for n in G.nodes() if not n in connected]

        if len(unconnected):
            if rnd.random() < 0.8:
                edge = rnd.choice(unconnected)
                if node != edge:
                    G.add_edge(node, edge, weight = rnd.randint(1,4))
                edges.append( (node, edge) )
                unconnected.remove(edge)
                connected.append(edge)

def draw(G, pos):
    plt.figure(1); plt.clf()
    fig, ax = plt.subplots(2,1, num=1, sharex=True, sharey=True)
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos=pos, ax=ax[0])
    plt.show()

def assign_pos(G):
    maxnode = max(G.nodes())
    pos = {node: (rnd.randint(1, maxnode), rnd.randint(1, maxnode)) for node in G.nodes}
    nx.set_node_attributes(G, pos, "position")
    return pos

def write(G, pos, dest, origin):
    directory = './GeneratedPaths/'
    i = 0
    os.chdir(directory)
    while os.path.exists(f"GenPathFinder{i}.txt"):
        i += 1
    f = open(f"GenPathFinder{i}.txt", "x")
    f.write("Nodes:\n")
    for node, (x, y) in pos.items():
        f.write(f"{node}: ({x},{y})\n")
    f.write(f"Edges: \n")
    for edge in G.edges:
        print(edge)
        cut = str(edge)
        cut = cut.strip()
        cut = cut.replace(" ", "") #this all just cleans up the formatting of saving edges
        f.write(f"{cut}: {rnd.randint(1,4)}\n")
    f.write(f"Origin:\n{origin}\nDestinations:\n{dest}")
    os.chdir("..")
    return i

def main():
    G = nx.DiGraph()
    li = randnodes()
    G.add_nodes_from(li)

    randedges(G)
    randedges(G) #deepen connections


    dest = rnd.randint(1, len(G.nodes))
    origin = rnd.choice([i for i in range(1, len(G.nodes)) if i != dest])

    for edge in G.edges:
        if edge[0] == edge[1]:
                print(f"{edge} deleted")
                G.remove_edge(edge)

    pos = assign_pos(G)
    index = write(G, pos, dest, origin)
    draw(G, pos)  
    return index

# main() #uncomment if you want to run this seperately to search.py


