#!/usr/bin/env python3
import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime, date, timedelta
import pickle
import networkx as nx
import keras
from data_loader import load_interval_data, aggregate_hourly, load_site_coords
from adjacency import adj
from haversine import haversine
from train_all_models import find_speed
import search 
import loadproblem

# ── CONFIG ─────────────────────────────────────────────────
DEFAULT_DATE       = date(2006, 10, 31)
INTERSECTION_DELAY = 30       # seconds per intersection/node
SPEED_LIMIT        = 60       # km/h speed cap
# Fundamental diagram constants
A_CONST            = -1.4648375
B_CONST            = 93.75
# Theoretical maximum flow Qmax = -b^2/(4a)
FLOW_CAPACITY      = -B_CONST*B_CONST/(4*A_CONST)
MODEL_DIR          = "models"   # files: {model}_site_{site}.h5
SCALER_DIR         = "scalers"  # files: {site}_scaler.pkl
SEQ_LEN            = 24       # hours of history
# ────────────────────────────────────────────────────────────

def pathfind(i, ori, dest, times):
    filename = f"Test_{ori}_to_{dest}_{i}.txt"
    origin, destination, edges, nodes = loadproblem.loadproblem(filename)
    adjacency_list = search.convert_to_adjacency_list(loadproblem.edges)
    ourGraph = search.Graph(adjacency_list)
    ourGraph.locations = loadproblem.nodes
    k = 5
    paths = []
    for attempt in range(k):
        copy_adj_list = {currnode: dict(nxtnode) for currnode, nxtnode in adjacency_list.items()}
        for idx, path in enumerate(paths): #loops through list of paths
            if len(path) > 1: #make sure the path has at least one edge
                i = idx % (len(path) - 1)  # selects a node between 0 and len(path) - 2. % to rotate each edge between loops
                currnode, nxtnode = path[i], path[i+1] 
                if nxtnode in copy_adj_list.get(currnode, {}): #checks if they already exist in the copied list and removes them if it does
                    del copy_adj_list[currnode][nxtnode]

        ourGraph = search.Graph(copy_adj_list)
        ourGraph.locations = nodes

        result, explored, goal_state = search.runOurGraph(ourGraph, origin, destination, search.astar_search)
        if not result or any(result == n for n in paths): #doesnt break but does skip the tried node
            continue
        paths.append(result) 
        nodes_expanded = explored.union(result)
        print(f"{filename} A* Search Path {attempt + 1}")
        print(f"goal = {goal_state}, number_of_nodes = {len(nodes_expanded)}")
        timeadd = []
        timeadd.append(float(times[f"{origin},{result[0]}"]))
        i = 0
        #print(f"{origin} + {result[0]} = {times[f"{origin},{result[0]}"]}")
        for n in result:
            if 0 <= i+1 < len(result):
                #print(f"{n} + {result[i+1]} = {times[f"{n},{result[i+1]}"]}")
                timeadd.append(float(times[f"{n},{result[i+1]}"]))
            i += 1
        print(origin, "->", " -> ".join(map(str, result)))
        print(f"Total time of journey: {sum(timeadd)}mins")
    if not paths:
        print("No paths found.")

    return paths

def write(G, origin, dest): #method based on testcasegen.py from Assignment 2A
        base_dir = os.path.dirname(os.path.abspath(__file__))
        directory = os.path.join(base_dir, "Tests")
        
        i = 0
        os.chdir(directory)
        while os.path.exists(f"Test_{origin}_to_{dest}_{i}.txt"):
            i += 1
        f = open(f"Test_{origin}_to_{dest}_{i}.txt", "x") 
        f.write("Nodes:\n")
        for node in G.nodes():
            cut = G.nodes[node]['pos'][0]
            cut = str(cut).replace(" ", "")
            f.write(f"{node}: {cut}\n")
        f.write(f"Edges: \n")
        for edge in G.edges(data=True):
            node1, node2, data = edge
            cut = str((node1, node2)).replace(" ", "")
            cut = cut.replace("'", "")
            weight = data.get('weight')
            f.write(f"{cut}: {weight}\n")
        f.write(f"Origin:\n{origin}\nDestinations:\n{dest}")
        return i

def load(model):
    # preload scalers & models
    print("Loading models....")
    scalers = {site: pickle.load(open(f"{SCALER_DIR}/{model}_site_{site}_scaler.pkl","rb")) for site in adj}
    models  = {site: keras.models.load_model(f"{MODEL_DIR}/{model}_site_{site}.keras", compile=False)
               for site in adj}

    # load & aggregate
    flows15 = load_interval_data()
    flows15["SCATS Number"] = flows15["SCATS Number"].astype(str)
    hourly  = aggregate_hourly(flows15)
    hourly["SCATS Number"] = hourly["SCATS Number"].astype(str)

    return scalers, models, hourly

def reachset(origin, dest):
    # origin, dest = args.origin, args.destination
    if origin not in adj or dest not in adj:
        raise ValueError("Origin or destination not in adjacency list")
    reachable = {origin}; stack=[origin]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in reachable:
                reachable.add(v); stack.append(v)
    if dest not in reachable:
        raise ValueError(f"No path from {origin} to {dest}")
    return reachable
    
def findspeed(scalers, models, hourly, dep_time, reachable):
     # predict & speed
    speeds = {}
    start = dep_time - timedelta(hours=SEQ_LEN)
    FLOW_CAPACITY_15MIN = -B_CONST**2 / (4 * A_CONST)  # should equal ~1500 vehicles/15-min

    for site in reachable:
        df = hourly[hourly["SCATS Number"] == site]
        hist = df[(df["timestamp"] > start) & (df["timestamp"] <= dep_time)]

        if len(hist) == SEQ_LEN:
            vals = hist["flow"].values.reshape(1, SEQ_LEN, 1)
            scaler = scalers[site]
            scaled = scaler.transform(vals.reshape(-1, 1)).reshape(1, SEQ_LEN, 1)
            pred = models[site].predict(scaled, verbose=0)
            flow_hourly = float(scaler.inverse_transform(pred)[0, 0])

            # convert hourly flow prediction to 15-minute flow THE hack ... was getting none speed values otherwise
            flow_15min = flow_hourly / 4

            if flow_15min > FLOW_CAPACITY_15MIN:
                print(f"Site {site}: flow {flow_15min:.1f} exceeds capacity, capping at {FLOW_CAPACITY_15MIN:.1f}")
                flow_15min = FLOW_CAPACITY_15MIN

            spd = find_speed(flow_15min)

            if spd is not None:
                print(f"Site {site}: hourly flow={flow_hourly:.1f}, 15-min flow={flow_15min:.1f} → speed={spd:.1f}")
                speeds[site] = spd
            else:
                print(f"Site {site}: flow produced invalid speed, defaulting to speed limit")
                speeds[site] = SPEED_LIMIT
        else:
            print(f"Site {site}: insufficient history ({len(hist)} rows), defaulting to speed limit")
            speeds[site] = SPEED_LIMIT
    return speeds

def graph(reachable, speeds):
        # graph
    _, coords = load_site_coords()
    times = dict()
    G=nx.DiGraph()
    for u in reachable:
        for v in adj[u]:
            if v in reachable:
                d = haversine(coords[u],coords[v])
                spd = speeds.get(u,SPEED_LIMIT)
                tt = d/spd*3600 + INTERSECTION_DELAY
                times.update({f"{u},{v}": f"{tt/60:.2f}"})
                print(f"Edge {u}->{v}: {d:.2f}km, {tt/60:.2f}min")
                G.add_edge(u,v,weight=tt)
    for node in G.nodes:
        if node in coords:
            G.nodes[node]['pos'] = coords[u], coords[v]
    return G, times

#just takes arguments from the gui instead of the command line
def run_model(origin, destination, time, model, scalers, models, hourly):
    t = datetime.strptime(time, "%H:%M").time()
    dep_time = datetime.combine(DEFAULT_DATE, t)
    
    if scalers is None and models is None and hourly is None:
        scalers, models, hourly = load(model.lower())
        
    reachable = reachset(origin, destination)
    speeds = findspeed(scalers, models, hourly, dep_time, reachable)
    G, times = graph(reachable, speeds)
    index = write(G, origin, destination)
    paths = pathfind(index, origin, destination)
    for path in paths:
        if path[0] != origin:
            path.insert(0,origin)
            
    return paths
    

def main():
    parser = argparse.ArgumentParser(
        description="Estimate fastest travel time using per-site TensorFlow models"
    )
    parser.add_argument("--origin",      required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--time",        required=True)
    parser.add_argument("--model",       default="gru", choices=["gru","lstm","rnn"])
    args = parser.parse_args()

    # parse departure time
    t = datetime.strptime(args.time, "%H:%M").time()
    dep_time = datetime.combine(DEFAULT_DATE, t)

    origin, dest, model = args.origin, args.destination, args.model

    scalers, models, hourly = load(model) 

    reachable = reachset(origin, dest)

    speeds = findspeed(scalers, models, hourly, dep_time, reachable)
 
    G, times = graph(reachable, speeds)

    index = write(G, origin, dest)

    paths = pathfind(index, origin, dest, times)

if __name__=="__main__": main()
