import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import heapq
import numpy as np
import pandas as pd
import networkx as nx
from haversine import haversine
from keras.models import load_model
from keras.losses import MeanSquaredError
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

#TODO: Add in hour value to preictions, Improve error handling, squash bugs


# Load SCATS site info with lat/lon
scats_df = pd.read_excel('Resources/Scats_Data_October_2006.xls', sheet_name='Data', skiprows=1)
sites_df = scats_df[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].dropna().drop_duplicates()
sites_df.columns = ['id', 'lat', 'lon']
sites_df['id'] = sites_df['id'].astype(int)

# Flow to speed conversion (robbed from fawn)
def findspeed(flow):
    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None 
    
    sqrt_disc = math.sqrt(discriminant)
    speed1 = (-b + sqrt_disc) / (2*a)
    speed2 = (-b - sqrt_disc) / (2*a)

    return max(speed1, speed2) if speed1 >= 0 and speed2 >= 0 else (speed1 if speed1 >= 0 else speed2)

# Model cache
model_cache = {}
def predict_flow(site_id, model_type='lstm', seq_len=24):
    model_key = f"{model_type}_{site_id}"
    if model_key not in model_cache:
        path = f"models/{model_type}_site_{site_id}.h5"
        if not os.path.exists(path):
            print(f"!!! Model not found for site {site_id} — using default flow")
            return 300
        try:
            model = load_model(path, compile=False)
            model.compile(optimizer='adam', loss=MeanSquaredError())
            model_cache[model_key] = model
        except Exception as e:
            print(f"!!! Error loading model for site {site_id}: {e}")
            return 300
    model = model_cache[model_key]
    dummy_seq = np.zeros((1, seq_len, 2), dtype=np.float32)
    pred = model.predict(dummy_seq, verbose=0)[0][0]
    return max(pred, 10)

# Build SCATS graph
def build_graph(threshold_km=5):
    G = nx.Graph()
    for _, row in sites_df.iterrows():
        G.add_node(int(row['id']), pos=(row['lat'], row['lon']))
    for i, row1 in sites_df.iterrows():
        for j, row2 in sites_df.iterrows():
            if row1['id'] == row2['id']:
                continue
            dist = haversine((row1['lat'], row1['lon']), (row2['lat'], row2['lon']))
            if dist <= threshold_km:
                G.add_edge(int(row1['id']), int(row2['id']), distance=dist)
    return G

# Edge cost (travel time)
def edge_travel_time(scats_a, scats_b, model_type):
    flow = predict_flow(scats_b, model_type)
    speed = findspeed(flow)
    latlon_a = tuple(sites_df.loc[sites_df['id']==scats_a][['lat','lon']].values[0])
    latlon_b = tuple(sites_df.loc[sites_df['id']==scats_b][['lat','lon']].values[0])
    dist_km = haversine(latlon_a, latlon_b)
    if speed < 1: speed = 1
    time = (dist_km / speed) * 60 + 0.5  # minutes + delay
    return round(time, 2)

# Assign travel time to edges
def set_edge_weights(G, model_type):
    for u, v in G.edges():
        G[u][v]['time'] = edge_travel_time(u, v, model_type)
    return G

# Dijkstra
def dijkstra(G, source, target):
    q = [(0, source, [])]
    visited = set()
    while q:
        cost, node, path = heapq.heappop(q)
        if node in visited:
            continue
        path = path + [node]
        if node == target:
            return (cost, path)
        visited.add(node)
        for neighbor in G.neighbors(node):
            edge_time = G[node][neighbor]['time']
            heapq.heappush(q, (cost + edge_time, neighbor, path))
    return (float('inf'), [])

# Top-K paths
def top_k_paths(G, origin, dest, k=5):
    paths = []
    for _ in range(k):
        cost, path = dijkstra(G, origin, dest)
        if not path or path in [p[1] for p in paths]:
            break
        paths.append((cost, path))
        if len(path) > 1:
            G.remove_edge(path[0], path[1])
    return paths

def get_routes(origin, dest, model_type):
    G = build_graph()
    G = set_edge_weights(G, model_type)
    return top_k_paths(G, origin, dest)

# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python marcushellhole.py <model_type> <source_scats> <destination_scats>")
        print("Example: python marcushellhole.py gru 2000 970")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    source = int(sys.argv[2])
    dest = int(sys.argv[3])

    print(f"Finding routes using model: {model_type.upper()}")
    routes = get_routes(source, dest, model_type)

    if not routes:
        print("No routes found.")
    else:
        for i, (time, path) in enumerate(routes, start=1):
            path_str = " → ".join(str(p) for p in path)
            print(f"\nRoute {i}:")
            print(f"  Time: {time:.2f} minutes")
            print(f"  Path: {path_str}")
