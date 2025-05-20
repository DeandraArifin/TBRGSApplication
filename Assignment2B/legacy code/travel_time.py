#!/usr/bin/env python3
import argparse
from datetime import datetime, date, timedelta
import pickle
import networkx as nx
import tensorflow as tf
from data_loader import load_interval_data, aggregate_hourly, load_site_coords
from adjacency import adj
from haversine import haversine
from train_all_models import find_speed

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

    # preload scalers & models
    scalers = {site: pickle.load(open(f"{SCALER_DIR}/{args.model}_site_{site}_scaler.pkl","rb")) for site in adj}
    models  = {site: tf.keras.models.load_model(f"{MODEL_DIR}/{args.model}_site_{site}.keras", compile=False)
               for site in adj}

    # load & aggregate
    flows15 = load_interval_data()
    flows15["SCATS Number"] = flows15["SCATS Number"].astype(str)
    hourly  = aggregate_hourly(flows15)
    hourly["SCATS Number"] = hourly["SCATS Number"].astype(str)

    # reachable set
    origin, dest = args.origin, args.destination
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

    # graph & Dijkstra
    _, coords = load_site_coords()
    G=nx.DiGraph()
    for u in reachable:
        for v in adj[u]:
            if v in reachable:
                d = haversine(coords[u],coords[v])
                spd = speeds.get(u,SPEED_LIMIT)
                tt = d/spd*3600 + INTERSECTION_DELAY
                print(f"Edge {u}->{v}: {d:.2f}km, {tt/60:.2f}min")
                G.add_edge(u,v,weight=tt)
    path=nx.dijkstra_path(G,origin,dest,weight="weight")
    tot=nx.dijkstra_path_length(G,origin,dest,weight="weight")
    m,s=divmod(tot,60)
    print("Fastest route:","->".join(path))
    print(f"Estimated time: {int(m)}m {int(s)}s")

if __name__=="__main__": main()
