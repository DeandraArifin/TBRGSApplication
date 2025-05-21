"""
Per-site Model Training

Trains one LSTM, GRU, and RNN per SCATS site
...
Adds pickling of each site's scaler into the 'scalers' folder.
"""
import argparse
import os
import pandas as pd
import numpy as np
import math
import pickle                              
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Conv1D, MaxPooling1D
from data_loader import load_interval_data, aggregate_hourly
import matplotlib.pyplot as plt

# CONFIG
INTERVAL_CSV = "Resources/interval_data.csv"
SEQ_LEN      = 24
TRAIN_SPLIT  = 0.8
MODEL_DIR    = "models"
RESULTS_DIR  = "results"
SCALER_DIR   = "scalers"                  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ensure directories exist
for d in (MODEL_DIR, RESULTS_DIR, SCALER_DIR):
    os.makedirs(d, exist_ok=True)

# function for finding speed from flow using the formula provided and applying constraints for free flow // congested flow
def find_speed(flow):
    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        print(f"Invalid discriminant for flow={flow}")
        return None

    sqrt_disc = math.sqrt(discriminant)
    speed1 = (-b + sqrt_disc) / (2*a)
    speed2 = (-b - sqrt_disc) / (2*a)

    valid_speeds = [s for s in (speed1, speed2) if s >= 0]
    if not valid_speeds:
        print(f"No valid positive speeds for flow={flow}, speeds={speed1}, {speed2}")
        return None

    chosen_speed = max(valid_speeds)

    if flow <= 351:
        return min(chosen_speed, 60)
    else:
        return chosen_speed
    
# loading data for training
flows15 = load_interval_data(INTERVAL_CSV)
hourly  = aggregate_hourly(flows15)
sites   = hourly['SCATS Number'].unique()

# prepare sequencing and scaling for a site
def prepare_site_data(df_site, seq_len=SEQ_LEN):
    df_sorted = df_site.sort_values('timestamp')
    values = df_sorted['flow'].values.reshape(-1,1)
    scaler = MinMaxScaler().fit(values)
    scaled = scaler.transform(values)
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len][0])
    return np.array(X), np.array(y), scaler

# build the model, depending on type, uses indentical layers for all models so the only difference is the algorithm
def build_model(model_type, input_shape):
    m = Sequential()
    m.add(Conv1D(128, 3, activation='relu', input_shape=input_shape))
    m.add(MaxPooling1D(2))
    if model_type == 'lstm':
        m.add(LSTM(64))
    elif model_type == 'gru':
        m.add(GRU(64))
    elif model_type == 'rnn':
        m.add(SimpleRNN(64))
    else:
        raise ValueError(f"Unknown model: {model_type}")
    m.add(Dense(64, activation='relu'))
    m.add(Dense(1))
    m.compile(optimizer='adam', loss='mse')
    return m

# per-site training and evaluation data
def train_site_models(model_type):
    results = []
    plot_dir = os.path.join(RESULTS_DIR, 'plots', model_type)
    os.makedirs(plot_dir, exist_ok=True)

    for site in sites:
        df_site = hourly[hourly['SCATS Number'] == site]
        if len(df_site) <= SEQ_LEN:
            continue
        # splits the data into training and testing sets
        X, y, scaler = prepare_site_data(df_site)
        split = int(len(X) * TRAIN_SPLIT)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        # building and fitting model
        model = build_model(model_type, (SEQ_LEN, 1))
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        # evaluate model
        pred = model.predict(X_test)
        pad_pred = np.pad(pred, ((0,0),(0,0)), 'constant')
        inv_pred = scaler.inverse_transform(pad_pred)[:,0]
        act = y_test.reshape(-1,1)
        inv_act = scaler.inverse_transform(act)[:,0]

        # calculate metrics
        mae = mean_absolute_error(inv_act, inv_pred)
        mse = mean_squared_error(inv_act, inv_pred)
        r2  = r2_score(inv_act, inv_pred)
        results.append((site, mae, mse, r2))
        print(f"Trained {model_type} site {site}: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

        # plot actual vs predicted with relative index
        idx = np.arange(len(inv_act))
        plt.figure(figsize=(8,4))
        plt.plot(idx, inv_act, label='Actual Flow')
        plt.plot(idx, inv_pred, label='Predicted Flow')
        plt.title(f"{model_type.upper()} Predictions vs Actual - Site {site}")
        plt.xlabel('Hours into test set')
        plt.ylabel('Flow per hour')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{site}_pred_vs_actual.png")
        plt.savefig(plot_path)
        plt.close()

        # save model
        model_path = os.path.join(MODEL_DIR, f"{model_type}_site_{site}.keras")
        keras.saving.save_model(model, model_path)

        # save scaler
        scaler_path = os.path.join(SCALER_DIR, f"{model_type}_site_{site}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    # save metrics CSV
    dfm = pd.DataFrame(results, columns=['site','mae','mse','r2'])
    metrics_path = os.path.join(RESULTS_DIR, f"{model_type}_metrics.csv")
    dfm.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # main entry
def main():
    parser = argparse.ArgumentParser(description="Train per-site flow models.")
    parser.add_argument("--model", type=str, choices=['lstm','gru','rnn'], required=True,
                        help="Model type to train (lstm, gru, rnn)")
    args = parser.parse_args()
    train_site_models(args.model)
    print(f"\nAll {args.model.upper()} models trained.")

if __name__ == '__main__':
    main()
