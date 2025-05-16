"""
Per-site Model Training

Trains one LSTM, GRU, and RNN per SCATS site
A lot of this was copied from fawnmess.py with some changes
For each site and model:
  - Scales features
  - Builds sliding-window sequences
  - Splits into train/test
  - Trains the model
  - Inverts scaling for prediction
  - Computes MAE, MSE, R²
  - Saves the Keras model
  - Collects metrics for CSV output
  - Plots actual vs predicted with relative index
"""
import argparse
import os
import pandas as pd
import numpy as np
import math
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ensure directories exist
for d in (MODEL_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

def findspeed(flow):
    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None #no real solution found
    
    sqrt_disc = math.sqrt(discriminant)
    speed1 = (-b + sqrt_disc) / (2*a)
    speed2 = (-b - sqrt_disc) / (2*a)

    return max(speed1, speed2) if speed1 >= 0 and speed2 >= 0 else (speed1 if speed1 >= 0 else speed2)

# preparing data for training
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

# build the model, depending on type.
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
        df_site = hourly[hourly['SCATS Number']==site]
        if len(df_site) <= SEQ_LEN:
            continue
        X, y, scaler = prepare_site_data(df_site)
        split = int(len(X) * TRAIN_SPLIT)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_model(model_type, (SEQ_LEN, 1))
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        pred = model.predict(X_test)
        pad_pred = np.pad(pred, ((0,0),(0,0)), 'constant')
        inv_pred = scaler.inverse_transform(pad_pred)[:,0]
        act = y_test.reshape(-1,1)
        inv_act = scaler.inverse_transform(act)[:,0]

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

        model_path = f"{MODEL_DIR}/{model_type}_site_{site}.h5"
        keras.saving.save_model(model, model_path)

    # save metrics CSV
    dfm = pd.DataFrame(results, columns=['site','mae','mse','r2'])
    metrics_path = f"{RESULTS_DIR}/{model_type}_metrics.csv"
    dfm.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

#  main entry, with argument parsing. to use python train_all_models.py --model lstm/gru/rnn CAN TRAIN ALL IN PARALLEL FOR TOTAL TRAIN TIME OF AROUND 3 MINUTES (on my machine)
def main():
    parser = argparse.ArgumentParser(description="Train per-site flow models.")
    parser.add_argument("--model", type=str, choices=['lstm','gru','rnn'], required=True,
                        help="Model type to train (lstm, gru, rnn)")
    args = parser.parse_args()
    train_site_models(args.model)
    print(f"\nAll {args.model.upper()} models trained.")

if __name__ == '__main__':
    main()
