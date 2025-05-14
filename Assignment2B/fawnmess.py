import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import math
import haversine as hs
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Disable oneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

def dataframe(scats_df):
    scats_df.columns = scats_df.columns.str.strip()
    scats_df['Date'] = pd.to_datetime(scats_df['Date'], dayfirst=True)
    intervals = [f'V{i:02}' for i in range(96)]

    long_df = scats_df.melt(
        id_vars=['Date', 'SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE'],
        value_vars=intervals,
        var_name='Interval',
        value_name='Flow'
    )
    long_df['IntervalNum'] = long_df['Interval'].str.extract(r'V(\d+)').astype(int)
    long_df['Hour'] = (long_df['IntervalNum'] * 15) // 60

    hourly = long_df.groupby(
        ['SCATS Number', 'Date', 'Hour']
    )['Flow'].sum().reset_index()
    hourly['Speed'] = hourly['Flow'].apply(findspeed)

    # fill NaNs per site with the site mean
    for site in hourly['SCATS Number'].unique():
        mean_speed = hourly.loc[hourly['SCATS Number']==site, 'Speed'].mean()
        mask = (hourly['SCATS Number']==site) & (hourly['Speed'].isna())
        hourly.loc[mask, 'Speed'] = mean_speed

    #temporal features based from previous assessment
    hourly['DayOfWeek'] = hourly['Date'].dt.dayofweek
    hourly['IsWeekend'] = hourly['DayOfWeek'].isin([5,6]).astype(int)
    hourly['Hour_sin'] = np.sin(2 * np.pi * hourly['Hour'] / 24)
    hourly['Hour_cos'] = np.cos(2 * np.pi * hourly['Hour'] / 24)
    hourly['DOW_sin'] = np.sin(2 * np.pi * hourly['DayOfWeek'] / 7)
    hourly['DOW_cos'] = np.cos(2 * np.pi * hourly['DayOfWeek'] / 7)

    # drop any invalid SCATS Number entries
    hourly = hourly[hourly['SCATS Number'].astype(bool)]
    return hourly


def prepare_all_sites(hourly_data, seq_length=24):
    all_X, all_y = [], []
    scalers = {}
    site_ids = hourly_data['SCATS Number'].unique()
    for site_id in site_ids:
        site_df = hourly_data[hourly_data['SCATS Number']==site_id].sort_values(['Date','Hour'])
        feature_cols = ['Flow', 'Speed', 'Hour_sin', 'Hour_cos', 'DayOfWeek', 'IsWeekend']
        features = site_df[feature_cols].values
        if len(features) <= seq_length:
            continue  # Not enough data to form a sequence

        # Create and fit a scaler for this site
        scaler = MinMaxScaler()
        scaler.fit(features)
        scalers[site_id] = scaler
        scaled = scaler.transform(features)

        for i in range(len(scaled) - seq_length):
            X_seq = scaled[i:i + seq_length]
            y_seq = scaled[i + seq_length][0]  # Only predict flow
            all_X.append(X_seq)
            all_y.append(y_seq)

    return np.array(all_X), np.array(all_y), scaler


def build_lstm(input_shape):
    model = keras.models.Sequential([
        keras.layers.Input(input_shape),
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        #keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        keras.layers.LSTM(64),
        keras.layers.Dense(64, activation='relu'), 
        keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss=keras.losses.Huber())
    return model


def build_gru(timesteps, features):
    input_shape = (timesteps, features)
    model = keras.models.Sequential([
        keras.layers.Input(input_shape),
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.GRU(64),
        keras.layers.Dense(64, activation='relu'), 
        keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss=keras.losses.Huber())
    return model

def build_rnn(timesteps, features):
    input_shape = (timesteps, features)
    model = keras.models.Sequential([
        keras.layers.Input(input_shape),
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        keras.layers.MaxPool1D(pool_size=2),
        keras.layers.SimpleRNN(64),
        keras.layers.Dense(64, activation='relu'), 
        keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss=keras.losses.Huber())
    return model

def prep_all_sites(hourly_data, model_type):
    """Train a separate model on each SCATS site."""
    results = []
    os.makedirs('models', exist_ok=True)
    for site in hourly_data['SCATS Number'].unique():
        site_data = hourly_data[hourly_data['SCATS Number']==site].copy()
        X, y, scaler = prepare_all_sites(site_data, seq_length=24)
        if len(X) == 0:
            continue

        # split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # build
        if model_type == 'lstm':
            model = build_lstm((X.shape[1], X.shape[2]))
        elif model_type == 'gru': # Change this when adding more models
            model = build_gru(X.shape[1], X.shape[2])
        elif model_type == 'rnn':
            model = build_rnn(X.shape[1], X.shape[2])

        # train
        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

        # predict & evaluate
        pred = model.predict(X_test)
        pad = np.pad(pred, ((0, 0), (0, 5)), mode='constant')
        pred_inv = scaler.inverse_transform(pad)[:, 0] #only take the flow column
        y_testpad = np.pad(y_test.reshape(-1, 1), ((0, 0), (0, 5)), mode='constant')
        act_inv  = scaler.inverse_transform(y_testpad)[:, 0] #only take flow
        mae = mean_absolute_error(act_inv, pred_inv)
        mse = mean_squared_error(act_inv, pred_inv)
        r2  = r2_score(act_inv, pred_inv)
        print(f"Site {site}: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2*100:.2f}%")

        # save model & metrics
        keras.saving.save_model(model, f'models/{model_type}site{site}.h5')
        results.append((site, mae, mse, r2))

    return results


def makemap(predicted, actual):
    plt.figure(figsize=(12, 6))
    plt.plot(actual[:, 0], label='Actual Flow')
    plt.plot(predicted[:, 0], label='Predicted Flow')
    plt.title('Flow Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Flow')
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python fawnmess.py [lstm|gru|rnn]")
        sys.exit(1)
    model_type = sys.argv[1].lower()
    assert model_type in ('lstm', 'gru', 'rnn'), "Model must be 'lstm', 'gru' or 'rnn'"

    scats_df = pd.read_excel(
        'Resources/Scats_Data_October_2006.xls',
        sheet_name='Data',
        skiprows=1
    )
    hourly_data = dataframe(scats_df)

    scores = prep_all_sites(hourly_data, model_type)

    # save metrics summary
    metrics_df = pd.DataFrame(scores, columns=['site', 'mae', 'mse', 'r2'])
    metrics_df.to_csv(f"{model_type}_site_metrics.csv", index=False)

    print(f"Saved per-site metrics to {model_type}_site_metrics.csv")

if __name__ == "__main__":
    main()
