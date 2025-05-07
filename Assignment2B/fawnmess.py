import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
import math
import haversine as hs
import networkx as nx
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TF_ENABLE_ONEDNN_OPTS=0

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
    #print(scats_df.columns)
    scats_df.columns = scats_df.columns.str.strip()

    scats_df['Date'] = pd.to_datetime(scats_df['Date'], dayfirst=True)

    intervals = [f'V{i:02}' for i in range(96)] #V00 to V95

    scats_longform = scats_df.melt(id_vars=['Date', 'SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE'], value_vars=intervals, var_name='Interval', value_name='Flow')
    # print(scats_longform)

    scats_longform['IntervalNum'] = scats_longform['Interval'].str.extract(r'V(\d+)').astype(int)
    scats_longform['Time'] = pd.to_timedelta(scats_longform['IntervalNum'] * 15, unit='min')

    # scats_longform['DateTime'] = scats_longform['Date'] + scats_longform['Time']

    scats_longform['Date'] = scats_longform['Date'].dt.normalize()
    scats_longform['Hour'] = (scats_longform['IntervalNum'] * 15) // 60

    # print(scats_longform['DateTime', 'Hour'])

    scats_longform = scats_longform.sort_values('Hour')

    series = scats_longform[['NB_LATITUDE','NB_LONGITUDE','Date', 'Flow']]

    hourly_data = scats_longform.groupby(['SCATS Number', 'Date', 'Hour'])['Flow'].sum().reset_index()
    hourly_data['Speed'] = hourly_data['Flow'].apply(findspeed)

    uniqueSCATS = hourly_data['SCATS Number'].unique()

    for i in uniqueSCATS:
        mean = hourly_data.loc[hourly_data['SCATS Number'].eq(i), 'Speed'].mean()
        print(f"Mean of {i} is {mean}")
        condition = (hourly_data['SCATS Number'] == i) & (hourly_data['Speed'].isna())
        hourly_data.loc[condition, 'Speed'] = mean

    false_ints = hourly_data[hourly_data['SCATS Number'] == False].index
    hourly_data = hourly_data.drop(false_ints)

    #print(hourly_data)

    return hourly_data

def prepare_all_sites(hourly_data, seq_length=24):
    all_X, all_y = [], []
    site_ids = hourly_data['SCATS Number'].unique() #might want to change to long/lat instead of scats number
    scalers = {}
    for site_id in site_ids:
        site_data = hourly_data[hourly_data['SCATS Number'] == site_id].sort_values(['Date', 'Hour'])
        features = site_data[['Flow', 'Speed']].values

        if len(features) <= seq_length:
            continue  # Not enough data to form a sequence

        # Create and fit a scaler for this site
        scaler = MinMaxScaler()
        scaler.fit(features)
        scalers[site_id] = scaler

        scaled = scaler.transform(features)

        for i in range(len(scaled) - seq_length):
            X_seq = scaled[i:i + seq_length]
            y_seq = scaled[i + seq_length]
            all_X.append(X_seq)
            all_y.append(y_seq)
    return np.array(all_X), np.array(all_y), scaler

def build_lstm(input_shape):
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        keras.layers.LSTM(128, input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'), 
        keras.layers.Dense(input_shape[1])])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(timesteps, features):
    input_shape = (timesteps, features)
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.GRU(128),
        keras.layers.Dense(64, activation='relu'), 
        keras.layers.Dense(features)])
    model.compile(optimizer='adam', loss='mse')
    return model

def prep(hourly_data):
    X, y, scaler = prepare_all_sites(hourly_data)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:] #either side of split
    y_train, y_test = y[:split], y[split:]
    model = sys.argv[1]
    if model == "lstm".lower():
        model = build_lstm((X.shape[1], X.shape[2]))
    if model == 'gru'.lower():
        model = build_gru(X.shape[1], X.shape[2])
    model.summary()
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
    preditctions = model.predict(X_test)
    predicted = scaler.inverse_transform(preditctions)
    actual = scaler.inverse_transform(y_test)
    return predicted, actual

def makemap(predicted, actual):
    plt.figure(figsize=(12, 6))
    plt.plot(actual[:, 0], label='Actual Flow')
    plt.plot(predicted[:, 0], label="Predicted Flow")
    plt.title('Flow Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Flow')
    plt.legend()
    plt.show()

def main():
    scats_df = pd.read_excel('Resources/Scats_Data_October_2006.xls', sheet_name='Data', skiprows=1)
    hourly_data = dataframe(scats_df)
    #print(hourly_data)
    predicted, actual = prep(hourly_data)
    results_df = pd.DataFrame({'Actual Flow': actual[:, 0],
        'Predicted Flow': predicted[:, 0],
        'Actual Speed': actual[:, 1],
        'Predicted Speed': predicted[:, 1]})
    results_df.to_csv('predicted_vs_actual.csv', index=False)
    makemap(predicted, actual)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')
#haversine works by hs.haversine(loc1, loc2) where locs are (latitude, longitude)    

if __name__ == "__main__":
      main()
