import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
from sklearn.preprocessing import MinMaxScaler
import math
import haversine as hs
import networkx as nx

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
    scaler = MinMaxScaler()
    all_feats = hourly_data[['Flow', 'Speed']].values
    scaler.fit(all_feats)
    for id in site_ids:
        data = hourly_data[hourly_data['SCATS Number'] == id].sort_values(['Date', 'Hour'])
        features = data[['Flow', 'Speed']].values
        if len(features) <= seq_length: #skips incomplete sequences in sites
            continue
        scaled = scaler.transform(features)
        for i in range(len(scaled) - seq_length):
            X_seq = scaled[i:i + seq_length]
            y_seq = scaled[1 + seq_length]
            all_X.append(X_seq)
            all_y.append(y_seq)
    return np.array(all_X), np.array(all_y), scaler

def build_model(input_shape):
    model = keras.models.Sequential([keras.layers.LSTM(64, input_shape=input_shape), keras.layers.Dense(32, activation='relu'), keras.layers.Dense(input_shape[1])])
    model.compile(optimizer='adam', loss='mse')
    return model

def prep(hourly_data):
    X, y, scaler = prepare_all_sites(hourly_data)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:] #either side of split
    y_train, y_test = y[:split], y[split:]
    model = build_model((X.shape[1], X.shape[2]))
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
    makemap(predicted, actual)

#haversine works by hs.haversine(loc1, loc2) where locs are (latitude, longitude)    

main()
