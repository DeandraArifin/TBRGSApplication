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

    scats_longform = scats_df.melt(id_vars=['Date', 'SCATS Number'], value_vars=intervals, var_name='Interval', value_name='Flow')

    scats_longform['IntervalNum'] = scats_longform['Interval'].str.extract(r'V(\d+)').astype(int)
    scats_longform['Time'] = pd.to_timedelta(scats_longform['IntervalNum'] * 15, unit='min')

    scats_longform['DateTime'] = scats_longform['Date'] + scats_longform['Time']

    scats_longform['Date'] = scats_longform['Date'].dt.normalize()
    scats_longform['Hour'] = scats_longform['DateTime'].dt.hour

    scats_longform = scats_longform.sort_values('Hour')

    #series = scats_longform[['SCATS Number','Date', 'Flow']]

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

    print(hourly_data)

    return hourly_data

def makemap(scats_data):
    scats_sites = scats_data[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()
    G = nx.Graph()

    for idx, row in scats_sites.iterrows():
        scats_id = row['SCATS Number']
        lat = row['NB_LATITUDE']
        lon = row['NB_LONGITUDE']
        G.add_node(scats_id, pos=(lon, lat))  

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=50)
    plt.title('SCATS Sites Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    
def main():
    scats_df = pd.read_excel('Resources/Scats_Data_October_2006.xls', sheet_name='Data', skiprows=1)
    hourly_data = dataframe(scats_df)
    makemap(scats_df)
    #print(hourly_data)

#haversine works by hs.haversine(loc1, loc2) where locs are (latitude, longitude)    

main()
