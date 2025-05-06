import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
from sklearn.preprocessing import MinMaxScaler
import math

scats_df = pd.read_excel('Resources\Scats_Data_October_2006.xls', sheet_name='Data', skiprows=1)
print(scats_df.columns)
scats_df.columns = scats_df.columns.str.strip()

scats_df['Date'] = pd.to_datetime(scats_df['Date'], dayfirst=True)

intervals = [f'V{i:02}' for i in range(96)] #V00 to V95

scats_longform = scats_df.melt(id_vars='Date', value_vars=intervals, var_name='Interval', value_name='Flow')

scats_longform['IntervalNum'] = scats_longform['Interval'].str.extract(r'V(\d+)').astype(int)
scats_longform['Time'] = pd.to_timedelta(scats_longform['IntervalNum'] * 15, unit='min')

scats_longform['DateTime'] = scats_longform['Date'] + scats_longform['Time']

scats_longform = scats_longform.sort_values('DateTime')

series = scats_longform[['DateTime', 'Flow']]

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

series['speed'] = np.nan

for flow in series['flow']:
    

print(series)
