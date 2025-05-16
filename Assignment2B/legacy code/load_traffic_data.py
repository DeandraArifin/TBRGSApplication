import pandas as pd
import numpy as np

def load_site_traffic(file_path, site_id):
    df = pd.read_excel(file_path, sheet_name='Data', skiprows=1)

    # Clean SCATS site ID
    df['SCATS Number'] = df['SCATS Number'].astype(str).str.strip()
    site_df = df[df['SCATS Number'] == site_id]

    # Build full list of volume columns (V00 to V95)
    volume_cols = [f'V{str(i).zfill(2)}' for i in range(96)]

    # Melt into long format (one row per interval)
    melted = site_df.melt(id_vars=['Date'], value_vars=volume_cols,
                          var_name='Interval', value_name='Volume')

    # Extract minutes from interval
    melted['Interval_Num'] = melted['Interval'].str.extract(r'V(\d+)').astype(int)
    melted['Timestamp'] = pd.to_datetime(melted['Date']) + pd.to_timedelta(melted['Interval_Num'] * 15, unit='m') - pd.to_timedelta(15, unit='m')

    # Combine all lanes — sum traffic volume (Could be wrong but not doing so was going to be a pain)
    melted = melted.groupby('Timestamp', as_index=False)['Volume'].sum()

    return melted[['Timestamp', 'Volume']]

# creates the sequences for the models
def create_sequences(series, window=4):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)