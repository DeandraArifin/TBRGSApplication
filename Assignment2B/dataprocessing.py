import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Disable oneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#gonna keep this for future use
# def findspeed(flow):
#     a = -1.4648375
#     b = 93.75
#     c = -flow

#     discriminant = b**2 - 4*a*c
#     if discriminant < 0:
#         return None #no real solution found
    
#     sqrt_disc = math.sqrt(discriminant)
#     speed1 = (-b + sqrt_disc) / (2*a)
#     speed2 = (-b - sqrt_disc) / (2*a)

#     return max(speed1, speed2) if speed1 >= 0 and speed2 >= 0 else (speed1 if speed1 >= 0 else speed2)

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
    # print(long_df['IntervalNum'])
    # long_df['Hour'] = (long_df['IntervalNum'] * 15) // 60

    # hourly = long_df.groupby(
    #     ['SCATS Number', 'Date', 'Hour']
    # )['Flow'].sum().reset_index()
    
    interval_data = long_df.groupby(
        ['SCATS Number', 'Date', 'IntervalNum']
    )['Flow'].sum().reset_index()
    interval_data['Hour'] = (interval_data['IntervalNum'] * 15) // 60
    # hourly['Speed'] = hourly['Flow'].apply(findspeed)

    # # fill NaNs per site with the site mean
    # for site in hourly['SCATS Number'].unique():
    #     mean_speed = hourly.loc[hourly['SCATS Number']==site, 'Speed'].mean()
    #     mask = (hourly['SCATS Number']==site) & (hourly['Speed'].isna())
    #     hourly.loc[mask, 'Speed'] = mean_speed

    # drop any invalid SCATS Number entries
    # hourly = hourly[hourly['SCATS Number'].astype(bool)]
    interval_data = interval_data[interval_data['SCATS Number'].astype(bool)]
    return interval_data

def chart_correlation_matrix(df):
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['number'])

    # Compute correlation matrix
    corr = numeric_df.corr()

    # Check if it's 2D and has at least 2 numeric features
    if corr.shape[0] < 2:
        print("Not enough numeric columns to compute a correlation matrix.")
        return None

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')  # annot=True if you want values shown
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.show()

    return corr

    
def main():
    scats_df = pd.read_csv(
        'Resources/merged_data.csv'
    )
    interval_data = dataframe(scats_df)
    chart_correlation_matrix(interval_data)
    interval_data.to_csv('Resources/interval_data.csv', index=False)
    print(interval_data)

if __name__ == "__main__":
    main()
