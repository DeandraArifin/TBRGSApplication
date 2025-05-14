import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from fawnmess import dataframe, findspeed

# load and preprocess the raw SCATS data into hourly Flow+Speed
df_raw    = pd.read_excel('Resources/Scats_Data_October_2006.xls',
                          sheet_name='Data', skiprows=1)
df_hourly = dataframe(df_raw)
df_hourly['SCATS Number'] = df_hourly['SCATS Number'].astype(int)

# choose the site and its saved model file
site_id    = 970
model_file = f'models/lstm_site_{site_id}.h5'

# ensure we have 24 hours of previous data
site_data = (df_hourly[df_hourly['SCATS Number'] == site_id]
             .sort_values(['Date', 'Hour']))
if len(site_data) < 24:
    raise RuntimeError('Need at least 24 hours of data for site', site_id)

# fit a scaler on [Flow,Speed] so we can normalize inputs exactly as during training
features = site_data[['Flow','Speed']].values
scaler   = MinMaxScaler().fit(features)

# prepare the last 24 hours for model input: scale then reshape to (1,24,2)
X_last24 = scaler.transform(features[-24:]).reshape(1, 24, 2)

# load the pre-trained model and predict
model = load_model(model_file, compile=False)
pred_scaled = model.predict(X_last24)[0]           # a length-2 array [flow, speed]
flow_pred, speed_pred = scaler.inverse_transform([pred_scaled])[0]

# print the next hour forecast
print(f'Next-hour Flow:  {flow_pred:.1f} vehicles/hour')
print(f'Next-hour Speed: {speed_pred:.1f} km/h')
