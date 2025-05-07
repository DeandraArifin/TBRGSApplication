import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from load_traffic_data import load_site_traffic, create_sequences

# Load data using load_site_traffic - function defined within load_traffic_data
data = load_site_traffic("Resources/Scats_Data_October_2006.xls", "2200")

# Normalize volume (models expect normalized data)
data = data.copy()
scaler = MinMaxScaler()
data['Volume'] = scaler.fit_transform(data[['Volume']])


# Create sequences for the model to train on - create_sequences function defined within load_traffic_data
window_size = 4
X, y = create_sequences(data['Volume'].values, window=window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects 3D input

# Train/test split (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define model
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict and invert scaling
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.legend()
plt.title("LSTM Prediction vs Actual Volume")
plt.xlabel("Time Step")
plt.ylabel("Vehicle Count")
plt.tight_layout()
plt.show()
