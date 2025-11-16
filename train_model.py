# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data
df = pd.read_csv('weatherHistory.csv')
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df = df.set_index('Formatted Date').sort_index()

# Select only Temperature and Humidity
features = ['Temperature (C)', 'Humidity']
data = df[features].dropna()

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences: 10 timesteps → predict next [Temp, Humidity]
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])  # next row (both features)
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(10, 2)))
model.add(LSTM(50))
model.add(Dense(2))  # Output: [Temperature, Humidity]
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model and scaler
model.save('model/model.h5')
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model trained: predicts next [Temperature, Humidity]")