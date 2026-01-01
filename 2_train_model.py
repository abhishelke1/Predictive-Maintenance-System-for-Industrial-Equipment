import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle # To save the trained model

# 1. Load the Processed Data
df = pd.read_csv('processed_train_data.csv')

# 2. Feature Engineering (The Secret Sauce)
# We focus on specific sensors that change the most before failure
# (Sensors 7, 12, 21 usually show clear degradation trends in this dataset)
sensor_cols = ['s7', 's12', 's21']

print("Generating Rolling Averages (Smoothing sensor noise)...")

# Calculate Rolling Mean and Standard Deviation for these sensors
# Window=5 means we look at the last 5 cycles to calculate the average
for col in sensor_cols:
    df[f'{col}_mean'] = df.groupby('id')[col].transform(lambda x: x.rolling(window=5).mean())
    df[f'{col}_std'] = df.groupby('id')[col].transform(lambda x: x.rolling(window=5).std())

# Rolling functions create NaNs (empty values) for the first 5 rows. Fill them.
df.fillna(method='bfill', inplace=True)

# 3. Prepare Training Data
# We drop 'id' (AI shouldn't memorize engine IDs) and 'RUL' (Target) from input
features = sensor_cols + [f'{c}_mean' for c in sensor_cols] + [f'{c}_std' for c in sensor_cols]
X = df[features]
y = df['RUL']

# Split Data: Train on first 80 engines, Test on last 20
# We split by ID, not randomly. We don't want the AI to see "future" data of the same engine.
train_mask = df['id'] <= 80
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

# 4. Train the Model (XGBoost)
print("\nTraining XGBoost Model...")
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\nModel Performance:")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
# RMSE of 40 means our prediction is usually off by +/- 40 cycles.
# (In real life, we would tune this to get it lower, usually under 20).

# 6. Save the Model
# We save it so our API can load it later without retraining
with open('rul_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nSuccess! Model saved to 'rul_model.pkl'")

# Optional: Show a prediction example
print("\nExample Prediction:")
print(f"True RUL: {y_test.iloc[0]}, Predicted RUL: {predictions[0]:.2f}")