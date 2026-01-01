import pandas as pd
import matplotlib.pyplot as plt

# 1. Define Column Names (The raw file doesn't have them)
# 'id': Engine ID
# 'cycle': The current time cycle (e.g., hour 1, hour 2)
# 'set1'-'set3': Operational settings
# 's1'-'s21': The 21 sensors (Temperature, Pressure, Speed, etc.)
columns = ['id', 'cycle', 'set1', 'set2', 'set3'] + [f's{i}' for i in range(1, 22)]

# 2. Load the Data
# Ensure 'train_FD001.txt' is in the same folder as this script
df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=columns)

print(f"Raw Data Shape: {df.shape}")
print("First 5 rows:")
print(df.head())

# --- CRITICAL STEP: Calculate RUL (Remaining Useful Life) ---
# Logic: If Engine 1 runs for 192 cycles total, then:
# At cycle 1, RUL = 191
# At cycle 2, RUL = 190
# ...
# At cycle 192, RUL = 0 (Failure)

# Get the maximum cycle for each engine (the cycle it failed at)
max_cycles = df.groupby('id')['cycle'].max().reset_index()
max_cycles.columns = ['id', 'max_cycle']

# Merge this back into the original data
df = df.merge(max_cycles, on='id', how='left')

# Calculate RUL: Max Cycle - Current Cycle
df['RUL'] = df['max_cycle'] - df['cycle']

# Drop the 'max_cycle' column as we don't need it for training
df = df.drop(columns=['max_cycle'])

print("\nData with RUL Target:")
print(df[['id', 'cycle', 'RUL']].head())

# 3. Save this prepared data
df.to_csv('processed_train_data.csv', index=False)
print("\nSuccess! Processed data saved to 'processed_train_data.csv'")