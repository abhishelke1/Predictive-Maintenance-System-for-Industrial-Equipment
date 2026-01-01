import streamlit as st
import pandas as pd
import requests
import time
import plotly.graph_objects as go

# Setup Page Layout
st.set_page_config(page_title="Industrial IoT Monitor", layout="wide")
st.title("🏭 Real-Time Predictive Maintenance Dashboard")

# 1. Load Data & Re-Create Features
@st.cache_data
def load_data():
    # Load the raw data
    df = pd.read_csv('processed_train_data.csv')
    
    # --- FIX: Generate the Missing Features (Same as Step 2) ---
    sensor_cols = ['s7', 's12', 's21']
    
    # Calculate Rolling Mean and Std Dev again so the dashboard has them
    for col in sensor_cols:
        df[f'{col}_mean'] = df.groupby('id')[col].transform(lambda x: x.rolling(window=5).mean())
        df[f'{col}_std'] = df.groupby('id')[col].transform(lambda x: x.rolling(window=5).std())
        
    # Fill any empty values created by the rolling window
    df.fillna(method='bfill', inplace=True)
    
    # Select Engine 2 for the simulation
    engine_data = df[df['id'] == 2]
    return engine_data

data = load_data()

# 2. Sidebar Controls
st.sidebar.header("Control Panel")
start_button = st.sidebar.button("🔴 Start Live Simulation")
speed = st.sidebar.slider("Simulation Speed (sec delay)", 0.1, 2.0, 0.5)

# 3. Layout: Metrics on top, Chart on bottom
col1, col2, col3 = st.columns(3)
with col1:
    metric_rul = st.empty()
with col2:
    metric_status = st.empty()
with col3:
    metric_sensor = st.empty()

# Placeholder for the Live Chart
chart_placeholder = st.empty()

# 4. Simulation Logic
if start_button:
    # Lists to store history for plotting
    history_rul = []
    history_time = []
    
    # Iterate through the engine's life cycle
    for i, row in data.iterrows():
        
        # Prepare the payload for the API
        payload = {
            "s7": row['s7'],
            "s12": row['s12'],
            "s21": row['s21'],
            "s7_mean": row['s7_mean'],
            "s7_std": row['s7_std'],
            "s12_mean": row['s12_mean'],
            "s12_std": row['s12_std'],
            "s21_mean": row['s21_mean'],
            "s21_std": row['s21_std']
        }
        
        try:
            # SEND DATA TO API
            # Ensure the port matches your Uvicorn window (usually 8000)
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                pred_rul = result['predicted_RUL']
                status = result['status']
                
                # Update Metrics
                metric_rul.metric("Predicted RUL (Cycles)", f"{pred_rul:.1f}")
                
                if status == "Critical":
                    metric_status.error(f"Status: {status}")
                else:
                    metric_status.success(f"Status: {status}")
                    
                metric_sensor.metric("Sensor 7 (Pressure)", f"{row['s7']:.2f}")

                # Update Chart
                history_rul.append(pred_rul)
                history_time.append(i)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history_time, y=history_rul, mode='lines', name='RUL Prediction', line=dict(color='blue')))
                
                # Add a Red Line for "Failure Threshold"
                fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
                
                fig.update_layout(
                    title="Live RUL Prediction Over Time", 
                    xaxis_title="Time Cycle", 
                    yaxis_title="Remaining Useful Life",
                    height=400
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"API Error: {response.text}")
                
        except Exception as e:
            st.error(f"Connection Error: Is the API running? {e}")
            break
            
        time.sleep(speed)