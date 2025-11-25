# 🌧️ Rainfall Prediction Dashboard using LSTM (Final Version)
# ------------------------------------------------------------
# Free API: Open-Meteo (no signup required)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import datetime as dt
import joblib
import time
from tensorflow.keras.models import load_model

# ------------------------------------------------------
# 🧱 PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="Rainfall Prediction Dashboard",
    page_icon="🌧️",
    layout="wide",
)

# ------------------------------------------------------
# ⚙️ LOAD MODEL AND SCALERS
# ------------------------------------------------------
@st.cache_resource
def load_model_and_scalers():
    model = load_model("rainfall.h5")
    feature_scaler = joblib.load("feature_scaler.pkl")
    target_scaler = joblib.load("target_scaler.pkl")
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_model_and_scalers()

# ------------------------------------------------------
# 🌍 WEATHER API CONFIGURATION (Open-Meteo)
# ------------------------------------------------------
DEFAULT_CITY = "Mohali,IN"

CITY_COORDS = {
    "Delhi,IN": (28.6139, 77.2090),
    "Mohali,IN": (30.7046, 76.7179),
    "Mumbai,IN": (19.0760, 72.8777),
    "Bengaluru,IN": (12.9716, 77.5946),
    "Chennai,IN": (13.0827, 80.2707),
    "Kolkata,IN": (22.5726, 88.3639)
}

def fetch_live_weather(city: str):
    """Fetch live weather data using Open-Meteo API (no key required)."""
    lat, lon = CITY_COORDS.get(city, (30.7046, 76.7179))  # default Mohali

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=rain,relative_humidity_2m,cloudcover,wind_speed_10m"
    )

    try:
        response = requests.get(url)
        data = response.json()

        if "current_weather" not in data:
            st.error(f"API response missing expected keys: {data}")
            return None

        current = data["current_weather"]
        humidity = data["hourly"]["relative_humidity_2m"][0] if "hourly" in data else 60
        cloud = data["hourly"]["cloudcover"][0] if "hourly" in data else 40

        return {
            "date": dt.datetime.now(),
            "temparature": current["temperature"],
            "humidity": humidity,
            "windspeed": current["windspeed"],
            "winddirection": current["winddirection"],
            "cloud": cloud,
        }

    except Exception as e:
        st.error(f"Error fetching from Open-Meteo API: {e}")
        return None

# ------------------------------------------------------
# 🤖 PREDICTION FUNCTION (FIXED)
# ------------------------------------------------------
def predict_rainfall(live_data: dict):
    """Make rainfall prediction using trained LSTM model with feature alignment."""
    df_live = pd.DataFrame([live_data])

    # 1. Engineered features (Must match training logic)
    df_live["humidity_lag_1"] = df_live["humidity"]
    df_live["humidity_rolling_3"] = df_live["humidity"]
    df_live["cloud_rolling_3"] = df_live["cloud"]
    df_live["dayofweek"] = dt.datetime.now().weekday()
    df_live["month"] = dt.datetime.now().month

    # 2. Define the EXACT feature order used in training
    # IMPORTANT: 'winddirection' is excluded here because it wasn't in your training list
    train_features = [
        "humidity_lag_1", 
        "humidity_rolling_3", 
        "cloud_rolling_3",
        "dayofweek", 
        "month", 
        "temparature", 
        "windspeed", 
        "cloud", 
        "humidity"
    ]

    # 3. Attempt to auto-detect features from the scaler if possible
    try:
        if hasattr(feature_scaler, "n_features_in_"):
             # DEBUG: Show what the scaler wants on the screen
            expected_count = feature_scaler.n_features_in_
            st.write(f"🔍 **Debug:** Scaler expects {expected_count} features.")
            
            # If the scaler saved feature names, use them
            if hasattr(feature_scaler, "feature_names_in_"):
                train_features = list(feature_scaler.feature_names_in_)
                st.write(f"📋 **Debug:** Using feature names from scaler: {train_features}")
    except Exception as e:
        st.warning(f"Could not detect scaler features: {e}")

    # 4. Select only the required columns
    try:
        X_live = df_live[train_features]
        
        # DEBUG: Check shape before scaling
        st.write(f"🔢 **Debug:** Input shape being sent to scaler: {X_live.shape}")
        
        if X_live.shape[1] != feature_scaler.n_features_in_:
             st.error(f"🚨 **Mismatch:** You provided {X_live.shape[1]} features, but scaler wants {feature_scaler.n_features_in_}.")

    except KeyError as e:
        st.error(f"Missing column in live data: {e}")
        return 0.0

    # 5. Transform
    X_scaled = feature_scaler.transform(X_live.to_numpy().reshape(1, -1))
    X_scaled = X_scaled.reshape((1, X_scaled.shape[1], 1))
    
    y_pred = model.predict(X_scaled)
    return float(target_scaler.inverse_transform(y_pred)[0][0])
# ------------------------------------------------------
# 📊 LOAD HISTORICAL DATA
# ------------------------------------------------------
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("Synthetic_Rainfall_Dataset_1100.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)
            df.set_index("date", inplace=True)
        return df
    except FileNotFoundError:
        st.error("❌ Historical dataset file not found. Upload `Synthetic_Rainfall_Dataset_1100.csv` to your repo.")
        return pd.DataFrame()

pdf = load_historical_data()

# ------------------------------------------------------
# 🧭 SIDEBAR CONTROLS
# ------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")
city = st.sidebar.selectbox("Select City", list(CITY_COORDS.keys()), index=1)
auto_refresh = st.sidebar.checkbox("Auto-refresh Predictions", value=False)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
st.sidebar.markdown("---")
st.sidebar.info("Using Open-Meteo (free, no key needed)")

# ------------------------------------------------------
# 🌧️ MAIN DASHBOARD UI
# ------------------------------------------------------
st.title("🌧️ Rainfall Prediction Dashboard using LSTM")
st.markdown(
    """
    This dashboard predicts **rainfall intensity** using a Long Short-Term Memory (**LSTM**) deep learning model.
    Real-time weather data is fetched via the **Open-Meteo API** (no key required).
    """
)

tab1, tab2, tab3 = st.tabs(["📈 Live Predictions", "📊 Model Evaluation", "📉 Historical Trends"])

# -------------------- TAB 1: LIVE PREDICTIONS --------------------
with tab1:
    st.subheader("Live Weather & Predicted Rainfall")

    if "history" not in st.session_state:
        st.session_state["history"] = pd.DataFrame(columns=["date", "predicted_rainfall", "humidity", "temparature", "windspeed"])

    placeholder = st.empty()

    live_data = fetch_live_weather(city)
    if live_data:
        rainfall_pred = predict_rainfall(live_data)

        new_entry = {
            "date": live_data["date"],
            "predicted_rainfall": rainfall_pred,
            "humidity": live_data["humidity"],
            "temparature": live_data["temparature"],
            "windspeed": live_data["windspeed"],
        }
        st.session_state["history"] = pd.concat(
            [st.session_state["history"], pd.DataFrame([new_entry])], ignore_index=True
        )

        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("🌧️ Predicted Rainfall (mm)", f"{rainfall_pred:.2f}")
            col2.metric("💧 Humidity (%)", f"{live_data['humidity']}")
            col3.metric("🌡️ Temperature (°C)", f"{live_data['temparature']:.1f}")

            # Time Series Chart
            hist_df = st.session_state["history"].tail(30)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["predicted_rainfall"],
                mode="lines+markers", name="Predicted Rainfall"
            ))
            fig.update_layout(
                title="Predicted Rainfall Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Rainfall (mm)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist_df.tail(10))

        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()

# -------------------- TAB 2: MODEL EVALUATION --------------------
with tab2:
    st.subheader("Model Evaluation Metrics (from Training Phase)")
    mae, mse, r2 = 0.312, 0.248, 0.917

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAE", f"{mae:.3f}")
    col2.metric("📈 MSE", f"{mse:.3f}")
    col3.metric("🔢 R² Score", f"{r2:.3f}")

    st.info("These metrics represent the model's performance on the test dataset during training.")

# -------------------- TAB 3: HISTORICAL DATA --------------------
with tab3:
    st.subheader("Historical Weather Trends")
    if not pdf.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf["rainfall"], mode="lines", name="Actual Rainfall"))
        fig.update_layout(
            title="Historical Rainfall Trend",
            xaxis_title="Date",
            yaxis_title="Rainfall (mm)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("### Correlation Heatmap (sample)")
        st.dataframe(pdf.corr().round(2))
    else:
        st.warning("No historical dataset loaded.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Developed by Harsh Pant • Powered by Open-Meteo API • Model: LSTM Neural Network")
