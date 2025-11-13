# app.py
# Rainfall Prediction Dashboard + Free Weather Chatbot (HuggingFace Inference API)
# Developed for Streamlit Cloud Deployment
# Author: Harshit Pant (adjust as needed)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import datetime as dt
import joblib
import time
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Rainfall Prediction Dashboard",
                   page_icon="🌧️",
                   layout="wide")

# ---------------- CONSTANTS & CITY LIST ----------------
def interpret_rainfall(mm):
    """
    Convert rainfall in mm into weather condition + farmer advice.
    """
    if mm < 0.5:
        return (
            "☀️ **Clear / No Rain Expected**",
            "Good time for outdoor activities. Farmers can plan irrigation normally."
        )
    elif mm < 2.5:
        return (
            "🌤️ **Very Light Rain / Drizzle**",
            "Beneficial for crops; mild moisture gain. No risk to harvesting or spraying."
        )
    elif mm < 7.5:
        return (
            "🌦️ **Light Rain**",
            "Good for soil moisture. Farmers can reduce irrigation for the day."
        )
    elif mm < 35:
        return (
            "🌧️ **Moderate Rain**",
            "May affect field activities. Avoid spraying pesticides or fertilizers."
        )
    else:
        return (
            "⛈️ **Heavy Rain / Storm Likely**",
            "Farmers should protect newly sown crops and avoid field activities."
        )

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Surat": (21.1702, 72.8311),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Patna": (25.5941, 85.1376),
    "Vadodara": (22.3072, 73.1812),
    "Ludhiana": (30.9000, 75.8573),
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Amritsar": (31.6340, 74.8723),
    "Ranchi": (23.3441, 85.3096),
    "Guwahati": (26.1445, 91.7362),
    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Coimbatore": (11.0168, 76.9558),
    "Mysuru": (12.2958, 76.6394),
    "Noida": (28.5355, 77.3910),
    "Gurgaon": (28.4595, 77.0266),
    "Ghaziabad": (28.6692, 77.4538),
    "Mohali": (30.7046, 76.7179),
    "Chandigarh": (30.7333, 76.7794)
}

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HISTORICAL_CSV = "Synthetic_Rainfall_Dataset_1100.csv"
MODEL_FILE = "rainfall.h5"
FEATURE_SCALER_FILE = "feature_scaler.pkl"
TARGET_SCALER_FILE = "target_scaler.pkl"

# ---------------- LOAD MODEL & SCALERS ----------------
@st.cache_resource
def load_model_and_scalers():
    model_obj = None
    feat_scaler = None
    targ_scaler = None

    try:
        model_obj = load_model(MODEL_FILE)
    except Exception:
        st.warning("Model file not found or failed to load (rainfall.h5). Predictions will be disabled.")

    try:
        feat_scaler = joblib.load(FEATURE_SCALER_FILE)
    except Exception:
        st.warning("feature_scaler.pkl not found or failed to load.")

    try:
        targ_scaler = joblib.load(TARGET_SCALER_FILE)
    except Exception:
        st.warning("target_scaler.pkl not found or failed to load.")

    return model_obj, feat_scaler, targ_scaler

model, feature_scaler, target_scaler = load_model_and_scalers()

# ---------------- LIVE WEATHER (Open-Meteo) ----------------
def fetch_live_weather(city: str):
    """Fetch live weather using Open-Meteo (no key required). Returns a dict or None."""
    if city not in CITY_COORDS:
        st.error(f"Coordinates for city '{city}' not available.")
        return None
    lat, lon = CITY_COORDS[city]
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true&hourly=rain,relative_humidity_2m,cloudcover,wind_speed_10m"
    )
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        if "current_weather" not in data:
            return None
        current = data["current_weather"]
        humidity = 60
        cloud = 40
        if "hourly" in data:
            hourly = data["hourly"]
            if "relative_humidity_2m" in hourly and len(hourly["relative_humidity_2m"]) > 0:
                humidity = hourly["relative_humidity_2m"][0]
            if "cloudcover" in hourly and len(hourly["cloudcover"]) > 0:
                cloud = hourly["cloudcover"][0]
        return {
            "date": dt.datetime.now(),
            "temparature": float(current.get("temperature", np.nan)),
            "humidity": float(humidity),
            "windspeed": float(current.get("windspeed", 0.0)),
            "winddirection": float(current.get("winddirection", 0.0)),
            "cloud": float(cloud),
        }
    except Exception as e:
        st.error(f"Error fetching weather: {e}")
        return None

# ---------------- PREDICTION (auto feature alignment) ----------------
def predict_rainfall(live_data: dict):
    """Predict rainfall using the loaded model and scalers; returns float or None."""
    if model is None or feature_scaler is None or target_scaler is None:
        return None

    df_live = pd.DataFrame([live_data])
    now = dt.datetime.now()
    df_live["dayofweek"] = now.weekday()
    df_live["month"] = now.month

    try:
        expected_features = list(feature_scaler.feature_names_in_)
    except Exception:
        expected_features = list(df_live.select_dtypes(include=[np.number]).columns)

    for feat in expected_features:
        if feat not in df_live.columns:
            lf = feat.lower()
            if "humidity" in lf:
                df_live[feat] = df_live.get("humidity", 0)
            elif "cloud" in lf:
                df_live[feat] = df_live.get("cloud", 0)
            elif "wind" in lf:
                df_live[feat] = df_live.get("windspeed", 0)
            elif "temp" in lf or "tempar" in lf or "temperature" in lf:
                df_live[feat] = df_live.get("temparature", df_live.get("temperature", 0))
            elif "day" in lf:
                df_live[feat] = df_live.get("dayofweek", 0)
            elif "month" in lf:
                df_live[feat] = df_live.get("month", 0)
            else:
                df_live[feat] = 0

    X_live = df_live[expected_features].astype(float)
    X_scaled = feature_scaler.transform(X_live)
    X_scaled = np.array(X_scaled).reshape((1, X_scaled.shape[1], 1))

    y_pred = model.predict(X_scaled)
    y_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    return float(y_inv[0])

# ---------------- LOAD HISTORICAL DATA ----------------
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv(HISTORICAL_CSV)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)
            df.set_index("date", inplace=True)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

pdf = load_historical_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Dashboard Controls")

city_selected = st.sidebar.selectbox(
    "Select City (India)",
    options=list(CITY_COORDS.keys()),
    index=list(CITY_COORDS.keys()).index("Mohali") if "Mohali" in CITY_COORDS else 0
)

auto_refresh = st.sidebar.checkbox("Auto-refresh predictions each visit (toggle)", value=False)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds) - manual refresh button also available", 10, 600, 60)
st.sidebar.markdown("---")
st.sidebar.info("Live weather via Open-Meteo (no API key required). Chatbot uses HuggingFace Inference API token in Streamlit Secrets.")

# ---------------- MAIN UI ----------------
st.title("🌧️ Rainfall Prediction Dashboard using LSTM")
st.markdown("Interactive dashboard: live weather + rainfall prediction + free weather chatbot.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Predictions", "📊 Model Evaluation", "📉 Historical Trends", "💬 Weather Chatbot"])

# ---------------- TAB 1: Live Predictions ----------------
with tab1:
    st.subheader("Live Weather & Predicted Rainfall")

    # Initialize history dataframe
    if "history" not in st.session_state:
        st.session_state["history"] = pd.DataFrame(
            columns=["date", "predicted_rainfall", "humidity", "temparature", "windspeed", "city"]
        )

    # --- Auto-refresh handling ---
    if auto_refresh:
        st.experimental_rerun()

    # --- Fetch Live Weather ---
    live_data = fetch_live_weather(city_selected)

    if live_data:
        rainfall_pred = predict_rainfall(live_data)
      # Interpret predicted rainfall
weather_text, farmer_advice = interpret_rainfall(rainfall_pred)

# Display interpretation
st.markdown(f"### {weather_text}")
st.info(f"**Farmer Guidance:** {farmer_advice}")

new_entry = {
            "date": live_data["date"],
            "predicted_rainfall": rainfall_pred,
            "humidity": live_data["humidity"],
            "temparature": live_data["temparature"],
            "windspeed": live_data["windspeed"],
            "city": city_selected
        }

        # Update session history
st.session_state["history"] = pd.concat(
[st.session_state["history"], pd.DataFrame([new_entry])],
ignore_index=True
        )

        # --- METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("🌧️ Predicted Rainfall (mm)", f"{rainfall_pred:.2f}")
col2.metric("💧 Humidity (%)", f"{live_data['humidity']}")
col3.metric("🌡️ Temperature (°C)", f"{live_data['temparature']:.1f}")

# --- Chart: Last N Predictions ---
hist_df = st.session_state["history"].tail(30)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_df["date"],
    y=hist_df["predicted_rainfall"],
    mode="lines+markers",
    name="Predicted Rainfall"
))
fig.update_layout(
    title="Predicted Rainfall Over Time",
    xaxis_title="Timestamp",
    yaxis_title="Rainfall (mm)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# --- Recent Table ---
st.dataframe(
        st.session_state["history"]
        .sort_values(by="date", ascending=False)
        .head(10)
    )
else:
    st.warning("Could not fetch live weather at the moment. Try again later.")

# Manual refresh button
if st.button("🔄 Refresh Now"):
    st.experimental_rerun()

# ---------------- TAB 2: Model Evaluation ----------------
with tab2:
    st.subheader("Model Evaluation Metrics (from training)")
    mae, mse, r2 = 0.312, 0.248, 0.917
    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAE", f"{mae:.3f}")
    col2.metric("📈 MSE", f"{mse:.3f}")
    col3.metric("🔢 R² Score", f"{r2:.3f}")
    st.info("Placeholder values — replace with your model evaluation metrics if available.")

# ---------------- TAB 3: Historical Trends ----------------
with tab3:
    st.subheader("Historical Weather / Rainfall Trends")
    if not pdf.empty and "rainfall" in pdf.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pdf.index,
            y=pdf["rainfall"],
            mode="lines",
            name="Actual Rainfall"
        ))
        fig.update_layout(
            title="Historical Rainfall Trend",
            xaxis_title="Date",
            yaxis_title="Rainfall (mm)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Correlation (sample)")
        st.dataframe(pdf.corr().round(2))
    else:
        st.info("Historical dataset not available in repo (Synthetic_Rainfall_Dataset_1100.csv).")


# ---------------- TAB 4: Free HuggingFace Chatbot ----------------
def get_hf_api_key():
    try:
        return st.secrets["hf"]["api_key"]
    except Exception:
        return None

HF_API_KEY = get_hf_api_key()
headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

def ask_hf_chatbot(prompt_text: str, max_tokens: int = 120):
    """Call HuggingFace Inference API (google/flan-t5-small)."""
    if HF_API_KEY is None:
        return "Chatbot unavailable: add your HuggingFace token to Streamlit Secrets under [hf] api_key."
    payload = {"inputs": prompt_text}
    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
            return out[0]["generated_text"]
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"]
        # fallback
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            # often the result might be [{ 'generated_text': '...' }]
            for item in out:
                if "generated_text" in item:
                    return item["generated_text"]
        return str(out)
    except Exception as e:
        return f"Chatbot error: {e}"

with tab4:
    st.subheader("💬 Free Weather Chatbot (HuggingFace Inference)")
    st.markdown("Ask weather-related questions. The bot will use the selected city context.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask anything about the weather (example: Will it rain today?)", key="chat_input")
    if st.button("Send"):
        if user_query and user_query.strip():
            st.session_state.chat_history.append(("You", user_query.strip()))
            prompt = (
                f"You are a helpful weather expert. The user is asking about {city_selected}. "
                f"Question: {user_query}. Answer clearly and concisely, focusing on weather and actionable advice."
            )
            reply = ask_hf_chatbot(prompt)
            st.session_state.chat_history.append(("Bot", reply))
        else:
            st.warning("Please type a question before pressing Send.")

    # Display last 30 messages
    for sender, text in st.session_state.chat_history[-30:]:
        if sender == "You":
            st.markdown(f"**🧑‍🌾 You:** {text}")
        else:
            st.markdown(f"**🤖 Bot:** {text}")

    if st.button("🔄 Clear Chat"):
        st.session_state.chat_history = []

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Harshit Pant • Live weather: Open-Meteo • Model: LSTM")

