# app.py
# Rainfall Prediction Dashboard + Free Weather Chatbot (HuggingFace Inference API)
# Developed by Harshit Pant (adjust author line as you want)

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
                   page_icon="🌧️", layout="wide")

# ---------------- HELPERS & CONSTANTS ----------------

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


# ---------------- LOAD MODEL & SCALERS (cached) ----------------
@st.cache_resource
def load_model_and_scalers():
    """Load model and scalers if available. Return (model, feature_scaler, target_scaler)."""
    model_obj = None
    feat_scaler = None
    targ_scaler = None

    # load model
    try:
        model_obj = load_model("rainfall.h5")
    except Exception as e:
        st.warning("Model file not found or failed to load (rainfall.h5). Prediction will be disabled.")
        model_obj = None

    # load scalers
    try:
        feat_scaler = joblib.load("feature_scaler.pkl")
    except Exception:
        st.warning("feature_scaler.pkl not found or failed to load.")
        feat_scaler = None

    try:
        targ_scaler = joblib.load("target_scaler.pkl")
    except Exception:
        st.warning("target_scaler.pkl not found or failed to load.")
        targ_scaler = None

    return model_obj, feat_scaler, targ_scaler


model, feature_scaler, target_scaler = load_model_and_scalers()


# ---------------- LIVE WEATHER (Open-Meteo) ----------------
def fetch_live_weather(city: str):
    """Fetch live weather using Open-Meteo (no key required). Returns dict or None."""
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
            # sometimes hourly keys exist but current missing, handle gracefully
            return None
        current = data["current_weather"]
        # Best-effort humidity/cloud extraction
        humidity = 60
        cloud = 40
        if "hourly" in data:
            # hourly arrays align with 'time' index; pick first if present
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
    """
    Make rainfall prediction using trained LSTM model with feature name auto-alignment.
    Returns predicted rainfall (mm) as float. Requires model & scalers loaded.
    """
    if model is None or feature_scaler is None or target_scaler is None:
        # missing artifacts
        return None

    df_live = pd.DataFrame([live_data])

    # add basic time features in case model used them
    now = dt.datetime.now()
    df_live["dayofweek"] = now.weekday()
    df_live["month"] = now.month

    # Try to align with the scaler's expected feature names
    try:
        expected_features = list(feature_scaler.feature_names_in_)
    except Exception:
        expected_features = list(df_live.select_dtypes(include=[np.number]).columns)

    # Create missing features with reasonable defaults based on available live_data
    for feat in expected_features:
        if feat not in df_live.columns:
            feat_lower = feat.lower()
            if "humidity" in feat_lower:
                df_live[feat] = df_live.get("humidity", 0)
            elif "cloud" in feat_lower:
                df_live[feat] = df_live.get("cloud", 0)
            elif "wind" in feat_lower:
                df_live[feat] = df_live.get("windspeed", 0)
            elif "temp" in feat_lower or "tempar" in feat_lower or "temperature" in feat_lower:
                df_live[feat] = df_live.get("temparature", df_live.get("temperature", 0))
            elif "day" in feat_lower:
                df_live[feat] = df_live.get("dayofweek", 0)
            elif "month" in feat_lower:
                df_live[feat] = df_live.get("month", 0)
            else:
                # rolling or lag features - safe default 0
                df_live[feat] = 0

    # select exactly expected features in order
    X_live = df_live[expected_features].astype(float)

    # scale and reshape for LSTM (samples, timesteps, features) - we used features as timesteps before
    X_scaled = feature_scaler.transform(X_live)
    # If scaler returns 1D for single sample, ensure correct shape
    X_scaled = np.array(X_scaled).reshape((1, X_scaled.shape[1], 1))

    # predict and inverse transform
    y_pred = model.predict(X_scaled)
    y_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    return float(y_inv[0])


# ---------------- LOAD HISTORICAL DATA ----------------
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
        # not fatal: just return empty dataframe
        return pd.DataFrame()


pdf = load_historical_data()


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙️ Dashboard Controls")

city_selected = st.sidebar.selectbox(
    "Select City (India)",
    options=list(CITY_COORDS.keys()),
    index=list(CITY_COORDS.keys()).index("Mohali") if "Mohali" in CITY_COORDS else 0
)

auto_refresh = st.sidebar.checkbox("Auto-refresh predictions each visit (toggle)", value=False)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds) - (manual refresh below)", 10, 600, 60)
st.sidebar.markdown("---")
st.sidebar.info("Live weather via Open-Meteo (no API key required). Chatbot uses HuggingFace Inference API token stored in Streamlit Secrets.")


# ---------------- MAIN UI ----------------
st.title("🌧️ Rainfall Prediction Dashboard using LSTM")
st.markdown("This dashboard predicts rainfall intensity in real-time using a Long Short-Term Memory (LSTM) model.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Predictions", "📊 Model Evaluation", "📉 Historical Trends", "💬 Weather Chatbot"])

# ---------------- TAB 1: Live Predictions ----------------
with tab1:
    st.subheader("Live Weather & Predicted Rainfall")

    if "history" not in st.session_state:
        st.session_state["history"] = pd.DataFrame(columns=["date", "predicted_rainfall", "humidity", "temparature", "windspeed", "city"])

    # Single fetch per app run (safer for Streamlit Cloud). Use button to refresh immediately.
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh Now"):
            # force a rerun: simple trick is to write to session state then rerun
            st.session_state["_refresh_ts"] = time.time()
            st.experimental_rerun()

    with col_info:
        st.write(f"Showing data for: **{city_selected}** • Auto-refresh (toggle) = {auto_refresh}")

    # fetch live
    live_data = fetch_live_weather(city_selected)
    if live_data:
        rainfall_pred = predict_rainfall(live_data)
        if rainfall_pred is None:
            st.warning("Prediction unavailable (model or scalers missing). Showing live weather only.")
        else:
            st.success(f"Predicted rainfall: {rainfall_pred:.3f} mm")

        # append to history
        new_entry = {
            "date": live_data["date"],
            "predicted_rainfall": rainfall_pred if rainfall_pred is not None else np.nan,
            "humidity": live_data["humidity"],
            "temparature": live_data["temparature"],
            "windspeed": live_data["windspeed"],
            "city": city_selected
        }
        st.session_state["history"] = pd.concat([st.session_state["history"], pd.DataFrame([new_entry])], ignore_index=True)

        # top metrics
        col1, col2, col3 = st.columns(3)
        if rainfall_pred is not None:
            col1.metric("🌧️ Predicted Rainfall (mm)", f"{rainfall_pred:.2f}")
        else:
            col1.metric("🌧️ Predicted Rainfall (mm)", "N/A")
        col2.metric("💧 Humidity (%)", f"{live_data['humidity']}")
        col3.metric("🌡️ Temperature (°C)", f"{live_data['temparature']:.1f}")

        # chart
        hist_df = st.session_state["history"].tail(30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["predicted_rainfall"], mode="lines+markers", name="Predicted Rainfall"))
        fig.update_layout(
    title="Predicted Rainfall Over Time",
    xaxis_title="Timestamp",
    yaxis_title="Rainfall (mm)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)


        st.dataframe(st.session_state["history"].sort_values(by="date", ascending=False).head(10))
    else:
        st.warning("Could not fetch live weather for the selected city. Check network or city coordinates.")


# ---------------- TAB 2: Model Evaluation ----------------
with tab2:
    st.subheader("Model Evaluation Metrics (from training)")
    mae, mse, r2 = 0.312, 0.248, 0.917
    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAE", f"{mae:.3f}")
    col2.metric("📈 MSE", f"{mse:.3f}")
    col3.metric("🔢 R² Score", f"{r2:.3f}")
    st.info("These are placeholder values — replace with your model evaluation metrics if available.")


# ---------------- TAB 3: Historical Trends ----------------
with tab3:
    st.subheader("Historical Weather / Rainfall Trends")
    if not pdf.empty:
        fig = go.Figure()
        if "rainfall" in pdf.columns:
            fig.add_trace(go.Scatter(x=pdf.index, y=pdf["rainfall"], mode="lines", name="Actual Rainfall"))
            fig.update_layout(title="Historical Rainfall Trend", xaxis_title="Date", yaxis_title="Rainfall (mm)", template="plotly_white", width='stretch')
            st.plotly_chart(fig, use_container_width=True)
        st.write("### Correlation (sample)")
        st.dataframe(pdf.corr().round(2))
    else:
        st.info("Historical dataset not available in repo (Synthetic_Rainfall_Dataset_1100.csv).")


# ---------------- TAB 4: Free HF Chatbot ----------------
# Uses HuggingFace Inference API (no heavy model downloads)
def get_hf_api_key():
    try:
        return st.secrets["hf"]["api_key"]
    except Exception:
        return None

HF_API_KEY = get_hf_api_key()
headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

def ask_hf_chatbot(prompt_text):
    """Call HuggingFace Inference API (google/flan-t5-small)."""
    if HF_API_KEY is None:
        return "Chatbot unavailable: add your HuggingFace token to Streamlit Secrets under [hf] api_key."
    payload = {"inputs": prompt_text}
    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        out = resp.json()
        # HF sometimes returns a list with dict containing 'generated_text'
        if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
            return out[0]["generated_text"]
        # some models return dict with 'generated_text' directly
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"]
        # otherwise stringify response
        return str(out)
    except Exception as e:
        return f"Chatbot error: {e}"

# ---------------- TAB 4: Free HF Chatbot ----------------
with tab4:
    st.subheader("💬 Free Weather Chatbot (HuggingFace Inference)")
    st.markdown("Ask weather-related questions. The bot will use the selected city context.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # user input
    user_query = st.text_input("Ask anything about the weather:")

    # when user submits
    if st.button("Send") and user_query:
        st.session_state.chat_history.append(("You", user_query))

        # Add city context
        prompt = (
            f"You are a helpful weather expert. The user is asking about {city_selected}. "
            f"Question: {user_query}. Answer clearly and concisely."
        )

        reply = ask_hf_chatbot(prompt)
        st.session_state.chat_history.append(("Bot", reply))

    # Display chat
    for sender, text in st.session_state.chat_history[-30:]:
        if sender == "You":
            st.markdown(f"**🧑‍🌾 You:** {text}")
        else:
            st.markdown(f"**🤖 Bot:** {text}")

    if st.button("🔄 Clear Chat"):
        st.session_state.chat_history = []

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Harshit Pant • Powered by Open-Meteo (live weather) • Model: LSTM (local).")
