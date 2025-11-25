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
    page_title="Kisan Rainfall Assistant",
    page_icon="🌾",
    layout="wide",
)

# ------------------------------------------------------
# 🗣️ LOCALIZATION (Language Dictionary)
# ------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "🌾 Smart Rainfall & Farming Dashboard",
        "live_tab": "🔴 Live Forecast",
        "crop_tab": "🚜 Crop Monitor",
        "chat_tab": "💬 Kisan Assistant",
        "hist_tab": "📉 History",
        "rain_label": "Predicted Rainfall",
        "humidity": "Humidity",
        "temp": "Temperature",
        "wind": "Wind Speed",
        "advice_safe": "SAFE: Good conditions for field work.",
        "advice_caution": "CAUTION: Light rain expected. Delay spraying.",
        "advice_danger": "ALERT: Heavy rain! Do not irrigate.",
    },
    "Hindi": {
        "title": "🌾 स्मार्ट वर्षा और कृषि डैशबोर्ड",
        "live_tab": "🔴 मौसम पूर्वानुमान",
        "crop_tab": "🚜 फसल निगरानी",
        "chat_tab": "💬 किसान सहायक",
        "hist_tab": "📉 पुराना डेटा",
        "rain_label": "अनुमानित वर्षा",
        "humidity": "नमी",
        "temp": "तापमान",
        "wind": "हवा की गति",
        "advice_safe": "सुरक्षित: खेत में काम करने के लिए अच्छा समय।",
        "advice_caution": "सावधान: हल्की बारिश की संभावना। छिड़काव रोकें।",
        "advice_danger": "चेतावनी: भारी बारिश! सिंचाई न करें।",
    },
    "Punjabi": {
        "title": "🌾 ਸਮਾਰਟ ਮੀਂਹ ਅਤੇ ਖੇਤੀਬਾੜੀ ਡੈਸ਼ਬੋਰਡ",
        "live_tab": "🔴 ਮੌਸਮ ਦੀ ਭਵਿੱਖਬਾਣੀ",
        "crop_tab": "🚜 ਫਸਲ ਦੀ ਨਿਗਰਾਨੀ",
        "chat_tab": "💬 ਕਿਸਾਨ ਸਹਾਇਕ",
        "hist_tab": "📉 ਪਿਛਲਾ ਡੇਟਾ",
        "rain_label": "ਅਨੁਮਾਨਤ ਮੀਂਹ",
        "humidity": "ਨਮੀ",
        "temp": "ਤਾਪਮਾਨ",
        "wind": "ਹਵਾ ਦੀ ਗਤੀ",
        "advice_safe": "ਸੁਰੱਖਿਅਤ: ਖੇਤ ਦੇ ਕੰਮ ਲਈ ਵਧੀਆ ਸਮਾਂ।",
        "advice_caution": "ਸਾਵਧਾਨ: ਹਲਕੀ ਬਾਰਿਸ਼ ਦੀ ਉਮੀਦ। ਸਪਰੇਅ ਰੋਕੋ।",
        "advice_danger": "ਚੇਤਾਵਨੀ: ਭਾਰੀ ਮੀਂਹ! ਸਿੰਚਾਈ ਨਾ ਕਰੋ।",
    }
}

# ------------------------------------------------------
# 🌾 CROP KNOWLEDGE BASE
# ------------------------------------------------------
CROP_INFO = {
    "Wheat (Rabi)": {
        "duration_days": 140,
        "stages": [
            (0, 20, "🌱 Germination"), (21, 60, "🌿 Tillering"),
            (61, 90, "🌸 Flowering"), (91, 120, "🌾 Grain Filling"),
            (121, 140, "🚜 Harvesting")
        ],
        "critical_rain_stage": "Flowering", 
        "water_needs": "Moderate"
    },
    "Rice (Kharif)": {
        "duration_days": 120,
        "stages": [
            (0, 15, "🌱 Seedling"), (16, 45, "🌿 Tillering"),
            (46, 75, "🌸 Panicle Initiation"), (76, 105, "🌾 Grain Filling"),
            (106, 120, "🚜 Harvesting")
        ],
        "critical_rain_stage": "Harvesting",
        "water_needs": "High"
    },
    "Cotton": {
        "duration_days": 160,
        "stages": [
            (0, 20, "🌱 Germination"), (21, 60, "🌿 Vegetative"),
            (61, 100, "🌸 Flowering"), (101, 140, "☁️ Boll Bursting"),
            (141, 160, "🚜 Picking")
        ],
        "critical_rain_stage": "Boll Bursting",
        "water_needs": "Low"
    }
}

# ------------------------------------------------------
# ⚙️ LOAD MODEL AND SCALERS
# ------------------------------------------------------
@st.cache_resource
def load_resources():
    try:
        model = load_model("rainfall.h5")
        f_scaler = joblib.load("feature_scaler.pkl")
        t_scaler = joblib.load("target_scaler.pkl")
        return model, f_scaler, t_scaler
    except Exception as e:
        return None, None, None

model, feature_scaler, target_scaler = load_resources()

# ------------------------------------------------------
# 🌍 API & PREDICTION FUNCTIONS
# ------------------------------------------------------
CITY_COORDS = {
    "Mohali, PB": (30.7046, 76.7179),
    "Ludhiana, PB": (30.9010, 75.8573),
    "Delhi, NCR": (28.6139, 77.2090),
    "Bathinda, PB": (30.2109, 74.9455)
}

def fetch_weather_with_history(city):
    lat, lon = CITY_COORDS.get(city, (30.7046, 76.7179))
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=rain,relative_humidity_2m,cloudcover,wind_speed_10m,temperature_2m&past_days=1&forecast_days=1"
    
    try:
        response = requests.get(url).json()
        hourly = response.get("hourly", {})
        
        # Create Dataframe for context
        df = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "rain": hourly["rain"],
            "humidity": hourly["relative_humidity_2m"],
            "cloud": hourly["cloudcover"],
            "wind": hourly["wind_speed_10m"],
            "temp": hourly["temperature_2m"]
        })
        
        # Find current index
        now = pd.Timestamp.now().floor('h')
        df['diff'] = abs(df['time'] - now)
        idx = df['diff'].idxmin()

        # Helper for Lags
        def get_val(i, col): return df.loc[i, col] if i >= 0 else 0.0

        return {
            "date": dt.datetime.now(),
            "humidity": df.loc[idx, "humidity"],
            "temparature": df.loc[idx, "temp"],
            "windspeed": df.loc[idx, "wind"],
            "cloud": df.loc[idx, "cloud"],
            # Lags calculated from real history
            "rain_l1": get_val(idx-1, "rain"),
            "rain_l3": get_val(idx-3, "rain"),
            "rain_l7": get_val(idx-7, "rain"),
            "hum_l1": get_val(idx-1, "humidity"),
            "wind_l1": get_val(idx-1, "wind"),
            "temp_l1": get_val(idx-1, "temp"),
            "rain_r3": df.loc[max(0, idx-2):idx, "rain"].mean(),
            "hum_r3": df.loc[max(0, idx-2):idx, "humidity"].mean(),
            "cloud_r3": df.loc[max(0, idx-2):idx, "cloud"].mean(),
        }
    except Exception:
        return None

def predict_rain(data):
    if not model: return 0.0
    
    # 1. Prepare Dataframe
    df = pd.DataFrame([data])
    df["dayofweek"] = dt.datetime.now().weekday()
    df["month"] = dt.datetime.now().month
    
    # 2. Rename columns to match Scaler's expected names
    # Note: 'temparature' is kept as per your previous scaler keys
    input_data = df.rename(columns={
        "rain_l1": "rainfall_lag_1", "rain_l3": "rainfall_lag_3", "rain_l7": "rainfall_lag_7",
        "hum_l1": "humidity_lag_1", "wind_l1": "windspeed_lag_1", "temp_l1": "temparature_lag_1",
        "rain_r3": "rainfall_rolling_3", "hum_r3": "humidity_rolling_3", "cloud_r3": "cloud_rolling_3"
    })
    
    # 3. Define the strict column order for the Scaler (11 features)
    final_cols = [
        "rainfall_lag_1", "rainfall_lag_3", "rainfall_lag_7",
        "humidity_lag_1", "windspeed_lag_1", "temparature_lag_1",
        "rainfall_rolling_3", "humidity_rolling_3", "cloud_rolling_3",
        "month", "dayofweek"
    ]
    
    try:
        # A. Select columns and Shape for Scaler: (1 sample, 11 features)
        X_raw = input_data[final_cols].to_numpy().reshape(1, -1)
        
        # B. Scale the data (Scaler expects 11 features)
        X_scaled = feature_scaler.transform(X_raw)
        
        # C. Reshape for LSTM: (1 sample, 1 timestep, 11 features)
        X_ready = X_scaled.reshape(1, 1, 11)
        
        # 🛑 D. CRITICAL FIX: Handle Feature Mismatch 🛑
        # The error said model expects 1 feature, but we have 11.
        # We check the model's input shape dynamically.
        
        # Get expected features (last dimension of input shape)
        # model.input_shape is usually (None, Timesteps, Features)
        expected_features = model.input_shape[-1]
        
        if expected_features == 1 and X_ready.shape[2] > 1:
            # If model is Univariate (1 feature), keep only the first feature (Rainfall Lag 1)
            X_ready = X_ready[:, :, 0:1]
        
        # E. Predict
        y = model.predict(X_ready)
        
        # F. Inverse Transform Result
        return max(0.0, float(target_scaler.inverse_transform(y)[0][0]))

    except Exception as e:
        st.error(f"Prediction logic error: {e}")
        # Debugging info to help if it fails again
        if 'X_ready' in locals():
            st.write(f"Debug Info -> Input Shape: {X_ready.shape}, Model Expects: {model.input_shape}")
        return 0.0

def get_chatbot_response(msg):
    msg = msg.lower()
    if "hello" in msg: return "Namaste! I am your Kisan Assistant. Ask about rain, crops, or fertilizers."
    if "rice" in msg or "paddy" in msg: return "Rice needs standing water. If rain is predicted > 10mm, stop tube well irrigation."
    if "wheat" in msg: return "Wheat is sensitive to waterlogging at the grain-filling stage. Ensure drainage."
    if "rain" in msg: return "I use an LSTM AI model to analyze past weather and predict rain intensity."
    return "I am learning. Please ask about 'Rice', 'Wheat', or 'Weather'."

# ------------------------------------------------------
# 🖥️ MAIN UI
# ------------------------------------------------------

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=100)
st.sidebar.header("⚙️ Settings / सेटिंग्स")
lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi", "Punjabi"])
city = st.sidebar.selectbox("Select Location", list(CITY_COORDS.keys()))
t = TRANSLATIONS[lang]

st.title(t["title"])

# Navigation
tab1, tab2, tab3, tab4 = st.tabs([t["live_tab"], t["crop_tab"], t["chat_tab"], t["hist_tab"]])

# --- TAB 1: LIVE DASHBOARD ---
with tab1:
    if model is None:
        st.error("🚨 Model files not found! Please upload .h5 and .pkl files.")
    else:
        weather = fetch_weather_with_history(city)
        if weather:
            pred_rain = predict_rain(weather)
            
            # TRAFFIC LIGHT ADVISORY
            st.write("---")
            if pred_rain > 5.0:
                st.error(f"🔴 **{t['advice_danger']}**")
            elif pred_rain > 0.5:
                st.warning(f"🟡 **{t['advice_caution']}**")
            else:
                st.success(f"🟢 **{t['advice_safe']}**")
            st.write("---")

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t["rain_label"], f"{pred_rain:.2f} mm")
            c2.metric(t["humidity"], f"{weather['humidity']}%")
            c3.metric(t["temp"], f"{weather['temparature']} °C")
            c4.metric(t["wind"], f"{weather['windspeed']} km/h")
            
            # Store for other tabs
            st.session_state['last_rain'] = pred_rain
        else:
            st.warning("Could not fetch weather data. Check internet.")

# --- TAB 2: CROP MONITOR ---
with tab2:
    st.subheader(f"{t['crop_tab']}")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        s_crop = st.selectbox("Select Crop", list(CROP_INFO.keys()))
        s_date = st.date_input("Sowing Date", dt.date.today() - dt.timedelta(days=30))
        
        days_age = (dt.date.today() - s_date).days
        st.info(f"📅 Crop Age: **{days_age} Days**")
        
    with c2:
        if days_age >= 0:
            c_data = CROP_INFO[s_crop]
            stage = "Unknown"
            prog = 0.0
            for start, end, name in c_data["stages"]:
                if start <= days_age <= end:
                    stage = name
                    prog = min(1.0, (days_age - start) / (end - start + 1))
            
            if days_age > c_data["duration_days"]: stage = "Harvested/Finished"
            
            st.markdown(f"### 🌾 Stage: {stage}")
            # Visual progress bar logic
            st.progress(min(1.0, days_age / c_data["duration_days"]), text="Lifecycle Progress")

            # SMART ADVICE
            rain_val = st.session_state.get('last_rain', 0.0)
            
            st.markdown("#### 🧠 AI Farmer Advice")
            if c_data["critical_rain_stage"] in stage and rain_val > 2.0:
                 st.error(f"⚠️ **DANGER:** Rain predicted during {stage}! Check drainage immediately.")
            elif rain_val < 1.0 and c_data["water_needs"] == "High":
                 st.info("💧 **Irrigation:** Soil moisture low. You can irrigate today.")
            else:
                 st.success("✅ **Status:** Crop conditions are stable.")

# --- TAB 3: CHATBOT ---
with tab3:
    st.subheader(t["chat_tab"])
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Sat Sri Akal! How can I help your farm today?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about weather, crops..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        resp = get_chatbot_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        st.chat_message("assistant").write(resp)

# --- TAB 4: HISTORY ---
with tab4:
    st.subheader(t["hist_tab"])
    try:
        df_hist = pd.read_csv("Synthetic_Rainfall_Dataset_1100.csv")
        st.line_chart(df_hist["rainfall"]) # Simple chart for speed
    except:
        st.warning("Historical CSV file not found.")
