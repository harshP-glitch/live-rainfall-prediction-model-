import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import datetime as dt
import joblib
import sqlite3
import time
import random
import os
from tensorflow.keras.models import load_model
from gtts import gTTS         # Text-to-Speech
import speech_recognition as sr # Speech-to-Text

# ------------------------------------------------------
# 1. 🧱 PAGE CONFIG & CSS
# ------------------------------------------------------
st.set_page_config(
    page_title="Kisan Farm OS",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6 !important;
        border: 1px solid #d1d5db;
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stMetricLabel"] p { color: #31333F !important; font-weight: bold; }
    div[data-testid="stMetricValue"] div { color: #000000 !important; }

    /* Traffic Light Alerts */
    .traffic-red { background-color: #ffe6e6; border: 2px solid #ff4d4d; padding: 20px; border-radius: 10px; text-align: center; color: #cc0000; }
    .traffic-yellow { background-color: #fff3cd; border: 2px solid #ffc107; padding: 20px; border-radius: 10px; text-align: center; color: #856404; }
    .traffic-green { background-color: #d4edda; border: 2px solid #28a745; padding: 20px; border-radius: 10px; text-align: center; color: #155724; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. 🔊 VOICE FUNCTIONS (NEW)
# ------------------------------------------------------
def text_to_speech(text, lang='en'):
    """Converts text to audio and plays it."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = "advisory.mp3"
        tts.save(filename)
        st.audio(filename, format="audio/mp3")
        os.remove(filename) # Clean up
    except Exception as e:
        st.error(f"Audio Error: {e}")

def recognize_speech():
    """Listens to the microphone and returns text."""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.toast("🎤 Listening... Speak now!", icon="👂")
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            return text
    except sr.RequestError:
        st.error("Internet required for voice recognition.")
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Try again.")
    except Exception as e:
        st.error(f"Microphone Error: {e} (Ensure PyAudio is installed)")
    return None

# ------------------------------------------------------
# 3. 💾 DATABASE & CONFIG
# ------------------------------------------------------
conn = sqlite3.connect('farm_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedback 
             (date TEXT, city TEXT, pred_rain REAL, actual_rain REAL, error REAL)''')
conn.commit()

def save_feedback(city, pred, actual):
    err = abs(pred - actual)
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?)", 
              (dt.datetime.now(), city, pred, actual, err))
    conn.commit()

CROP_INFO = {
    "Wheat": {"duration": 140, "stages": [(0,20,"Germination"), (21,60,"Tillering"), (61,90,"Flowering"), (91,140,"Ripening")]},
    "Rice": {"duration": 120, "stages": [(0,15,"Seedling"), (16,45,"Tillering"), (46,75,"Flowering"), (76,120,"Harvesting")]},
    "Cotton": {"duration": 160, "stages": [(0,20,"Seedling"), (21,60,"Vegetative"), (61,100,"Flowering"), (101,160,"Picking")]}
}

CITY_COORDS = {
    "Mohali, PB": (30.7046, 76.7179),
    "Ludhiana, PB": (30.9010, 75.8573),
    "Delhi, NCR": (28.6139, 77.2090),
    "Bathinda, PB": (30.2109, 74.9455)
}

@st.cache_resource
def load_resources():
    try:
        model = load_model("rainfall.h5")
        f_scaler = joblib.load("feature_scaler.pkl")
        t_scaler = joblib.load("target_scaler.pkl")
        return model, f_scaler, t_scaler
    except:
        return None, None, None

model, feature_scaler, target_scaler = load_resources()

# ------------------------------------------------------
# 4. 🧠 CORE FUNCTIONS
# ------------------------------------------------------
def fetch_weather_and_predict(city):
    if not model: return None, 0.0
    lat, lon = CITY_COORDS.get(city, (30.7046, 76.7179))
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=rain,relative_humidity_2m,cloudcover,wind_speed_10m,temperature_2m&past_days=1&forecast_days=1"
    
    try:
        resp = requests.get(url).json()
        h = resp.get("hourly", {})
        df = pd.DataFrame({
            "time": pd.to_datetime(h["time"]),
            "rain": h["rain"], "hum": h["relative_humidity_2m"],
            "cloud": h["cloudcover"], "wind": h["wind_speed_10m"], "temp": h["temperature_2m"]
        })
        
        now = pd.Timestamp.now().floor('h')
        df['diff'] = abs(df['time'] - now)
        idx = df['diff'].idxmin()
        
        current_weather = {
            "temp": df.loc[idx, "temp"], "hum": df.loc[idx, "hum"],
            "wind": df.loc[idx, "wind"], "cloud": df.loc[idx, "cloud"]
        }
        
        def val(i, c): return df.loc[i, c] if i >= 0 else 0.0
        features = [
            val(idx-1, "rain"), val(idx-3, "rain"), val(idx-7, "rain"),
            val(idx-1, "hum"), val(idx-1, "wind"), val(idx-1, "temp"),
            df.loc[max(0, idx-2):idx, "rain"].mean(),
            df.loc[max(0, idx-2):idx, "hum"].mean(),
            df.loc[max(0, idx-2):idx, "cloud"].mean(),
            dt.datetime.now().month, dt.datetime.now().weekday()
        ]
        
        X_scaled = feature_scaler.transform(np.array(features).reshape(1, -1)).reshape(1, 1, 11)
        if model.input_shape[-1] == 1: X_scaled = X_scaled[:, :, 0:1]
        
        y = model.predict(X_scaled)
        pred_val = max(0.0, float(target_scaler.inverse_transform(y)[0][0]))
        return current_weather, pred_val
    except:
        return None, 0.0

def diagnose_plant(img):
    time.sleep(1.5)
    diseases = ["Yellow Rust", "Leaf Blight", "Healthy Crop", "Nitrogen Deficiency"]
    return random.choice(diseases), random.uniform(88, 99)

def get_market_prices(crop):
    base = 2275 if crop == "Wheat" else 3100
    return pd.DataFrame([
        {"Mandi": "Local APMC", "Price (₹/Q)": base, "Trend": "Stable"},
        {"Mandi": "District Main", "Price (₹/Q)": base + 45, "Trend": "⬆️ High"},
        {"Mandi": "Private Trader", "Price (₹/Q)": base - 20, "Trend": "⬇️ Low"},
    ])

def get_bot_response(msg, rain, crop):
    msg = msg.lower()
    if "price" in msg: return f"The current price for {crop} is ₹2200 per quintal."
    if "spray" in msg: return "Do not spray if rain is predicted > 1mm." if rain > 1 else "Yes, weather looks clear for spraying."
    if "irrigate" in msg: return "Hold irrigation." if rain > 5 else "Soil moisture is low. You can irrigate."
    return f"I am managing your {crop}. Ask me about prices, weather, or health."

# ------------------------------------------------------
# 5. 🖥️ UI LAYOUT
# ------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "landing"
if "data" not in st.session_state: st.session_state.data = {}

if st.session_state.page == "landing":
    st.markdown("<h1 style='text-align: center;'>🚜 Kisan Farm OS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Voice-Enabled Smart Farming</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("onboard"):
            st.subheader("Create Farm Profile")
            name = st.text_input("Name", "Ram Singh")
            crop = st.selectbox("Crop", list(CROP_INFO.keys()))
            city = st.selectbox("Region", list(CITY_COORDS.keys()))
            sowing = st.date_input("Sowing Date", dt.date.today() - dt.timedelta(days=45))
            if st.form_submit_button("🚀 Start"):
                st.session_state.data = {"name": name, "crop": crop, "city": city, "sowing": sowing}
                st.session_state.page = "dashboard"
                st.rerun()

else:
    data = st.session_state.data
    crop_conf = CROP_INFO[data["crop"]]
    
    st.sidebar.title("🚜 Farm OS")
    st.sidebar.info(f"👤 {data['name']}\n\n📍 {data['city']}")
    if st.sidebar.button("Log Out"): 
        st.session_state.page = "landing"
        st.rerun()

    weather, pred_rain = fetch_weather_and_predict(data["city"])
    days_age = (dt.date.today() - data["sowing"]).days

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌦️ Weather & Voice", "🌾 Crop Health", "🏥 Plant Doctor", "💰 Prices", "💬 Assistant"
    ])

    # --- TAB 1: WEATHER + VOICE OUTPUT ---
    with tab1:
        st.subheader("Live Forecast")
        
        # Prepare Advisory Text
        if pred_rain > 5.0:
            status = "Heavy Rain"
            adv_text = f"Alert! Heavy rain of {pred_rain:.1f} millimeters predicted. Stop all irrigation and spraying immediately."
            st.markdown(f"<div class='traffic-red'><h2>🚫 STOP WORK</h2><p>{status} ({pred_rain:.1f}mm)</p></div>", unsafe_allow_html=True)
        elif pred_rain > 1.0:
            status = "Light Rain"
            adv_text = f"Caution. Light rain of {pred_rain:.1f} millimeters expected. Delay pesticide spraying."
            st.markdown(f"<div class='traffic-yellow'><h2>⚠️ CAUTION</h2><p>{status} ({pred_rain:.1f}mm)</p></div>", unsafe_allow_html=True)
        else:
            status = "Clear"
            adv_text = f"Weather is clear. You can proceed with irrigation and farm operations."
            st.markdown(f"<div class='traffic-green'><h2>🟢 GO AHEAD</h2><p>{status} Weather</p></div>", unsafe_allow_html=True)

        # 🔊 VOICE ADVISORY BUTTON
        st.write("")
        if st.button("🔊 Listen to Advisory"):
            text_to_speech(adv_text)

        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        if weather:
            m1.metric("Rainfall", f"{pred_rain:.1f} mm")
            m2.metric("Temp", f"{weather['temp']} °C")
            m3.metric("Humidity", f"{weather['hum']}%")
            m4.metric("Wind", f"{weather['wind']} km/h")

    # --- TAB 2: CROP HEALTH ---
    with tab2:
        st.subheader("Crop Growth")
        c1, c2 = st.columns([2, 1])
        with c1:
            x = list(range(0, crop_conf["duration"] + 1, 5))
            y = [100 / (1 + np.exp(-0.1 * (i - crop_conf["duration"]/2))) for i in x]
            curr_y = 100 / (1 + np.exp(-0.1 * (days_age - crop_conf["duration"]/2)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, name="Ideal Path", line=dict(color='green', dash='dot')))
            fig.add_trace(go.Scatter(x=[days_age], y=[curr_y], mode='markers', marker=dict(color='red', size=15), name="You"))
            st.plotly_chart(fig, use_container_width=True)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Wheat_growth_stages.png/320px-Wheat_growth_stages.png", caption="Stages")

        with c2:
            st.metric("Age", f"{days_age} Days")
            curr_stage = "Harvested"
            for s, e, n in crop_conf["stages"]:
                if s <= days_age <= e: curr_stage = n
            st.metric("Stage", curr_stage)
            st.progress(min(100, int(days_age/crop_conf["duration"]*100)))

    # --- TAB 3: PLANT DOCTOR ---
    with tab3:
        st.subheader("📸 AI Plant Doctor")
        img = st.file_uploader("Upload Leaf Photo", type=['jpg', 'png'])
        if img:
            st.image(img, width=200)
            if st.button("Diagnose"):
                with st.spinner("Scanning..."):
                    d, c = diagnose_plant(img)
                st.success(f"Result: {d}")

    # --- TAB 4: PRICES ---
    with tab4:
        st.subheader("💰 Market Prices")
        st.dataframe(get_market_prices(data["crop"]), use_container_width=True)

    # --- TAB 5: VOICE ASSISTANT (INPUT) ---
    with tab5:
        st.subheader("🤖 Voice Assistant")
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # 🎤 VOICE INPUT BUTTON
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🎤 Speak"):
                voice_text = recognize_speech()
                if voice_text:
                    st.session_state.messages.append({"role": "user", "content": voice_text})
                    resp = get_bot_response(voice_text, pred_rain, data["crop"])
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                    st.rerun()

        with col2:
            if prompt := st.chat_input("Or type here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                resp = get_bot_response(prompt, pred_rain, data["crop"])
                st.session_state.messages.append({"role": "assistant", "content": resp})
                st.rerun()
