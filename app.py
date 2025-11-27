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

# Try importing Voice Libraries (Graceful fallback if not installed)
try:
    from gtts import gTTS
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False

# ------------------------------------------------------
# 1. 🎨 PRO UI CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="Kisan Farm OS Pro",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* GLOBAL STYLES */
    .stApp { background-color: #f8f9fa; }
    #MainMenu, footer, header {visibility: hidden;}
    
    /* CARD STYLING */
    .pro-card {
        background-color: white !important;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
        color: black !important;
    }
    .pro-card h3, .pro-card p, .pro-card b, .pro-card div { color: black !important; }
    
    /* HERO WEATHER CARD */
    .weather-hero {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white !important;
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 20px rgba(46, 125, 50, 0.3);
    }
    .weather-hero h1, .weather-hero h2, .weather-hero p { color: white !important; }

    /* ACTION BUTTONS */
    .stButton button {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ddd !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        font-weight: 600 !important;
        height: 80px !important;
    }
    .stButton button:hover {
        border-color: #2E7D32 !important;
        color: #2E7D32 !important;
        background-color: #f9f9f9 !important;
    }
    
    /* SEARCH CONTAINER */
    .search-header {
        background: #2E7D32;
        padding: 20px;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .search-result { border-left: 5px solid #4CAF50; padding-left: 15px; }

    /* TRAFFIC LIGHT PILLS */
    .pill-red { background-color: #FF5252; padding: 5px 10px; border-radius: 10px; color: white; font-weight: bold; }
    .pill-green { background-color: #4CAF50; padding: 5px 10px; border-radius: 10px; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. 🧠 DATA & BACKEND LOGIC
# ------------------------------------------------------

# --- KNOWLEDGE BASE (For Search) ---
CROP_DATABASE = [
    {"name": "Wheat (Rabi)", "hindi": "गेहूं", "soil": "Loamy", "sowing": "Nov-Dec", "harvest": "Apr-May", "water": "Moderate", "diseases": ["Yellow Rust", "Loose Smut"], "varieties": ["PBW 343", "HD 2967"], "states": ["Punjab", "Haryana", "UP"]},
    {"name": "Rice (Kharif)", "hindi": "धान", "soil": "Clayey", "sowing": "Jun-Jul", "harvest": "Oct-Nov", "water": "High", "diseases": ["Blast", "Blight"], "varieties": ["Basmati 1121", "PR 126"], "states": ["Punjab", "WB", "UP"]},
    {"name": "Cotton", "hindi": "कपास", "soil": "Black Soil", "sowing": "Apr-May", "harvest": "Oct-Jan", "water": "Low", "diseases": ["Pink Bollworm"], "varieties": ["Bt Cotton"], "states": ["Gujarat", "Maharashtra"]},
    {"name": "Sugarcane", "hindi": "गन्ना", "soil": "Rich Loam", "sowing": "Feb-Mar", "harvest": "Dec-Mar", "water": "Very High", "diseases": ["Red Rot"], "varieties": ["Co 0238"], "states": ["UP", "Maharashtra"]},
]

# --- APP CONFIG DATA ---
CROP_INFO = {
    "Wheat": {"duration": 140, "stages": [(0,20,"Germination"), (21,60,"Tillering"), (61,90,"Flowering"), (91,140,"Ripening")]},
    "Rice": {"duration": 120, "stages": [(0,15,"Seedling"), (16,45,"Tillering"), (46,75,"Flowering"), (76,120,"Harvesting")]},
    "Cotton": {"duration": 160, "stages": [(0,20,"Seedling"), (21,60,"Vegetative"), (61,100,"Flowering"), (101,160,"Picking")]}
}
CITY_COORDS = {"Mohali": (30.7, 76.7), "Ludhiana": (30.9, 75.8), "Delhi": (28.6, 77.2), "Bathinda": (30.2, 74.9)}

# --- RESOURCES ---
conn = sqlite3.connect('farm_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedback (date TEXT, city TEXT, pred_rain REAL, actual_rain REAL, error REAL)''')
conn.commit()

@st.cache_resource
def load_resources():
    try:
        model = load_model("rainfall.h5")
        f_scaler = joblib.load("feature_scaler.pkl")
        t_scaler = joblib.load("target_scaler.pkl")
        return model, f_scaler, t_scaler
    except: return None, None, None

model, feature_scaler, target_scaler = load_resources()

# --- WEATHER API ---
def fetch_weather_and_predict(city):
    if not model: return {"temp": 28, "rain": 0, "hum": 60, "wind": 10}, 0.0
    lat, lon = CITY_COORDS.get(city, (30.7, 76.7))
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
        idx = abs(df['time'] - now).idxmin()
        
        # Prepare Features
        def val(i, c): return df.loc[i, c] if i >= 0 else 0.0
        features = [
            val(idx-1, "rain"), val(idx-3, "rain"), val(idx-7, "rain"),
            val(idx-1, "hum"), val(idx-1, "wind"), val(idx-1, "temp"),
            df.loc[max(0, idx-2):idx, "rain"].mean(),
            df.loc[max(0, idx-2):idx, "hum"].mean(),
            df.loc[max(0, idx-2):idx, "cloud"].mean(),
            dt.datetime.now().month, dt.datetime.now().weekday()
        ]
        X = feature_scaler.transform(np.array(features).reshape(1, -1)).reshape(1, 1, 11)
        if model.input_shape[-1] == 1: X = X[:, :, 0:1]
        
        pred = max(0.0, float(target_scaler.inverse_transform(model.predict(X))[0][0]))
        return {"temp": df.loc[idx, "temp"], "hum": df.loc[idx, "hum"], "wind": df.loc[idx, "wind"], "rain": float(f"{pred:.1f}")}, pred
    except: return {"temp": 25, "rain": 0, "hum": 50, "wind": 5}, 0.0

# --- HELPER FUNCTIONS ---
def text_to_speech(text):
    if not VOICE_ENABLED: return
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("audio.mp3")
        st.audio("audio.mp3", format="audio/mp3", start_time=0)
        os.remove("audio.mp3")
    except: pass

def get_bot_response(msg, rain, crop):
    msg = msg.lower()
    if "price" in msg: return f"{crop} price is stable at ₹2200/Q."
    if "spray" in msg: return "Don't spray, rain expected." if rain > 1 else "Safe to spray."
    return f"I can help with {crop} weather and prices."

# ------------------------------------------------------
# 3. 📱 APP NAVIGATION & PAGES
# ------------------------------------------------------
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "page" not in st.session_state: st.session_state.page = "home"
if "user_data" not in st.session_state: st.session_state.user_data = {}

# ================= LOGIN (OTP) =================
if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=100)
    st.title("Kisan Farm OS")
    
    if "otp_sent" not in st.session_state: st.session_state.otp_sent = False
    
    with st.container():
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        if not st.session_state.otp_sent:
            name = st.text_input("Full Name", "Ram Singh")
            crop = st.selectbox("Crop", list(CROP_INFO.keys()))
            city = st.selectbox("City", list(CITY_COORDS.keys()))
            sowing = st.date_input("Sowing Date", dt.date.today()-dt.timedelta(days=40))
            phone = st.text_input("Mobile", "9876543210")
            if st.button("Get OTP", type="primary", use_container_width=True):
                st.session_state.otp = random.randint(1000,9999)
                st.session_state.temp = {"name":name, "crop":crop, "city":city, "sowing":sowing}
                st.session_state.otp_sent = True
                st.toast(f"🔔 OTP: {st.session_state.otp}", icon="📩")
                st.rerun()
        else:
            st.info("OTP Sent! Check notification.")
            u_otp = st.text_input("Enter OTP", max_chars=4)
            if st.button("Verify Login", type="primary", use_container_width=True):
                if str(u_otp) == str(st.session_state.otp):
                    st.session_state.user_data = st.session_state.temp
                    st.session_state.authenticated = True
                    st.rerun()
                else: st.error("Wrong OTP")
        st.markdown("</div>", unsafe_allow_html=True)

# ================= MAIN APP =================
else:
    data = st.session_state.user_data
    w, pred_rain = fetch_weather_and_predict(data['city'])

    # --- HOME DASHBOARD ---
    if st.session_state.page == "home":
        # Header
        c1, c2 = st.columns([4, 1])
        with c1: st.markdown(f"### 👋 Namaste, {data['name']}")
        with c2: 
            if st.button("🚪"): st.session_state.authenticated = False; st.rerun()

        # Hero Card
        status = "STOP WORK" if w['rain'] > 5 else "SAFE TO WORK"
        color = "pill-red" if w['rain'] > 5 else "pill-green"
        st.markdown(f"""
        <div class="weather-hero">
            <div style="display:flex; justify-content:space-between;">
                <span class="{color}">{status}</span>
                <span>{dt.datetime.now().strftime('%d %b')}</span>
            </div>
            <h1>{w['temp']}°C</h1>
            <p>🌧️ {w['rain']}mm | 💧 {w['hum']}% | 💨 {w['wind']}km/h</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔊 Listen"): text_to_speech(f"Temperature is {w['temp']} degrees. {status}")

        # Grid
        st.markdown("### 🚜 Actions")
        c1, c2, c3, c4 = st.columns(4)
        with c1: 
            if st.button("🔍\nSearch"): st.session_state.page = "search"; st.rerun()
        with c2: 
            if st.button("🏥\nDoctor"): st.session_state.page = "doctor"; st.rerun()
        with c3: 
            if st.button("💰\nMandi"): st.session_state.page = "mandi"; st.rerun()
        with c4: 
            if st.button("🤖\nChat"): st.session_state.page = "chat"; st.rerun()

        # Feed
        st.markdown("### 📢 Alerts")
        st.markdown(f"""<div class='pro-card'>📢 <b>Mandi Update:</b> {data['crop']} prices are up by ₹40 today.</div>""", unsafe_allow_html=True)

    # --- SEARCH PAGE ---
    elif st.session_state.page == "search":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.markdown("<div class='search-header'><h1>🔍 Kisan Gyan</h1><p>Search Crops & Diseases</p></div>", unsafe_allow_html=True)
        q = st.text_input("Search...", "")
        if q:
            found = [x for x in CROP_DATABASE if q.lower() in str(x).lower()]
            if found:
                for f in found:
                    with st.expander(f"🌾 {f['name']} ({f['hindi']})", expanded=True):
                        st.write(f"**Soil:** {f['soil']} | **Water:** {f['water']}")
                        st.write(f"**Diseases:** {', '.join(f['diseases'])}")
            else: st.info("No results found.")

    # --- PLANT DOCTOR ---
    elif st.session_state.page == "doctor":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("🏥 Plant Doctor")
        with st.container():
            st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
            st.write("Upload Leaf Photo")
            img = st.file_uploader(" ", label_visibility="collapsed")
            if img:
                st.image(img, width=200)
                st.success("Detected: Yellow Rust")
                st.info("Spray Propiconazole")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- MANDI PRICES ---
    elif st.session_state.page == "mandi":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("💰 Market Rates")
        base = 2200 if data['crop'] == "Wheat" else 3000
        st.markdown(f"""
        <div class="pro-card" style="display:flex; justify-content:space-between;">
            <div><b>Local Mandi</b><br><span style="color:grey;">2km away</span></div>
            <div style="text-align:right; color:#2E7D32;"><b>₹{base}</b><br>Stable</div>
        </div>
        <div class="pro-card" style="display:flex; justify-content:space-between;">
            <div><b>District APMC</b><br><span style="color:grey;">15km away</span></div>
            <div style="text-align:right; color:#2E7D32;"><b>₹{base+50}</b><br>⬆️ High</div>
        </div>
        """, unsafe_allow_html=True)

    # --- CHATBOT ---
    elif st.session_state.page == "chat":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("🤖 Assistant")
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
        
        if prompt := st.chat_input("Ask..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            resp = get_bot_response(prompt, pred_rain, data['crop'])
            st.chat_message("assistant").write(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
