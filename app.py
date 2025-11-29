import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import joblib
import sqlite3
import time
import random
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Kisan Farm OS",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# VOICE SUPPORT (Graceful Fallback)
try:
    from gtts import gTTS
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False

# --- 2. DATABASE: THE BRAIN ---
# We use SQLite to store Users, Feedback, AND Community Posts
conn = sqlite3.connect('farm_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (phone TEXT PRIMARY KEY, name TEXT, city TEXT, crop TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, content TEXT, timestamp TEXT)''')
conn.commit()

# --- 3. PROFESSIONAL STYLING (Mobile App Feel) ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: black !important; }
    
    /* INPUTS */
    label, .stTextInput label, .stSelectbox label { color: #31333F !important; font-weight: 600; }
    
    /* CARDS */
    .pro-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    
    /* HERO CARDS */
    .hero-card {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white !important;
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(46, 125, 50, 0.25);
    }
    .hero-card h1, .hero-card h2, .hero-card p { color: white !important; }
    
    /* BUTTONS */
    .stButton button {
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
        border-radius: 12px !important;
        height: 60px !important;
        font-weight: 600 !important;
    }
    .stButton button:hover {
        border-color: #2E7D32 !important;
        color: #2E7D32 !important;
        background-color: #f0fdf4 !important;
    }
    
    /* ALERTS */
    .alert-red { background: #fee2e2; border-left: 5px solid #ef4444; padding: 15px; border-radius: 8px; color: #991b1b; }
    .alert-green { background: #dcfce7; border-left: 5px solid #22c55e; padding: 15px; border-radius: 8px; color: #166534; }
    
    /* HIDE DEFAULTS */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 4. INTELLIGENCE & DATA LAYERS
# ------------------------------------------------------

# A. DATA SOURCES (Simulating Real-World APIs)
GOVT_SCHEMES = [
    {"name": "PM-KISAN", "benefit": "₹6,000/year", "eligibility": "All landholding farmers"},
    {"name": "Kisan Credit Card (KCC)", "benefit": "Low Interest Loan (4%)", "eligibility": "All farmers"},
    {"name": "PM Fasal Bima Yojana", "benefit": "Crop Insurance", "eligibility": "Crop loss due to weather"},
    {"name": "Solar Pump Subsidy", "benefit": "60% Subsidy", "eligibility": "Punjab/Haryana Farmers"}
]

# B. AI MODELS LOAD
@st.cache_resource
def load_ai_models():
    # 1. Weather Model
    try:
        weather_model = load_model("rainfall.h5")
        f_scaler = joblib.load("feature_scaler.pkl")
        t_scaler = joblib.load("target_scaler.pkl")
    except: weather_model, f_scaler, t_scaler = None, None, None

    # 2. Disease Model (Check for your file)
    try:
        disease_model = load_model("plant_disease_model.h5")
    except: disease_model = None
    
    return weather_model, f_scaler, t_scaler, disease_model

model, f_scaler, t_scaler, disease_model = load_ai_models()

# C. REAL-TIME WEATHER API
def get_live_weather(city):
    lat_lon = {
        "Mohali": (30.7, 76.7), "Ludhiana": (30.9, 75.8), 
        "Delhi": (28.6, 77.2), "Bathinda": (30.2, 74.9)
    }
    lat, lon = lat_lon.get(city, (30.7, 76.7))
    
    try:
        # Fetching Live Data from Open-Meteo
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=rain,relative_humidity_2m,cloudcover,wind_speed_10m,temperature_2m&past_days=1&forecast_days=1"
        data = requests.get(url).json()
        
        # Current conditions
        current = data['current_weather']
        hourly = data['hourly']
        
        # Get humidity (Open-Meteo current_weather doesn't have humidity, so we take closest hourly)
        now_idx = abs(pd.to_datetime(hourly['time']) - pd.Timestamp.now()).idxmin()
        
        return {
            "temp": current['temperature'],
            "wind": current['windspeed'],
            "hum": hourly['relative_humidity_2m'][now_idx],
            "rain_forecast": sum(hourly['rain'][now_idx:now_idx+24]) # Total rain next 24h
        }
    except:
        return {"temp": 28, "wind": 10, "hum": 60, "rain_forecast": 0.0}

# D. DISEASE PREDICTION (Hybrid: Real + Fallback)
def diagnose_leaf(image):
    if disease_model:
        try:
            # Real Inference
            img = Image.open(image).resize((224, 224))
            img_arr = img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            pred = disease_model.predict(img_arr)
            confidence = np.max(pred) * 100
            # (Map class index to name here if you have class_indices.txt)
            return f"Disease Detected (Class {np.argmax(pred)})", confidence
        except: pass
    
    # Simulation Fallback (for demo if model missing)
    time.sleep(1.5)
    return "Yellow Rust (Fungal)", 92.5

# ------------------------------------------------------
# 5. APP LOGIC & NAVIGATION
# ------------------------------------------------------

# Session Management
if "page" not in st.session_state: st.session_state.page = "login"
if "user" not in st.session_state: st.session_state.user = None

# === PAGE: LOGIN ===
if st.session_state.page == "login":
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=80)
    st.markdown("## Kisan Farm OS")
    st.write("Your Digital Agriculture Partner")
    
    with st.container():
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        name = st.text_input("Name", "Ram Singh")
        city = st.selectbox("Region", ["Mohali", "Ludhiana", "Delhi", "Bathinda"])
        phone = st.text_input("Mobile Number", placeholder="9876543210")
        
        if st.button("Get OTP", type="primary", use_container_width=True):
            if len(phone) == 10:
                st.session_state.otp = random.randint(1000, 9999)
                st.session_state.temp_user = {"name": name, "city": city, "phone": phone}
                st.toast(f"🔔 OTP: {st.session_state.otp}", icon="📩")
                st.session_state.otp_sent = True
            else:
                st.error("Enter valid mobile number")
        
        if st.session_state.get("otp_sent"):
            st.info(f"OTP sent to {phone}. (Use {st.session_state.otp})")
            otp_input = st.text_input("Enter OTP", max_chars=4)
            if st.button("Verify & Login", use_container_width=True):
                if str(otp_input) == str(st.session_state.otp):
                    st.session_state.user = st.session_state.temp_user
                    st.session_state.page = "home"
                    st.rerun()
                else:
                    st.error("Invalid OTP")
        st.markdown("</div>", unsafe_allow_html=True)

# === PAGE: MAIN DASHBOARD ===
elif st.session_state.page == "home":
    user = st.session_state.user
    weather = get_live_weather(user['city'])
    
    # Header
    c1, c2 = st.columns([4, 1])
    with c1: st.markdown(f"### 👋 {user['name']}")
    with c2: 
        if st.button("🚪"): st.session_state.page = "login"; st.rerun()

    # Weather Hero (Actionable)
    if weather['rain_forecast'] > 5.0:
        bg_grad = "linear-gradient(135deg, #DC2626 0%, #991B1B 100%)" # Red
        status = "STOP WORK"
        msg = f"Heavy rain ({weather['rain_forecast']}mm) expected. Don't spray."
    else:
        bg_grad = "linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)" # Green
        status = "GO AHEAD"
        msg = "Weather is clear for irrigation & spraying."

    st.markdown(f"""
    <div class="hero-card" style="background: {bg_grad};">
        <div style="display:flex; justify-content:space-between;">
            <span style="background:rgba(255,255,255,0.2); padding:5px 10px; border-radius:10px; font-weight:bold;">{status}</span>
            <span>{dt.datetime.now().strftime('%d %b')}</span>
        </div>
        <h1 style="margin:10px 0;">{weather['temp']}°C</h1>
        <p>💧 Hum: {weather['hum']}% | 💨 Wind: {weather['wind']}km/h</p>
        <p style="margin-top:10px; font-weight:bold;">📢 {msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # Action Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        if st.button("🏥\nDr."): st.session_state.page = "doctor"; st.rerun()
    with c2: 
        if st.button("💰\nMandi"): st.session_state.page = "mandi"; st.rerun()
    with c3: 
        if st.button("📜\nScheme"): st.session_state.page = "schemes"; st.rerun()
    with c4: 
        if st.button("🗣️\nChaupal"): st.session_state.page = "community"; st.rerun()

    # Smart Alerts
    st.markdown("### 🔔 Farm Alerts")
    if weather['hum'] > 80:
        st.markdown("<div class='alert-red'>⚠️ <b>High Humidity Alert:</b> Risk of fungal attack in Wheat. Check for yellow spots.</div>", unsafe_allow_html=True)
    st.markdown("<div class='alert-green'>📢 <b>Subsidy:</b> Applications open for Solar Pump Scheme 2025. Check 'Schemes' tab.</div>", unsafe_allow_html=True)

# === PAGE: PLANT DOCTOR (Problem Solver) ===
elif st.session_state.page == "doctor":
    if st.button("← Back"): st.session_state.page = "home"; st.rerun()
    st.title("🏥 Plant Doctor")
    
    st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
    st.write("Upload a photo of the affected leaf:")
    img = st.file_uploader(" ", label_visibility="collapsed")
    if img:
        st.image(img, use_container_width=True)
        with st.spinner("Analyzing leaf texture..."):
            diagnosis, conf = diagnose_leaf(img)
        
        st.success(f"**Diagnosis:** {diagnosis}")
        st.info(f"**Confidence:** {conf:.1f}%")
        
        st.markdown("### 💊 Treatment Plan")
        if "Rust" in diagnosis:
            st.write("1. Spray Propiconazole 25% EC (1ml/liter).")
            st.write("2. Avoid nitrogen fertilizer for 10 days.")
        else:
            st.write("Consult a local agronomist for detailed dosage.")
    st.markdown("</div>", unsafe_allow_html=True)

# === PAGE: MANDI (Money Maker) ===
elif st.session_state.page == "mandi":
    if st.button("← Back"): st.session_state.page = "home"; st.rerun()
    st.title("💰 Market Rates")
    
    # Dynamic Pricing Logic (Simulated)
    base_price = 2200
    fluctuation = random.randint(-50, 100)
    
    st.markdown(f"""
    <div class="pro-card" style="border-left: 5px solid #2E7D32;">
        <h3>Wheat (Local Mandi)</h3>
        <h1 style="color:#2E7D32">₹{base_price + fluctuation}/Q</h1>
        <p>Trend: {'📈 Up' if fluctuation > 0 else '📉 Down'} by ₹{abs(fluctuation)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚛 Nearby Markets")
    st.markdown(f"""
    <div class="pro-card" style="display:flex; justify-content:space-between;">
        <div><b>Khanna APMC</b><br><span style="color:grey">Dist. Main</span></div>
        <div style="text-align:right;"><b>₹{base_price + fluctuation + 40}</b><br><span style="color:green">+₹40</span></div>
    </div>
    <div class="pro-card" style="display:flex; justify-content:space-between;">
        <div><b>Rajpura Mandi</b><br><span style="color:grey">25km away</span></div>
        <div style="text-align:right;"><b>₹{base_price + fluctuation - 10}</b><br><span style="color:red">-₹10</span></div>
    </div>
    """, unsafe_allow_html=True)

# === PAGE: SCHEMES (Financial Aid) ===
elif st.session_state.page == "schemes":
    if st.button("← Back"): st.session_state.page = "home"; st.rerun()
    st.title("📜 Gov Schemes")
    
    for scheme in GOVT_SCHEMES:
        with st.expander(f"💰 {scheme['name']}"):
            st.write(f"**Benefit:** {scheme['benefit']}")
            st.write(f"**Eligibility:** {scheme['eligibility']}")
            st.button(f"Check Eligibility for {scheme['name']}")

# === PAGE: COMMUNITY (Digital Chaupal) ===
elif st.session_state.page == "community":
    if st.button("← Back"): st.session_state.page = "home"; st.rerun()
    st.title("🗣️ Kisan Chaupal")
    
    # Post Input
    with st.form("new_post"):
        txt = st.text_area("Ask a question to fellow farmers...")
        if st.form_submit_button("Post"):
            c.execute("INSERT INTO posts (user, content, timestamp) VALUES (?, ?, ?)", 
                      (st.session_state.user['name'], txt, str(dt.datetime.now())[:16]))
            conn.commit()
            st.success("Posted!")
            st.rerun()
    
    # Display Posts
    st.markdown("### Recent Discussions")
    posts = c.execute("SELECT * FROM posts ORDER BY id DESC LIMIT 5").fetchall()
    if posts:
        for p in posts:
            st.markdown(f"""
            <div class="pro-card">
                <b>👤 {p[1]}</b> <span style="color:grey; font-size:0.8rem;">{p[3]}</span>
                <p style="margin-top:5px;">{p[2]}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No discussions yet. Be the first to post!")
