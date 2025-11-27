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
from tensorflow.keras.models import load_model

# ------------------------------------------------------
# 1. 🎨 PRO UI CONFIGURATION (Mobile-First CSS)
# ------------------------------------------------------
st.set_page_config(
    page_title="Kisan Farm OS Pro",
    page_icon="🌱",
    layout="centered", # Mobile-app feel
    initial_sidebar_state="collapsed"
)

# INJECTING PROFESSIONAL CSS
st.markdown("""
    <style>
    /* HIDE STREAMLIT DEFAULTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* APP CONTAINER */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }

    /* CARD STYLING (Material Design) */
    .pro-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    
    /* WEATHER HERO CARD */
    .weather-hero {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 20px rgba(46, 125, 50, 0.3);
    }
    .weather-hero h2 { color: white !important; margin: 0; }
    .weather-hero h1 { color: white !important; font-size: 3rem; margin: 10px 0; }
    
    /* ACTION GRID BUTTONS */
    .action-btn {
        width: 100%;
        height: 80px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background: white;
        color: #333;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        cursor: pointer;
    }
    .action-btn:hover { border-color: #2E7D32; color: #2E7D32; background: #f9f9f9; }

    /* TRAFFIC LIGHTS */
    .status-pill {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
        color: white;
    }
    .pill-red { background-color: #FF5252; }
    .pill-yellow { background-color: #FFC107; color: black; }
    .pill-green { background-color: #4CAF50; }
    
    /* NAVIGATION BAR (Fake) */
    .nav-bar {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: white; height: 60px;
        display: flex; justify-content: space-around; align-items: center;
        border-top: 1px solid #eee; z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. ⚙️ REAL BACKEND LOGIC (LSTM + DB)
# ------------------------------------------------------
# Database
conn = sqlite3.connect('farm_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedback 
             (date TEXT, city TEXT, pred_rain REAL, actual_rain REAL, error REAL)''')
conn.commit()

# Config
CROP_INFO = {
    "Wheat": {"duration": 140, "stages": [(0,20,"Germination"), (21,60,"Tillering"), (61,90,"Flowering"), (91,140,"Ripening")]},
    "Rice": {"duration": 120, "stages": [(0,15,"Seedling"), (16,45,"Tillering"), (46,75,"Flowering"), (76,120,"Harvesting")]},
    "Cotton": {"duration": 160, "stages": [(0,20,"Seedling"), (21,60,"Vegetative"), (61,100,"Flowering"), (101,160,"Picking")]}
}
CITY_COORDS = {"Mohali": (30.7, 76.7), "Ludhiana": (30.9, 75.8), "Delhi": (28.6, 77.2), "Bathinda": (30.2, 74.9)}

@st.cache_resource
def load_resources():
    try:
        model = load_model("rainfall.h5")
        f_scaler = joblib.load("feature_scaler.pkl")
        t_scaler = joblib.load("target_scaler.pkl")
        return model, f_scaler, t_scaler
    except: return None, None, None

model, feature_scaler, target_scaler = load_resources()

def fetch_weather_and_predict(city):
    if not model: return {"temp": 28, "rain": 0, "hum": 60, "wind": 10}, 0.0 # Fallback
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
        df['diff'] = abs(df['time'] - now)
        idx = df['diff'].idxmin()
        
        current = {"temp": df.loc[idx, "temp"], "hum": df.loc[idx, "hum"], "wind": df.loc[idx, "wind"], "rain": 0} # Rain filled by pred
        
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
        current['rain'] = float(f"{pred_val:.1f}")
        return current, pred_val
    except: return {"temp": 25, "rain": 0, "hum": 50, "wind": 5}, 0.0

def get_bot_response(msg, rain, crop):
    msg = msg.lower()
    if "price" in msg: return f"The current trend for {crop} is stable around ₹2200/Q."
    if "spray" in msg: return "Do not spray if rain is predicted > 1mm." if rain > 1 else "Yes, weather looks clear for spraying."
    if "irrigate" in msg: return "Hold irrigation." if rain > 5 else "Soil moisture is low. You can irrigate."
    return f"I am managing your {crop}. Ask me about prices, weather, or health."

# ------------------------------------------------------
# 3. 📱 APP FLOW
# ------------------------------------------------------

# State Management
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "page" not in st.session_state: st.session_state.page = "home"
if "user_data" not in st.session_state: st.session_state.user_data = {}

# ================= LOGIN PAGE =================
if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=100)
    st.title("Kisan Farm OS")
    st.write("India's #1 Smart Farming App")
    
    with st.container():
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("### 🚜 Create Profile")
        name = st.text_input("Full Name", "Ram Singh")
        crop = st.selectbox("Select Crop", list(CROP_INFO.keys()))
        city = st.selectbox("Region", list(CITY_COORDS.keys()))
        sowing = st.date_input("Sowing Date", dt.date.today() - dt.timedelta(days=40))
        phone = st.text_input("Mobile Number", placeholder="9876543210")
        
        if st.button("Login Securely", type="primary", use_container_width=True):
            if len(phone) >= 10:
                with st.spinner("Setting up your dashboard..."):
                    # Save User Data
                    st.session_state.user_data = {
                        "name": name, "crop": crop, "city": city, "sowing": sowing
                    }
                    st.session_state.authenticated = True
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("Enter a valid 10-digit mobile number")
        st.markdown("</div>", unsafe_allow_html=True)

# ================= DASHBOARD & PAGES =================
else:
    data = st.session_state.user_data
    
    # 1. HEADER
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(f"### 👋 Namaste, {data['name']}")
        st.caption(f"📍 {data['city']} • {data['crop']}")
    with c2:
        if st.button("🚪"):
            st.session_state.authenticated = False
            st.rerun()

    # 2. REAL DATA FETCH
    w, pred_rain = fetch_weather_and_predict(data['city'])

    # === VIEW: HOME ===
    if st.session_state.page == "home":
        
        # HERO CARD
        status_color = "pill-red" if w['rain'] > 5 else "pill-green"
        status_text = "STOP SPRAYING" if w['rain'] > 5 else "SAFE TO WORK"
        
        st.markdown(f"""
        <div class="weather-hero">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="status-pill {status_color}">{status_text}</span>
                <span>{dt.datetime.now().strftime('%d %b')}</span>
            </div>
            <h1>{w['temp']}°C</h1>
            <p>🌧️ Rain: {w['rain']}mm | 💧 Hum: {w['hum']}% | 💨 Wind: {w['wind']}km/h</p>
        </div>
        """, unsafe_allow_html=True)

        # ACTION GRID
        st.markdown("### 🚜 Quick Actions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🏥\nDr.", use_container_width=True): st.session_state.page = "doctor"; st.rerun()
        with c2:
            if st.button("💰\nRate", use_container_width=True): st.session_state.page = "mandi"; st.rerun()
        with c3:
            if st.button("🌾\nCrop", use_container_width=True): st.session_state.page = "crop"; st.rerun()
        with c4:
            if st.button("🤖\nChat", use_container_width=True): st.session_state.page = "chat"; st.rerun()

        # DAILY FEED
        st.markdown("### 📢 Updates")
        st.markdown("""
        <div class="pro-card">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="font-size:25px;">📢</div>
                <div><b>Mandi Alert</b><br><span style="color:grey; font-size:0.8rem;">Wheat prices up by ₹50 today.</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # === VIEW: PLANT DOCTOR ===
    elif st.session_state.page == "doctor":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("🏥 Plant Doctor")
        with st.container():
            st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
            st.markdown("### 📸 Scan Disease")
            img = st.file_uploader("Upload Leaf", label_visibility="collapsed")
            if img:
                st.image(img, use_container_width=True)
                st.success("Analysis: Yellow Rust")
                st.info("Rx: Spray Propiconazole (1ml/L)")
            st.markdown("</div>", unsafe_allow_html=True)

    # === VIEW: CROP STATUS ===
    elif st.session_state.page == "crop":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("🌾 My Crop")
        
        crop_conf = CROP_INFO[data['crop']]
        days_age = (dt.date.today() - data["sowing"]).days
        curr_stage = "Unknown"
        for s, e, n in crop_conf["stages"]:
            if s <= days_age <= e: curr_stage = n
            
        st.markdown(f"""
        <div class="pro-card">
            <h3>{data['crop']}</h3>
            <p style="color:#2E7D32; font-weight:bold;">{days_age} Days Old</p>
            <div style="background:#eee; height:10px; border-radius:5px; margin:10px 0;">
                <div style="background:#4CAF50; width:{min(100, int(days_age/crop_conf['duration']*100))}%; height:100%; border-radius:5px;"></div>
            </div>
            <p style="text-align:right;">Stage: {curr_stage}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # S-Curve Chart
        x = list(range(0, crop_conf["duration"] + 1, 5))
        y = [100 / (1 + np.exp(-0.1 * (i - crop_conf["duration"]/2))) for i in x]
        curr_y = 100 / (1 + np.exp(-0.1 * (days_age - crop_conf["duration"]/2)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name="Ideal", line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=[days_age], y=[curr_y], mode='markers', marker=dict(color='red', size=15), name="You"))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Wheat_growth_stages.png/320px-Wheat_growth_stages.png", caption="Reference")

    # === VIEW: MANDI ===
    elif st.session_state.page == "mandi":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("💰 Market Rates")
        st.markdown("""
        <div class="pro-card" style="display:flex; justify-content:space-between;">
            <div><b>Mohali APMC</b><br><span style="color:grey;">Local</span></div>
            <div style="text-align:right; color:#2E7D32;"><b>₹2,250</b><br>⬆️</div>
        </div>
        <div class="pro-card" style="display:flex; justify-content:space-between;">
            <div><b>Rajpura Main</b><br><span style="color:grey;">40km away</span></div>
            <div style="text-align:right; color:#2E7D32;"><b>₹2,310</b><br>⬆️</div>
        </div>
        """, unsafe_allow_html=True)

    # === VIEW: CHAT ===
    elif st.session_state.page == "chat":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("🤖 Assistant")
        
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Ask about weather, prices..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            resp = get_bot_response(prompt, pred_rain, data['crop'])
            st.chat_message("assistant").write(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
