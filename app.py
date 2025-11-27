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
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: white;
        border: 1px solid #eee;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        height: 100px;
        text-align: center;
        cursor: pointer;
    }
    .action-btn:hover { transform: translateY(-2px); border-color: #4CAF50; }
    .action-icon { font-size: 2rem; margin-bottom: 5px; }
    .action-label { font-size: 0.9rem; font-weight: 600; color: #333; }

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
    
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. ⚙️ BACKEND LOGIC (Mocked for UI Focus)
# ------------------------------------------------------
def mock_login(phone):
    """Simulates checking DB for user"""
    time.sleep(1)
    return True

def get_weather():
    """Returns dummy weather for the demo"""
    return {"temp": 28, "rain": 6.2, "hum": 75, "wind": 12}

def get_mandi_rates():
    return [
        {"Crop": "Wheat", "Mandi": "Mohali", "Price": "₹2,150", "Trend": "⬆️"},
        {"Crop": "Rice", "Mandi": "Khanna", "Price": "₹3,200", "Trend": "⬇️"},
        {"Crop": "Maize", "Mandi": "Rajpura", "Price": "₹1,850", "Trend": "➖"},
    ]

# ------------------------------------------------------
# 3. 📱 APP FLOW
# ------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- SCREEN 1: LOGIN (Onboarding) ---
if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=100)
    st.title("Kisan Farm OS")
    st.write("India's #1 Smart Farming App")
    
    with st.container():
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        phone = st.text_input("📱 Enter Mobile Number", placeholder="+91 98765 43210")
        lang = st.selectbox("🌐 Select Language", ["English", "Hindi (हिंदी)", "Punjabi (ਪੰਜਾਬੀ)"])
        
        if st.button("Get OTP", type="primary", use_container_width=True):
            if len(phone) >= 10:
                with st.spinner("Verifying..."):
                    mock_login(phone)
                    st.session_state.authenticated = True
                    st.rerun()
            else:
                st.error("Please enter valid number")
        st.markdown("</div>", unsafe_allow_html=True)

# --- SCREEN 2: MAIN APP ---
else:
    # --- HEADER ---
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown("### 👋 Namaste, Ram Singh")
        st.caption("📍 Mohali, Punjab")
    with c2:
        if st.button("🚪"):
            st.session_state.authenticated = False
            st.rerun()

    # --- NAVIGATION LOGIC ---
    if st.session_state.page == "home":
        
        # 1. WEATHER HERO CARD
        w = get_weather()
        status_color = "pill-red" if w['rain'] > 5 else "pill-green"
        status_text = "STOP SPRAYING" if w['rain'] > 5 else "SAFE TO WORK"
        
        st.markdown(f"""
        <div class="weather-hero">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="status-pill {status_color}">{status_text}</span>
                <span>{dt.datetime.now().strftime('%d %b, %I:%M %p')}</span>
            </div>
            <h1>{w['temp']}°C</h1>
            <p>🌧️ Rain: {w['rain']}mm | 💧 Hum: {w['hum']}% | 💨 Wind: {w['wind']}km/h</p>
        </div>
        """, unsafe_allow_html=True)

        # 2. ACTION GRID (The "App" Feel)
        st.markdown("### 🚜 Quick Actions")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            if st.button("🏥\nDoctor"): st.session_state.page = "doctor"; st.rerun()
        with c2:
            if st.button("💰\nMandi"): st.session_state.page = "mandi"; st.rerun()
        with c3:
            if st.button("🤖\nAsk AI"): st.session_state.page = "chat"; st.rerun()
        with c4:
            if st.button("🌾\nCrop"): st.session_state.page = "crop"; st.rerun()
        
        # 3. DAILY FEED (Stories style)
        st.markdown("### 📢 Daily Updates")
        
        # Feed Item 1
        st.markdown("""
        <div class="pro-card">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="font-size:25px;">🦠</div>
                <div>
                    <b>Yellow Rust Alert</b><br>
                    <span style="font-size:0.8rem; color:grey;">2 hours ago • Punjab Dept of Agri</span>
                </div>
            </div>
            <p style="margin-top:10px; font-size:0.9rem;">High humidity detected. Check your wheat leaves for yellow powder spots today.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feed Item 2
        st.markdown("""
        <div class="pro-card">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="font-size:25px;">💸</div>
                <div>
                    <b>Subsidy Scheme</b><br>
                    <span style="font-size:0.8rem; color:grey;">1 day ago • Govt of India</span>
                </div>
            </div>
            <p style="margin-top:10px; font-size:0.9rem;">Apply for 50% subsidy on Solar Water Pumps before 30th Nov.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- PAGE: PLANT DOCTOR ---
    elif st.session_state.page == "doctor":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("🏥 Crop Doctor")
        
        with st.container():
            st.markdown("<div class='pro-card' style='text-align:center;'>", unsafe_allow_html=True)
            st.markdown("### 📸 Snap & Cure")
            st.write("Take a clear photo of the affected leaf")
            img = st.file_uploader(" ", label_visibility="collapsed")
            if img:
                st.image(img, width=200)
                st.success("Analysis: Yellow Rust Detected")
                st.info("Treatment: Spray Propiconazole (1ml/L)")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- PAGE: MANDI RATES ---
    elif st.session_state.page == "mandi":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("💰 Market Rates")
        
        rates = get_mandi_rates()
        for r in rates:
            st.markdown(f"""
            <div class="pro-card" style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:bold; font-size:1.1rem;">{r['Crop']}</div>
                    <div style="color:grey; font-size:0.9rem;">📍 {r['Mandi']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-weight:bold; font-size:1.2rem; color:#2E7D32;">{r['Price']}</div>
                    <div style="font-size:0.8rem;">{r['Trend']} Trend</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- PAGE: CROP ---
    elif st.session_state.page == "crop":
        if st.button("← Back"): st.session_state.page = "home"; st.rerun()
        st.title("🌾 My Crop")
        
        st.markdown("""
        <div class="pro-card">
            <h3>Wheat (Rabi)</h3>
            <p>Age: 45 Days</p>
            <div style="background:#eee; height:10px; border-radius:5px; margin:10px 0;">
                <div style="background:#4CAF50; width:40%; height:100%; border-radius:5px;"></div>
            </div>
            <p style="text-align:right; font-size:0.8rem;">Tillering Stage</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Wheat_growth_stages.png/320px-Wheat_growth_stages.png", caption="Reference")
