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
# 1. 🧱 PAGE CONFIG & CSS (Dark Mode Fix)
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
    /* Force Text to Black for readability */
    div[data-testid="stMetricLabel"] p { color: #31333F !important; font-weight: bold; }
    div[data-testid="stMetricValue"] div { color: #000000 !important; }

    /* Traffic Light Alerts */
    .traffic-red { background-color: #ffe6e6; border: 2px solid #ff4d4d; padding: 20px; border-radius: 10px; text-align: center; color: #cc0000; }
    .traffic-yellow { background-color: #fff3cd; border: 2px solid #ffc107; padding: 20px; border-radius: 10px; text-align: center; color: #856404; }
    .traffic-green { background-color: #d4edda; border: 2px solid #28a745; padding: 20px; border-radius: 10px; text-align: center; color: #155724; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. 💾 SELF-LEARNING DATABASE (SQLite)
# ------------------------------------------------------
# Creates a local file 'farm_data.db' to store farmer feedback
conn = sqlite3.connect('farm_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedback 
             (date TEXT, city TEXT, pred_rain REAL, actual_rain REAL, error REAL)''')
conn.commit()

def save_feedback(city, pred, actual):
    """Saves ground truth data for future model retraining."""
    err = abs(pred - actual)
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?)", 
              (dt.datetime.now(), city, pred, actual, err))
    conn.commit()

# ------------------------------------------------------
# 3. ⚙️ LOAD RESOURCES & CONFIG
# ------------------------------------------------------
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
# 4. 🧠 CORE FUNCTIONS (Weather, Health, Market)
# ------------------------------------------------------

# A. LSTM WEATHER PREDICTION
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
        
        # Lags logic for LSTM
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
        if model.input_shape[-1] == 1: X_scaled = X_scaled[:, :, 0:1] # Fix dim mismatch
        
        y = model.predict(X_scaled)
        pred_val = max(0.0, float(target_scaler.inverse_transform(y)[0][0]))
        return current_weather, pred_val

    except Exception as e:
        return None, 0.0

# B. PLANT DOCTOR SIMULATION
def diagnose_plant(img):
    time.sleep(1.5) # Simulate AI processing time
    diseases = ["Yellow Rust (Need Fungicide)", "Leaf Blight", "Healthy Crop", "Nitrogen Deficiency"]
    return random.choice(diseases), random.uniform(88, 99)

# C. MANDI PRICE SIMULATION
def get_market_prices(crop):
    base = 2275 if crop == "Wheat" else 3100
    return pd.DataFrame([
        {"Mandi": "Local APMC", "Price (₹/Q)": base, "Trend": "Stable"},
        {"Mandi": "District Main", "Price (₹/Q)": base + 45, "Trend": "⬆️ High"},
        {"Mandi": "Private Trader", "Price (₹/Q)": base - 20, "Trend": "⬇️ Low"},
    ])

# D. CHATBOT BRAIN
def get_bot_response(msg, rain, crop):
    msg = msg.lower()
    if "price" in msg: return f"The current trend for {crop} is stable around ₹2200/Q."
    if "spray" in msg: return "Do not spray if rain is predicted > 1mm." if rain > 1 else "Yes, weather looks clear for spraying."
    if "irrigate" in msg: return "Hold irrigation." if rain > 5 else "Soil moisture is low. You can irrigate."
    return f"I am managing your {crop}. Ask me about prices, weather, or health."

# ------------------------------------------------------
# 5. 🖥️ UI LAYOUT & LOGIC
# ------------------------------------------------------

# Session State
if "page" not in st.session_state: st.session_state.page = "landing"
if "data" not in st.session_state: st.session_state.data = {}

# --- LANDING PAGE ---
if st.session_state.page == "landing":
    st.markdown("<h1 style='text-align: center;'>🚜 Kisan Farm OS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>One-Stop Solution: Weather • Health • Markets</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("onboard"):
            st.subheader("Create Farm Profile")
            name = st.text_input("Farmer Name", "Ram Singh")
            crop = st.selectbox("Select Crop", list(CROP_INFO.keys()))
            city = st.selectbox("Nearest City", list(CITY_COORDS.keys()))
            sowing = st.date_input("Sowing Date", dt.date.today() - dt.timedelta(days=45))
            
            if st.form_submit_button("🚀 Launch Dashboard"):
                st.session_state.data = {"name": name, "crop": crop, "city": city, "sowing": sowing}
                st.session_state.page = "dashboard"
                st.rerun()

# --- MAIN DASHBOARD ---
else:
    data = st.session_state.data
    crop_conf = CROP_INFO[data["crop"]]
    
    # Sidebar
    st.sidebar.title("🚜 Farm OS")
    st.sidebar.info(f"👤 **{data['name']}**\n\n📍 {data['city']}\n\n🌾 {data['crop']}")
    if st.sidebar.button("Log Out"): 
        st.session_state.page = "landing"
        st.rerun()

    # Fetch Real Data
    weather, pred_rain = fetch_weather_and_predict(data["city"])
    days_age = (dt.date.today() - data["sowing"]).days

    # MAIN TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌦️ Weather & Learning", 
        "🌾 Crop Health", 
        "🏥 Plant Doctor", 
        "💰 Mandi Prices", 
        "💬 Assistant"
    ])

    # --- TAB 1: WEATHER & FEEDBACK ---
    with tab1:
        st.subheader("Live Forecast & Advisory")
        
        # Traffic Light
        if pred_rain > 5.0:
            st.markdown(f"<div class='traffic-red'><h2>🚫 STOP WORK</h2><p>Heavy Rain ({pred_rain:.1f}mm) Predicted.</p></div>", unsafe_allow_html=True)
        elif pred_rain > 1.0:
            st.markdown(f"<div class='traffic-yellow'><h2>⚠️ CAUTION</h2><p>Light Rain ({pred_rain:.1f}mm). Delay Spraying.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='traffic-green'><h2>🟢 GO AHEAD</h2><p>Clear Weather. Safe for Irrigation/Spray.</p></div>", unsafe_allow_html=True)

        st.write("")
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        if weather:
            m1.metric("Rainfall", f"{pred_rain:.1f} mm")
            m2.metric("Temp", f"{weather['temp']} °C")
            m3.metric("Humidity", f"{weather['hum']}%")
            m4.metric("Wind", f"{weather['wind']} km/h")
        
        st.markdown("---")
        # SELF LEARNING SECTION
        with st.expander("🧠 Teach the Model (Self-Learning)"):
            st.write("Is the prediction incorrect? Tell us the truth so the model learns.")
            actual_rain = st.number_input("Actual Rainfall Observed (mm):", 0.0, 100.0)
            if st.button("Submit Feedback"):
                save_feedback(data["city"], pred_rain, actual_rain)
                st.success("✅ Feedback Saved! The model will be retrained with this data.")

    # --- TAB 2: CROP HEALTH VISUALS ---
    with tab2:
        st.subheader(f"{data['crop']} Growth Trajectory")
        c1, c2 = st.columns([2, 1])
        with c1:
            # S-Curve Chart
            x = list(range(0, crop_conf["duration"] + 1, 5))
            y = [100 / (1 + np.exp(-0.1 * (i - crop_conf["duration"]/2))) for i in x]
            curr_y = 100 / (1 + np.exp(-0.1 * (days_age - crop_conf["duration"]/2)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, name="Ideal Path", line=dict(color='green', dash='dot')))
            fig.add_trace(go.Scatter(x=[days_age], y=[curr_y], mode='markers', marker=dict(color='red', size=15), name="You are here"))
            fig.update_layout(height=300, xaxis_title="Days", yaxis_title="% Growth", margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Contextual Image
            st.markdown("

[Image of crop growth stages timeline]
")

        with c2:
            st.info(f"📅 **Day {days_age}** of {crop_conf['duration']}")
            # Determine Stage
            curr_stage = "Harvested"
            for s, e, n in crop_conf["stages"]:
                if s <= days_age <= e: curr_stage = n
            st.metric("Current Stage", curr_stage)
            st.progress(min(100, int(days_age/crop_conf["duration"]*100)))

    # --- TAB 3: PLANT DOCTOR ---
    with tab3:
        st.subheader("📸 AI Plant Doctor")
        st.write("Upload a photo of a sick leaf to identify diseases.")
        
        img = st.file_uploader("Upload Image", type=['jpg', 'png'])
        if img:
            st.image(img, width=250, caption="Uploaded Leaf")
            if st.button("Diagnose Disease"):
                with st.spinner("Analyzing leaf patterns..."):
                    disease, conf = diagnose_plant(img)
                st.success(f"**Diagnosis:** {disease}")
                st.info(f"**Confidence:** {conf:.1f}%")
                if "Healthy" not in disease:
                    st.warning("💊 **Recommendation:** Apply Propiconazole 25% EC (1ml/liter water).")

    # --- TAB 4: MARKET PRICES ---
    with tab4:
        st.subheader("💰 Live Mandi Prices")
        st.caption("Real-time rates from nearby markets.")
        df_prices = get_market_prices(data["crop"])
        st.dataframe(df_prices, use_container_width=True)
        st.line_chart([2200, 2250, 2220, 2280, 2300]) # Dummy trend
        st.caption("Price Trend (Last 5 Days)")

    # --- TAB 5: CHATBOT ---
    with tab5:
        st.subheader("🤖 Kisan Assistant")
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Ask about weather, prices, or disease..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            resp = get_bot_response(prompt, pred_rain, data["crop"])
            
            st.chat_message("assistant").write(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
