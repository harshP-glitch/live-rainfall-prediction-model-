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
# 🧱 PAGE CONFIGURATION & STYLING
# ------------------------------------------------------
st.set_page_config(
    page_title="Kisan Smart Farm",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🛑 CSS FIX: FORCE BLACK TEXT ON CARDS & TRAFFIC LIGHTS
st.markdown("""
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6 !important; /* Light Gray Background */
        border: 1px solid #d1d5db;
        padding: 15px;
        border-radius: 10px;
    }
    
    /* FORCE TEXT BLACK in Dark Mode */
    div[data-testid="stMetricLabel"] p { color: #31333F !important; font-weight: bold; }
    div[data-testid="stMetricValue"] div { color: #000000 !important; }
    div[data-testid="stMetricDelta"] div { color: #000000 !important; }

    /* Traffic Light Box Styles */
    .traffic-red {
        background-color: #ffe6e6; border: 2px solid #ff4d4d; 
        padding: 20px; border-radius: 10px; text-align: center; color: #cc0000;
    }
    .traffic-yellow {
        background-color: #fff3cd; border: 2px solid #ffc107; 
        padding: 20px; border-radius: 10px; text-align: center; color: #856404;
    }
    .traffic-green {
        background-color: #d4edda; border: 2px solid #28a745; 
        padding: 20px; border-radius: 10px; text-align: center; color: #155724;
    }
    
    /* Header Black Text Fix */
    h1, h2, h3 { color: inherit; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 🗣️ LANGUAGE & TRANSLATIONS
# ------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "welcome": "Welcome to Kisan Smart Farm",
        "enter_details": "Enter your farm details to get smart advisory.",
        "generate": "🚀 Generate Dashboard",
        "tab_live": "🌦️ Weather & Advisory",
        "tab_crop": "🌾 Crop Health",
        "tab_chat": "💬 Kisan Assistant",
        "tab_log": "📝 Farm Log",
    },
    "Hindi": {
        "welcome": "किसान स्मार्ट फार्म में आपका स्वागत है",
        "enter_details": "स्मार्ट सलाह प्राप्त करने के लिए अपना विवरण दर्ज करें।",
        "generate": "🚀 डैशबोर्ड बनाएं",
        "tab_live": "🌦️ मौसम और सलाह",
        "tab_crop": "🌾 फसल स्वास्थ्य",
        "tab_chat": "💬 किसान सहायक",
        "tab_log": "📝 खेती डायरी",
    },
    "Punjabi": {
        "welcome": "ਕਿਸਾਨ ਸਮਾਰਟ ਫਾਰਮ ਵਿੱਚ ਜੀ ਆਇਆਂ ਨੂੰ",
        "enter_details": "ਸਮਾਰਟ ਸਲਾਹ ਲੈਣ ਲਈ ਆਪਣੇ ਖੇਤ ਦੇ ਵੇਰਵੇ ਭਰੋ।",
        "generate": "🚀 ਡੈਸ਼ਬੋਰਡ ਤਿਆਰ ਕਰੋ",
        "tab_live": "🌦️ ਮੌਸਮ ਅਤੇ ਸਲਾਹ",
        "tab_crop": "🌾 ਫਸਲ ਦੀ ਸਿਹਤ",
        "tab_chat": "💬 ਕਿਸਾਨ ਸਹਾਇਕ",
        "tab_log": "📝 ਖੇਤੀ ਡਾਇਰੀ",
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
        "critical_rain_stage": "Flowering"
    },
    "Rice (Kharif)": {
        "duration_days": 120,
        "stages": [
            (0, 15, "🌱 Seedling"), (16, 45, "🌿 Tillering"),
            (46, 75, "🌸 Panicle Initiation"), (76, 105, "🌾 Grain Filling"),
            (106, 120, "🚜 Harvesting")
        ],
        "critical_rain_stage": "Harvesting"
    },
    "Cotton": {
        "duration_days": 160,
        "stages": [
            (0, 20, "🌱 Germination"), (21, 60, "🌿 Vegetative"),
            (61, 100, "🌸 Flowering"), (101, 140, "☁️ Boll Bursting"),
            (141, 160, "🚜 Picking")
        ],
        "critical_rain_stage": "Boll Bursting"
    }
}

CITY_COORDS = {
    "Mohali, PB": (30.7046, 76.7179),
    "Ludhiana, PB": (30.9010, 75.8573),
    "Delhi, NCR": (28.6139, 77.2090),
    "Bathinda, PB": (30.2109, 74.9455)
}

# ------------------------------------------------------
# ⚙️ LOAD RESOURCES
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
# 🌍 API & PREDICTION LOGIC
# ------------------------------------------------------
def fetch_weather_and_predict(city):
    """Fetches history + live data and runs LSTM prediction."""
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
        
        # Get Current Index
        now = pd.Timestamp.now().floor('h')
        df['diff'] = abs(df['time'] - now)
        idx = df['diff'].idxmin()
        
        # Current Weather Dict
        current_weather = {
            "temp": df.loc[idx, "temp"], "hum": df.loc[idx, "hum"],
            "wind": df.loc[idx, "wind"], "cloud": df.loc[idx, "cloud"]
        }
        
        # --- PREPARE FEATURES FOR LSTM ---
        def val(i, c): return df.loc[i, c] if i >= 0 else 0.0
        
        # STRICT ORDER for Scaler (11 Features)
        features = [
            val(idx-1, "rain"), val(idx-3, "rain"), val(idx-7, "rain"), # Lags
            val(idx-1, "hum"), val(idx-1, "wind"), val(idx-1, "temp"),
            df.loc[max(0, idx-2):idx, "rain"].mean(), # Rolling 3
            df.loc[max(0, idx-2):idx, "hum"].mean(),
            df.loc[max(0, idx-2):idx, "cloud"].mean(),
            dt.datetime.now().month, dt.datetime.now().weekday()
        ]
        
        X_raw = np.array(features).reshape(1, -1)
        X_scaled = feature_scaler.transform(X_raw).reshape(1, 1, 11)
        
        # FIX FOR 1 vs 11 FEATURE ERROR
        if model.input_shape[-1] == 1:
            X_scaled = X_scaled[:, :, 0:1]
            
        y = model.predict(X_scaled)
        pred_val = max(0.0, float(target_scaler.inverse_transform(y)[0][0]))
        
        return current_weather, pred_val

    except Exception as e:
        st.error(f"API/Prediction Error: {e}")
        return None, 0.0

# ------------------------------------------------------
# 🤖 CHATBOT BRAIN
# ------------------------------------------------------
def get_bot_response(msg, weather, crop_data):
    msg = msg.lower()
    rain = weather.get('pred_rain', 0.0)
    crop = crop_data.get('crop', 'Crop')
    
    if "irrigate" in msg or "water" in msg:
        if rain > 5.0: return f"🔴 **No!** Heavy rain ({rain:.1f}mm) is coming. Don't waste water."
        if rain > 1.0: return f"🟡 **Wait.** Light rain ({rain:.1f}mm) expected. Check soil first."
        return "🟢 **Yes.** Weather is clear. Good time to irrigate."
        
    if "spray" in msg or "pesticide" in msg:
        wind = weather.get('wind', 0)
        if rain > 0.5: return "🔴 **Don't Spray.** Rain will wash it away."
        if wind > 15: return "⚠️ **Caution.** High wind might cause drift."
        return "🟢 **Safe to spray.**"
        
    return f"I am tracking your **{crop}**. Ask me 'Should I irrigate?' or 'Can I spray?'"

# ------------------------------------------------------
# 🖥️ UI LAYOUTS
# ------------------------------------------------------

if "page" not in st.session_state: st.session_state.page = "landing"
if "crop_data" not in st.session_state: st.session_state.crop_data = {}
if "last_weather" not in st.session_state: st.session_state.last_weather = {}

# --- SIDEBAR (Global) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=80)
lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi", "Punjabi"])
t = TRANSLATIONS[lang]

# --- PAGE 1: LANDING ---
def render_landing():
    st.markdown(f"<h1 style='text-align: center;'>{t['welcome']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{t['enter_details']}</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("onboard"):
            name = st.text_input("Farmer Name", "Ram Singh")
            crop = st.selectbox("Select Crop / फसल", list(CROP_INFO.keys()))
            date = st.date_input("Sowing Date / बुवाई की तारीख", dt.date.today() - dt.timedelta(days=30))
            city = st.selectbox("Location / स्थान", list(CITY_COORDS.keys()))
            
            if st.form_submit_button(t["generate"]):
                st.session_state.crop_data = {"name": name, "crop": crop, "sowing_date": date, "city": city}
                st.session_state.page = "dashboard"
                st.rerun()

# --- PAGE 2: DASHBOARD ---
def render_dashboard():
    data = st.session_state.crop_data
    
    c1, c2 = st.columns([3, 1])
    c1.title(f"🚜 {data['name']}'s Farm")
    if c2.button("🔄 Change Crop"):
        st.session_state.page = "landing"
        st.rerun()

    weather, pred_rain = fetch_weather_and_predict(data["city"])
    if weather: 
        weather['pred_rain'] = pred_rain
        st.session_state.last_weather = weather
    
    crop_conf = CROP_INFO[data["crop"]]
    days_age = (dt.date.today() - data["sowing_date"]).days
    
    current_stage = "Unknown"
    for s, e, n in crop_conf["stages"]:
        if s <= days_age <= e: current_stage = n; break
    if days_age > crop_conf["duration_days"]: current_stage = "Harvest Ready"

    # --- KPI ROW (Black text forced via CSS) ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🌱 Age", f"{days_age} Days")
    k2.metric("📍 Stage", current_stage)
    days_left = max(0, crop_conf["duration_days"] - days_age)
    k3.metric("🚜 Harvest In", f"{days_left} Days")
    k4.metric("💧 Next Water", "2 Days" if pred_rain < 2 else "Not Needed")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([t["tab_live"], t["tab_crop"], t["tab_chat"], t["tab_log"]])

    with tab1:
        st.subheader("Live Advisory")
        if pred_rain > 5.0:
            st.markdown(f"""<div class='traffic-red'><h2>🚫 STOP</h2><p>Heavy Rain ({pred_rain:.1f}mm). Do not spray or irrigate.</p></div>""", unsafe_allow_html=True)
        elif pred_rain > 0.5:
            st.markdown(f"""<div class='traffic-yellow'><h2>⚠️ CAUTION</h2><p>Light Rain ({pred_rain:.1f}mm). Delay sensitive tasks.</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='traffic-green'><h2>🟢 GO</h2><p>Clear weather. Safe for farm operations.</p></div>""", unsafe_allow_html=True)

        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        if weather:
            m1.metric("Rainfall", f"{pred_rain:.1f} mm")
            m2.metric("Humidity", f"{weather['hum']}%")
            m3.metric("Wind", f"{weather['wind']} km/h")
            m4.metric("Temp", f"{weather['temp']} °C")

    with tab2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Growth Trajectory")
            

[Image of crop growth stages timeline]

            x = list(range(0, crop_conf["duration_days"] + 1, 5))
            y = [100 / (1 + np.exp(-0.1 * (i - crop_conf["duration_days"]/2))) for i in x]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, name="Ideal Growth", line=dict(color='green', dash='dot')))
            curr_y = 100 / (1 + np.exp(-0.1 * (days_age - crop_conf["duration_days"]/2)))
            fig.add_trace(go.Scatter(x=[days_age], y=[curr_y], mode='markers', marker=dict(color='red', size=15), name="You are here"))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Days", yaxis_title="% Growth")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Visual Guide")
            # Image Reference for stages
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Wheat_growth_stages.png/320px-Wheat_growth_stages.png", caption="Wheat Stages Reference")
            st.info(f"Current Phase: **{current_stage}**")

    with tab3:
        st.subheader("🤖 Kisan Assistant")
        if "messages" not in st.session_state: 
            st.session_state.messages = [{"role": "assistant", "content": "Sat Sri Akal! Ask me about irrigation or spraying."}]
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            resp = get_bot_response(prompt, st.session_state.last_weather, st.session_state.crop_data)
            st.chat_message("assistant").write(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})

    with tab4:
        st.subheader("📖 Farm Journal")
        note = st.text_area("Add a note", placeholder="e.g., Applied Urea today...")
        if st.button("Save Note"):
            st.success("Note saved to local log.")

if st.session_state.page == "landing":
    render_landing()
else:
    render_dashboard()
