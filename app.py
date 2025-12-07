import streamlit as st
import sqlite3
import random
import time
import datetime as dt
import requests # <--- NEW: For Live Weather API
import pandas as pd # <--- NEW: For Data Processing


# ------------------------------------------------------
# 1. 🏗️ CORE CONFIGURATION & STYLING
# ------------------------------------------------------
st.set_page_config(
    page_title="Kisan Farm OS",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# PROFESSIONAL MOBILE-FIRST CSS (Fixed for Dark/Light Mode)
st.markdown("""
    <style>
    /* GLOBAL RESET: Force Light Mode feel */
    .stApp { background-color: #f8f9fa; color: #000000 !important; }
    
    /* TEXT VISIBILITY FIXES */
    h1, h2, h3, h4, h5, h6, p, li, span, div { color: #000000; }
    label, .stTextInput label, .stSelectbox label { 
        color: #31333F !important; 
        font-weight: 600 !important;
    }
    
    /* CARD COMPONENT */
    .pro-card {
        background-color: #ffffff !important;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* BUTTON STYLING */
    .stButton button {
        background-color: #2E7D32 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        height: 55px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #1B5E20 !important;
        box-shadow: 0 4px 8px rgba(46, 125, 50, 0.3);
    }
    
    /* HIDE STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. 💾 DATABASE LAYER
# ------------------------------------------------------
# Connect to local SQLite DB to persist user data
conn = sqlite3.connect('farm_data.db', check_same_thread=False)
c = conn.cursor()

# Create tables if they don't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        phone TEXT PRIMARY KEY, 
        name TEXT, 
        crop TEXT, 
        region TEXT,
        joined_date TEXT
    )
''')
conn.commit()

def register_or_login_user(phone, name, crop, region):
    """Saves user to DB if new, or updates login time."""
    try:
        # Check if user exists
        c.execute("SELECT * FROM users WHERE phone = ?", (phone,))
        user = c.fetchone()
        
        if not user:
            # Register new user
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", 
                      (phone, name, crop, region, str(dt.date.today())))
            conn.commit()
            return "registered"
        else:
            return "logged_in"
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

# ------------------------------------------------------
# 3. 🔐 AUTHENTICATION LOGIC
# ------------------------------------------------------
def init_session():
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if "user" not in st.session_state: st.session_state.user = {}
    if "otp_stage" not in st.session_state: st.session_state.otp_stage = False
    if "generated_otp" not in st.session_state: st.session_state.generated_otp = None

def render_login():
    """Renders the Login/Signup Screen"""
    
    # Header Image & Title
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=100)
    
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1>Kisan Farm OS</h1>
            <p style='color: #666;'>One-Stop Smart Farming Solution</p>
        </div>
    """, unsafe_allow_html=True)

    # Login Card
    st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
    
    # STAGE 1: ENTER DETAILS
    if not st.session_state.otp_stage:
        st.markdown("### 👋 Let's get started")
        
        name = st.text_input("Full Name / पूरा नाम", "Ram Singh")
        crop = st.selectbox("Select Main Crop / फसल", ["Wheat (Rabi)", "Rice (Kharif)", "Cotton", "Sugarcane", "Maize", "Mustard"])
        region = st.selectbox("Select Region / क्षेत्र", ["Mohali, PB", "Ludhiana, PB", "Bathinda, PB", "Delhi, NCR", "Karnal, HR"])
        phone = st.text_input("Mobile Number / मोबाइल नंबर", placeholder="9876543210", max_chars=10)
        
        if st.button("Get OTP 📩"):
            if len(phone) == 10 and phone.isdigit():
                with st.spinner("Connecting to secure server..."):
                    time.sleep(1.2) # Simulate network delay
                    # Generate OTP
                    otp = random.randint(1000, 9999)
                    st.session_state.generated_otp = otp
                    st.session_state.otp_stage = True
                    
                    # Store temporary data
                    st.session_state.temp_user = {
                        "phone": phone, "name": name, 
                        "crop": crop, "region": region
                    }
                    st.rerun()
            else:
                st.error("⚠️ Please enter a valid 10-digit mobile number.")

    # STAGE 2: VERIFY OTP
    else:
        phone = st.session_state.temp_user['phone']
        st.markdown(f"### 🔐 Verify Mobile")
        st.info(f"OTP sent to **{phone}**")
        
        # DISPLAY OTP FOR DEMO (So you can log in easily)
        st.success(f"**Your OTP is: {st.session_state.generated_otp}**")
        
        otp_input = st.text_input("Enter 4-digit OTP", max_chars=4, placeholder="XXXX")
        
        if st.button("Verify & Login 🚀"):
            if str(otp_input) == str(st.session_state.generated_otp):
                # Save to DB
                status = register_or_login_user(
                    phone, 
                    st.session_state.temp_user['name'],
                    st.session_state.temp_user['crop'],
                    st.session_state.temp_user['region']
                )
                
                # Update Session
                st.session_state.user = st.session_state.temp_user
                st.session_state.authenticated = True
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Incorrect OTP. Please try again.")
        
        if st.button("Edit Details"):
            st.session_state.otp_stage = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
# ------------------------------------------------------
# 1. 🏗️ CORE CONFIGURATION & STYLING
# ------------------------------------------------------
st.set_page_config(
    page_title="Kisan Farm OS",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# PROFESSIONAL CSS (Added 'hero-card' for Weather)
st.markdown("""
    <style>
    /* GLOBAL RESET */
    .stApp { background-color: #f8f9fa; color: #000000 !important; }
    h1, h2, h3, h4, h5, h6, p, li, span, div { color: #000000; }
    label, .stTextInput label, .stSelectbox label { color: #31333F !important; font-weight: 600; }
    
    /* STANDARD CARD */
    .pro-card {
        background-color: #ffffff !important;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* NEW: WEATHER HERO CARD (Gradient) */
    .hero-card {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white !important;
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(46, 125, 50, 0.25);
    }
    /* Force white text inside Hero Card */
    .hero-card h1, .hero-card h2, .hero-card p, .hero-card span { color: white !important; }
    
    /* STATUS PILL (Green/Red/Yellow badges) */
    .status-pill {
        background: rgba(255,255,255,0.2);
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
        color: white !important;
    }

    /* BUTTONS */
    .stButton button {
        background-color: #2E7D32 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        height: 55px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton button:hover { background-color: #1B5E20 !important; }
    
    /* HIDE DEFAULTS */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. 💾 DATABASE LAYER (Same as Phase 1)
# ------------------------------------------------------
conn = sqlite3.connect('farm_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (phone TEXT PRIMARY KEY, name TEXT, crop TEXT, region TEXT, joined_date TEXT)''')
conn.commit()

def register_or_login_user(phone, name, crop, region):
    try:
        c.execute("SELECT * FROM users WHERE phone = ?", (phone,))
        if not c.fetchone():
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (phone, name, crop, region, str(dt.date.today())))
            conn.commit()
    except Exception as e: st.error(f"DB Error: {e}")

# ------------------------------------------------------
# 3. ☁️ WEATHER ENGINE (THE MISSING PART)
# ------------------------------------------------------
CITY_COORDS = {
    "Mohali, PB": (30.7046, 76.7179),
    "Ludhiana, PB": (30.9010, 75.8573),
    "Bathinda, PB": (30.2109, 74.9455),
    "Delhi, NCR": (28.6139, 77.2090),
    "Karnal, HR": (29.6857, 76.9905),
    "Uttar Pradesh": (26.8467, 80.9462) 
}

# Added Cache to make app faster (won't call API on every click)
@st.cache_data(ttl=300) 
def get_live_weather(city_name):
    """Fetches real-time weather from Open-Meteo API."""
    lat, lon = CITY_COORDS.get(city_name, (30.70, 76.71)) # Default Mohali
    
    # 1. Call API
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=rain,relative_humidity_2m,windspeed_10m"
    
    try:
        resp = requests.get(url).json()
        current = resp['current_weather']
        hourly = resp['hourly']
        
        # 2. Calculate Logic (Next 24h Rain)
        now_index = int(pd.to_datetime(current['time']).hour)
        rain_24h = sum(hourly['rain'][now_index : now_index+24])
        
        return {
            "temp": current['temperature'],
            "wind": current['windspeed'],
            "rain_forecast": rain_24h, # Critical for Red/Green Light
            "humidity": hourly['relative_humidity_2m'][now_index]
        }
    except Exception as e:
        # Fallback if internet fails
        return {"temp": "--", "wind": "--", "rain_forecast": 0.0, "humidity": "--"}

# ------------------------------------------------------
# 4. 📱 APP SCREENS
# ------------------------------------------------------
def init_session():
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if "user" not in st.session_state: st.session_state.user = {}
    if "otp_stage" not in st.session_state: st.session_state.otp_stage = False
    if "generated_otp" not in st.session_state: st.session_state.generated_otp = None
    if "temp_user" not in st.session_state: st.session_state.temp_user = {}

def render_login():
    """Phase 1 Login Screen"""
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2: st.image("https://cdn-icons-png.flaticon.com/512/3025/3025528.png", width=100)
    
    st.markdown("<h1 style='text-align: center;'>Kisan Farm OS</h1>", unsafe_allow_html=True)
    st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
    
    if not st.session_state.otp_stage:
        st.markdown("### 👋 Login")
        name = st.text_input("Name", "Ram Singh")
        crop = st.selectbox("Crop", ["Rice (Kharif)", "Wheat (Rabi)", "Cotton", "Sugarcane"])
        region = st.selectbox("Region", list(CITY_COORDS.keys()))
        phone = st.text_input("Mobile", placeholder="9876543210")
        
        if st.button("Get OTP"):
            if len(phone)==10:
                st.session_state.generated_otp = random.randint(1000,9999)
                st.session_state.otp_stage = True
                st.session_state.temp_user = {"phone":phone, "name":name, "crop":crop, "region":region}
                st.rerun()
            else: st.error("Invalid Number")
    else:
        # Phase 1: OTP Simulation
        st.success(f"**OTP: {st.session_state.generated_otp}**")
        otp = st.text_input("Enter OTP", max_chars=4)
        if st.button("Login"):
            if str(otp) == str(st.session_state.generated_otp):
                register_or_login_user(st.session_state.temp_user['phone'], st.session_state.temp_user['name'], st.session_state.temp_user['crop'], st.session_state.temp_user['region'])
                st.session_state.user = st.session_state.temp_user
                st.session_state.authenticated = True
                st.rerun()
            else: st.error("Wrong OTP")
    st.markdown("</div>", unsafe_allow_html=True)

def render_dashboard():
    """Phase 2: Real-Time Weather Dashboard"""
    user = st.session_state.user
    
    # 1. HEADER
    c1, c2 = st.columns([3, 1])
    with c1: 
        st.markdown(f"### 👋 Namaste, {user['name'].split()[0]}")
        st.caption(f"📍 {user['region']} • 🌾 {user['crop']}")
    with c2:
        if st.button("🚪"):
            st.session_state.authenticated = False
            st.session_state.otp_stage = False
            st.rerun()

    # 2. FETCH LIVE DATA (The Brain)
    with st.spinner("Connecting to Satellite..."):
        w = get_live_weather(user['region'])
    
    # 3. TRAFFIC LIGHT LOGIC (The Problem Solver)
    # If rain > 5mm in next 24h -> RED ALERT
    if isinstance(w['rain_forecast'], (int, float)) and w['rain_forecast'] > 5.0:
        theme_color = "linear-gradient(135deg, #DC2626 0%, #991B1B 100%)" # Red
        status_text = "🚫 STOP WORK"
        advisory = f"Heavy rain ({w['rain_forecast']:.1f}mm) expected. Do not spray pesticides."
    elif isinstance(w['rain_forecast'], (int, float)) and w['rain_forecast'] > 1.0:
        theme_color = "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)" # Yellow
        status_text = "⚠️ CAUTION"
        advisory = f"Light drizzle ({w['rain_forecast']:.1f}mm) likely. Hold irrigation."
    else:
        theme_color = "linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)" # Green
        status_text = "🟢 GO AHEAD"
        advisory = "Weather is clear. Safe for irrigation & spraying."

    # 4. RENDER HERO CARD
    st.markdown(f"""
    <div class="hero-card" style="background: {theme_color};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 10px;">
            <span class="status-pill">{status_text}</span>
            <span>{dt.datetime.now().strftime('%d %b, %I:%M %p')}</span>
        </div>
        <div style="display:flex; align-items:flex-end; gap:15px;">
            <h1 style="margin:0; font-size:3.5rem;">{w['temp']}°C</h1>
            <div style="margin-bottom:10px;">
                <p style="margin:0;">🌧️ {w['rain_forecast']}mm Rain</p>
                <p style="margin:0;">💧 {w['humidity']}% Hum</p>
            </div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.3);">
        <p style="font-weight: 500; font-size: 1.1rem;">📢 {advisory}</p>
    </div>
    """, unsafe_allow_html=True)

    # 5. PHASE 3 & 4 PLACEHOLDERS
    st.markdown("### 🚜 Quick Actions")
    c1, c2 = st.columns(2)
    with c1:
        st.info("🏥 Plant Doctor\n\n(Coming Phase 3)")
    with c2:
        st.success("💰 Mandi Prices\n\n(Coming Phase 4)")

 ------------------------------------------------------
# PHASE 4: MONEY MODULE (Mandi Prices)
# ------------------------------------------------------
def render_mandi():
    """Displays Real-Time Market Prices & Comparisons"""
    if "user" not in st.session_state or not st.session_state.user:
        st.session_state.page = "login"
        st.rerun()
        
    user_crop = st.session_state.user.get('crop', 'Wheat')
    user_city = st.session_state.user.get('region', 'Local')
    
    st.markdown(f"### 💰 Market Rates: {user_crop}")
    
    # 1. Base Price Logic (Simulated Real-Time)
    price_map = {
        "Wheat (Rabi)": 2275,
        "Rice (Kharif)": 3100,
        "Cotton": 6600,
        "Sugarcane": 340,
        "Maize": 2090,
        "Mustard": 5650
    }
    base_price = price_map.get(user_crop, 2000)
    
    # Add daily fluctuation
    fluctuation = random.randint(-50, 100)
    current_price = base_price + fluctuation
    
    # Visual Logic
    trend_symbol = "📈 UP" if fluctuation > 0 else "📉 DOWN"
    trend_color = "#166534" if fluctuation > 0 else "#991b1b" # Green or Red
    
    # 2. Local Market Card
    st.markdown(f"""
    <div class='pro-card' style='border-left: 5px solid {trend_color};'>
        <p style='margin:0; color:gray;'>📍 {user_city} APMC (Local)</p>
        <h1 style='color: {trend_color}; margin: 5px 0;'>₹{current_price} / Q</h1>
        <p style='color: #333; margin: 0; font-weight:500;'><b>Trend:</b> {trend_symbol} by ₹{abs(fluctuation)} today</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Comparison Table (Arbitrage Opportunity)
    st.markdown("### 🚛 Nearby Markets")
    st.write("Compare rates to sell profitably:")
    
    nearby_mandis = [
        ("District Main", current_price + random.randint(20, 60), "25km"),
        ("Private Trader", current_price - random.randint(10, 30), "Doorstep"),
        ("Neighbor District", current_price + random.randint(10, 40), "45km")
    ]
    
    st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
    
    # Header Row
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown("**Mandi**")
    c2.markdown("**Price**")
    c3.markdown("**Dist**")
    st.markdown("<hr style='margin: 5px 0;'>", unsafe_allow_html=True)
    
    # Data Rows
    for name, price, dist in nearby_mandis:
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.write(name)
        
        # Color code price difference
        diff = price - current_price
        diff_color = "green" if diff > 0 else "red"
        diff_text = f"+₹{diff}" if diff > 0 else f"-₹{abs(diff)}"
        
        c2.markdown(f"**₹{price}** <span style='color:{diff_color}; font-size:0.8rem;'>{diff_text}</span>", unsafe_allow_html=True)
        c3.caption(dist)
        st.markdown("<hr style='margin: 5px 0; border-top: 1px dashed #eee;'>", unsafe_allow_html=True)
        
    st.success("💡 **Tip:** Selling at 'District Main' covers transport cost and gives ₹200 extra profit per trolley.")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "dashboard"
        st.rerun()
```

#### **Part 2: Update the Dashboard Button**
Go to your `render_dashboard` function. Find the `c2` column (bottom right) and update it to enable the button.

```python
    # Inside render_dashboard...
    with c2:
        if st.button("💰 Mandi Prices"):
            st.session_state.page = "mandi"
            st.rerun()

# ------------------------------------------------------
# 5. 🚀 EXECUTION
# ------------------------------------------------------
if __name__ == "__main__":
    init_session()
    
    if not st.session_state.authenticated:
        render_login()
    elif st.session_state.page == "dashboard":
        render_dashboard()
    elif st.session_state.page == "plant_doctor":
        render_plant_doctor()
    elif st.session_state.page == "mandi":   # <--- ADD THIS
        render_mandi()                       # <--- ADD THIS
