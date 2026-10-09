import streamlit as st
import pandas as pd
import requests
import json
import os
import plotly.express as px
from google import genai

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SkyGuard AI - Fleet Risk Intelligence",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">SkyGuard AI: Real-Time Fleet Risk & Delay Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Combining live flight telemetry with Gemini-powered risk extraction from NASA ASRS safety logs.</div>', unsafe_allow_html=True)
# ---------------------------------------------------------
# Sidebar Configuration (Gemini + OpenSky Credentials)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

# Gemini API Key
default_gemini = os.environ.get("GEMINI_API_KEY", "")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", value=default_gemini, type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Live Telemetry Auth (OpenSky)")

# OpenSky Credentials (Pulled from Environment/Secrets if present)
default_opensky_user = os.environ.get("OPENSKY_USER", "")
default_opensky_pass = os.environ.get("OPENSKY_PASS", "")

opensky_user = st.sidebar.text_input("OpenSky Username", value=default_opensky_user)
opensky_pass = st.sidebar.text_input("OpenSky Password", value=default_opensky_pass, type="password")


# ---------------------------------------------------------
# Data Fetcher: Authenticated OpenSky Live Telemetry
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_flights(username="", password=""):
    """Fetch live aircraft vectors across CONUS using Basic Auth to bypass cloud IP blocks."""
    url = "https://opensky-network.org/api/states/all"

    params = {
        "lamin": 24.396308,   # Min Latitude (Southern US)
        "lamax": 49.384358,   # Max Latitude (Northern US)
        "lomin": -125.000000, # Min Longitude (West Coast)
        "lomax": -66.934570   # Max Longitude (East Coast)
    }

    headers = {
        "User-Agent": "SkyGuardAI-FleetMonitor/1.0"
    }

    # Configure Basic Auth if credentials exist
    auth = (username, password) if (username and password) else None

    try:
        response = requests.get(url, params=params, headers=headers, auth=auth, timeout=8)
        if response.status_code == 200:
            data = response.json()
            states = data.get('states', [])

            if states:
                parsed_flights = []
                for s in states[:30]:
                    callsign = s[1].strip() if (s[1] and s[1].strip()) else f"FLT-{s[0][:4].upper()}"
                    origin_country = s[2] if s[2] else "N/A"
                    longitude = s[5]
                    latitude = s[6]
                    baro_altitude = s[7] if s[7] is not None else 0
                    velocity = s[9] if s[9] is not None else 0

                    if latitude and longitude:
                        parsed_flights.append({
                            "Callsign": callsign,
                            "Country": origin_country,
                            "lat": latitude,
                            "lon": longitude,
                            "Altitude_m": int(baro_altitude),
                            "Velocity_m_s": int(velocity)
                        })

                if parsed_flights:
                    auth_label = "Authenticated" if auth else "Anonymous"
                    return pd.DataFrame(parsed_flights), f"OpenSky API (Live US Airspace - {auth_label})"
        else:
            st.sidebar.warning(f"OpenSky Status Code: {response.status_code}")
    except Exception as e:
        st.sidebar.info(f"Telemetry Network Note: {e}")

    # Fallback Telemetry
    fallback_flights = pd.DataFrame([
        {"Callsign": "AAL102", "Country": "United States", "lat": 32.7767, "lon": -96.7970, "Altitude_m": 10500, "Velocity_m_s": 240},
        {"Callsign": "DAL451", "Country": "United States", "lat": 33.7490, "lon": -84.3880, "Altitude_m": 11200, "Velocity_m_s": 255},
        {"Callsign": "UAL890", "Country": "United States", "lat": 41.8781, "lon": -87.6298, "Altitude_m": 9800, "Velocity_m_s": 230},
        {"Callsign": "BAW178", "Country": "United Kingdom", "lat": 39.8561, "lon": -104.6737, "Altitude_m": 10800, "Velocity_m_s": 248},
        {"Callsign": "AFR012", "Country": "France", "lat": 36.0840, "lon": -115.1537, "Altitude_m": 11500, "Velocity_m_s": 260}
    ])
    return fallback_flights, "Cached Flight Snapshot (Offline Fallback)"


# ---------------------------------------------------------
# NASA ASRS Data Ingestion (CSV / Local Load)
# ---------------------------------------------------------
@st.cache_data
def load_nasa_asrs_data():
    """Loads NASA ASRS dataset from local CSV."""
    csv_file = "asrs_data.csv"
    if os.path.exists(csv_file):
        try:
            return pd.read_csv(csv_file)
        except Exception:
            pass

    fallback_data = [
        {"ACN": "ASRS-192841", "Subsystem_Hint": "Hydraulics", "Narrative": "During preflight inspection on B737-800, crew discovered hydraulic fluid leaking from nose gear steering metering valve. Leak rate measured at 5 drops per minute. Component replaced by maintenance, resulting in 2.5 hour departure delay."},
        {"ACN": "ASRS-184920", "Subsystem_Hint": "Avionics", "Narrative": "Aircraft experienced intermittent loss of GPS 1 signal during initial descent. Maintenance team ran diagnostic checks on flight management computer and discovered loose antenna coupling connection. Re-seated cable and verified signal integrity."}
    ]
    return pd.DataFrame(fallback_data)

# ---------------------------------------------------------
# AI Processing Function (Google Gemini)
# ---------------------------------------------------------
def analyze_log_with_gemini(log_text, api_key):
    """Parses unstructured log text into structured JSON metrics via Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an aviation safety and project management risk analyst.
        Analyze the following aircraft maintenance narrative and extract key metrics into JSON format.

        Narrative: "{log_text}"

        Return ONLY a JSON object with these exact keys:
        - "Subsystem": (e.g., Hydraulics, Avionics, Propulsion, Environmental)
        - "Severity_Level": (Low, Medium, High, Critical)
        - "Severity_Score": (Integer 1-10)
        - "Estimated_Delay_Hours": (Integer estimated delay)
        - "FAA_Compliance_Flag": (True or False)
        - "AI_Summary": (A 1-sentence summary for a Project Manager)
        """

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

# ---------------------------------------------------------
# Main Dashboard Layout
# ---------------------------------------------------------
st.markdown("### 1. Live Telemetry & Active Fleet Overview")

flights_df, telemetry_source = fetch_live_flights()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Tracked Flights", len(flights_df))
col2.metric("Avg Fleet Altitude", f"{int(flights_df['Altitude_m'].mean())} m")
col3.metric("Avg Fleet Velocity", f"{int(flights_df['Velocity_m_s'].mean())} m/s")
col4.metric("Live Telemetry Status", "ONLINE", delta=telemetry_source)

fig = px.scatter_mapbox(
    flights_df,
    lat="lat",
    lon="lon",
    hover_name="Callsign",
    hover_data=["Country", "Altitude_m", "Velocity_m_s"],
    zoom=1,
    height=350
)
fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 2. NASA ASRS Maintenance Risk Ingestion")

asrs_df = load_nasa_asrs_data()

col_log_sel, col_log_view = st.columns([1, 2])

with col_log_sel:
    selected_acn = st.selectbox("Select NASA ASRS Report (ACN Number):", asrs_df["ACN"].tolist())
    selected_row = asrs_df[asrs_df["ACN"] == selected_acn].iloc[0]
    raw_log_text = selected_row["Narrative"]

    custom_log = st.text_area("Or Inspect/Modify Narrative:", raw_log_text, height=130)
    analyze_btn = st.button("🚀 Analyze Risk with Gemini AI", use_container_width=True)

with col_log_view:
    st.subheader("Raw Narrative Source (Audit Trail)")
    st.info(custom_log)

# ---------------------------------------------------------
# AI Analysis Results Display
# ---------------------------------------------------------
if analyze_btn:
    if not gemini_api_key:
        st.error("Please provide a Gemini API Key in the sidebar or Colab Secrets.")
    else:
        with st.spinner("Extracting structured metrics via Gemini AI..."):
            risk_data = analyze_log_with_gemini(custom_log, gemini_api_key)

        if risk_data:
            st.success("Analysis Complete!")
            st.markdown("### 3. Executive Decision Support Output")

            m1, m2, m3, m4 = st.columns(4)
            severity = risk_data.get("Severity_Level", "Low")
            m1.metric("Subsystem Impacted", risk_data.get("Subsystem", "N/A"))
            m2.metric("Severity Score", f"{risk_data.get('Severity_Score', 0)} / 10", delta=severity, delta_color="inverse")
            m3.metric("Projected Delay", f"{risk_data.get('Estimated_Delay_Hours', 0)} Hours")
            m4.metric("FAA Compliance Flag", "YES" if risk_data.get("FAA_Compliance_Flag") else "NO")

            st.markdown("#### **AI Risk Summary for Project Managers:**")
            st.warning(f"**Action Required:** {risk_data.get('AI_Summary', 'N/A')}")

            st.caption("🛡️ **Responsible AI Guardrail:** AI risk extractions are designed for decision-support prioritization. Certified engineering sign-off is required before altering aircraft dispatch state.")

st.markdown("---")
st.caption("SkyGuard AI Capstone | Developed with Streamlit, Google Gemini, and OpenSky Network.")
