import streamlit as st
import pandas as pd
import requests
import json
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

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">SkyGuard AI: Real-Time Fleet Risk & Delay Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Combining live flight telemetry with Gemini-powered risk extraction from unstructured safety logs.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Setup & API Configuration
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not gemini_api_key:
    st.sidebar.warning("⚠️ Please enter a valid Gemini API Key to enable AI extractions.")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Context")
st.sidebar.info("""
**Target Role:** Fleet PM / Aviation Business Analyst
**Data Sources:** 
- OpenSky Network REST API (Live Flights)
- NASA ASRS Maintenance Narratives
**AI Engine:** Google Gemini (Structured JSON)
""")

# ---------------------------------------------------------
# Data Fetchers: Live OpenSky Telemetry
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_live_flights():
    """Fetch live aircraft vectors from the OpenSky Network API."""
    url = "https://opensky-network.org/api/states/all"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            states = data.get('states', [])
            
            # Convert raw vector array into structured DataFrame
            parsed_flights = []
            for s in states[:30]:  # Limit to 30 active flights for performance
                callsign = s[1].strip() if s[1] else "UNKNOWN"
                origin_country = s[2] if s[2] else "N/A"
                longitude = s[5]
                latitude = s[6]
                baro_altitude = s[7] if s[7] else 0
                velocity = s[9] if s[9] else 0

                if latitude and longitude:
                    parsed_flights.append({
                        "Callsign": callsign,
                        "Country": origin_country,
                        "lat": latitude,
                        "lon": longitude,
                        "Altitude_m": baro_altitude,
                        "Velocity_m_s": velocity
                    })
            return pd.DataFrame(parsed_flights)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching OpenSky data: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# Sample NASA ASRS / Synthetic Incident Narratives
# ---------------------------------------------------------
SAMPLE_LOGS = {
    "LOG-8801": "Pre-flight inspection on Aircraft N737AA revealed a minor hydraulic leak around the nose landing gear actuator seal. Pressure dropped 8% over 2 hours. Seal replacement recommended prior to next dispatch.",
    "LOG-8802": "During flight cruise, Avionics Bay Fan 2 reported intermittent telemetry bus communication drops. Diagnostic test indicates potential wiring harness degradation. High risk of avionics overheating if unresolved.",
    "LOG-8803": "Cabin air quality check noted low recirculation flow on Pack 1. Filter replaced and flow restored to normal limits during scheduled turnaround. No schedule delay anticipated.",
    "LOG-8804": "Engine 1 fuel control unit displayed momentary vibration spike during takeoff roll. Maintenance inspection suggests sensor recalibration required. FAA AD compliance review needed before long-haul dispatch."
}

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
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
st.markdown("### 1. Live Telemetry & Fleet Overview")

flights_df = fetch_live_flights()

if not flights_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Tracked Flights", len(flights_df))
    col2.metric("Avg Fleet Altitude", f"{int(flights_df['Altitude_m'].mean())} m")
    col3.metric("Avg Fleet Velocity", f"{int(flights_df['Velocity_m_s'].mean())} m/s")
    col4.metric("Live Telemetry Status", "ONLINE", delta="OpenSky API")

    # Map Visualization
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
else:
    st.info("Loading live flight tracking vectors from OpenSky API...")

st.markdown("---")
st.markdown("### 2. Unstructured Maintenance Risk Ingestion")

col_log_sel, col_log_view = st.columns([1, 2])

with col_log_sel:
    selected_log_id = st.selectbox("Select Maintenance Log Narrative:", list(SAMPLE_LOGS.keys()))
    raw_log_text = SAMPLE_LOGS[selected_log_id]
    
    # Allow user to edit or paste custom text
    custom_log = st.text_area("Or Paste Custom Technician Log:", raw_log_text, height=120)
    analyze_btn = st.button("🚀 Analyze Risk with Gemini AI", use_container_width=True)

with col_log_view:
    st.subheader("Raw Narrative Source (Audit Trail)")
    st.info(custom_log)

# ---------------------------------------------------------
# AI Analysis Results Display
# ---------------------------------------------------------
if analyze_btn:
    if not gemini_api_key:
        st.error("Please provide a Gemini API Key in the sidebar to run the analysis.")
    else:
        with st.spinner("Extracting structured metrics via Gemini AI..."):
            risk_data = analyze_log_with_gemini(custom_log, gemini_api_key)

        if risk_data:
            st.success("Analysis Complete!")
            st.markdown("### 3. Executive Decision Support Output")

            m1, m2, m3, m4 = st.columns(4)
            
            # Dynamic Risk Badge Coloring
            severity = risk_data.get("Severity_Level", "Low")
            m1.metric("Subsystem Impacted", risk_data.get("Subsystem", "N/A"))
            m2.metric("Severity Score", f"{risk_data.get('Severity_Score', 0)} / 10", delta=severity, delta_color="inverse")
            m3.metric("Projected Delay", f"{risk_data.get('Estimated_Delay_Hours', 0)} Hours")
            m4.metric("FAA Compliance Flag", "YES" if risk_data.get("FAA_Compliance_Flag") else "NO")

            st.markdown("#### **AI Risk Summary for Project Managers:**")
            st.warning(f"**Action Required:** {risk_data.get('AI_Summary', 'N/A')}")

            # Human in the Loop Note
            st.caption("🛡️ **Responsible AI Guardrail:** AI risk extractions are designed for decision-support prioritization. Certified engineering sign-off is required before altering aircraft dispatch state.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("SkyGuard AI Capstone | Developed with Streamlit, Google Gemini, and OpenSky Network.")
