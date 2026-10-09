import streamlit as st
import pandas as pd
import requests
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from google import genai

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SkyGuard AI - Fleet Risk Intelligence",
    page_icon="✈️",
    layout="wide"
)

# Custom Executive CSS
st.markdown("""
    <style>
    /* Dark glassmorphism header & cards */
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-online { background-color: #064E3B; color: #34D399; }
    .badge-warning { background-color: #78350F; color: #FBBF24; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">SkyGuard AI: Fleet Risk & Operational Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time CONUS telemetry fusion with Gemini 3.8 safety log analytics for Fleet Project Managers.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration (Gemini + OpenSky Credentials)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration & Auth")

# Gemini API Key
default_gemini = os.environ.get("GEMINI_API_KEY", "")
gemini_api_key = st.sidebar.text_input("Gemini API Key", value=default_gemini, type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Live Telemetry Auth")

default_opensky_user = os.environ.get("OPENSKY_USER", "")
default_opensky_pass = os.environ.get("OPENSKY_PASS", "")

opensky_user = st.sidebar.text_input("OpenSky Username", value=default_opensky_user)
opensky_pass = st.sidebar.text_input("OpenSky Password", value=default_opensky_pass, type="password")


# ---------------------------------------------------------
# Data Fetcher: Authenticated OpenSky Live Telemetry
# ---------------------------------------------------------
@st.cache_data(ttl=15)
def fetch_live_flights(username="", password=""):
    """Fetch live aircraft vectors across CONUS using Basic Auth."""
    url = "https://opensky-network.org/api/states/all"
    
    params = {
        "lamin": 24.396308,   # Southern US
        "lamax": 49.384358,   # Northern US
        "lomin": -125.000000, # West Coast
        "lomax": -66.934570   # East Coast
    }
    
    headers = {"User-Agent": "SkyGuardAI-FleetMonitor/1.0"}
    auth = (username, password) if (username and password) else None

    try:
        response = requests.get(url, params=params, headers=headers, auth=auth, timeout=8)
        if response.status_code == 200:
            data = response.json()
            states = data.get('states', [])
            
            if states:
                parsed_flights = []
                for s in states[:40]:
                    callsign = s[1].strip() if (s[1] and s[1].strip()) else f"FLT-{s[0][:4].upper()}"
                    origin_country = s[2] if s[2] else "N/A"
                    longitude = s[5]
                    latitude = s[6]
                    baro_altitude = s[7] if s[7] is not None else 0
                    velocity = s[9] if s[9] is not None else 0
                    heading = s[10] if (len(s) > 10 and s[10] is not None) else 0
                    

                    if latitude and longitude:
                        parsed_flights.append({
                            "Callsign": callsign,
                            "Country": origin_country,
                            "lat": latitude,
                            "lon": longitude,
                            "Altitude_m": int(baro_altitude),
                            "Velocity_m_s": int(velocity),
                            "Heading": int(heading)  # True track angle in degrees
                        })
                
                if parsed_flights:
                    auth_label = "Authenticated" if auth else "Anonymous"
                    return pd.DataFrame(parsed_flights), f"OpenSky API ({auth_label})"
    except Exception:
        pass

    # Fallback Telemetry
    fallback_flights = pd.DataFrame([
        {"Callsign": "AAL102", "Country": "United States", "lat": 32.7767, "lon": -96.7970, "Altitude_m": 10500, "Velocity_m_s": 240},
        {"Callsign": "DAL451", "Country": "United States", "lat": 33.7490, "lon": -84.3880, "Altitude_m": 11200, "Velocity_m_s": 255},
        {"Callsign": "UAL890", "Country": "United States", "lat": 41.8781, "lon": -87.6298, "Altitude_m": 9800, "Velocity_m_s": 230},
        {"Callsign": "BAW178", "Country": "United Kingdom", "lat": 39.8561, "lon": -104.6737, "Altitude_m": 10800, "Velocity_m_s": 248},
        {"Callsign": "AFR012", "Country": "France", "lat": 36.0840, "lon": -115.1537, "Altitude_m": 11500, "Velocity_m_s": 260}
    ])
    return fallback_flights, "Cached Offline Fallback"


# ---------------------------------------------------------
# NASA ASRS Data Ingestion
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
    {
        "ACN": "ASRS-192841",
        "Subsystem_Hint": "Hydraulics",
        "Narrative": "During preflight inspection on B737-800, crew discovered hydraulic fluid leaking from nose gear steering metering valve. Leak rate measured at 5 drops per minute. Component replaced by maintenance, resulting in 2.5 hour departure delay."
    },
    {
        "ACN": "ASRS-184920",
        "Subsystem_Hint": "Avionics",
        "Narrative": "Aircraft experienced intermittent loss of GPS 1 signal during initial descent. Maintenance team ran diagnostic checks on flight management computer and discovered loose antenna coupling connection. Re-seated cable and verified signal integrity."
    },
    {
        "ACN": "ASRS-177301",
        "Subsystem_Hint": "Propulsion",
        "Narrative": "Engine 2 EGT indication fluctuated during climb out. Flight crew initiated non-normal checklist and leveled off. Maintenance inspection revealed faulty thermocouple wiring harness connection resulting in 1.8 hour turnaround delay."
    },
    {
        "ACN": "ASRS-195102",
        "Subsystem_Hint": "Flight Controls",
        "Narrative": "Inbound A320 reported slat asymmetry warning during approach configuration. Aircraft landed without incident. Line maintenance isolated fault to a malfunctioning slat actuator proximity switch. Unit replaced and system re-calibrated."
    },
    {
        "ACN": "ASRS-189443",
        "Subsystem_Hint": "Environmental",
        "Narrative": "Flight crew reported pack 1 overheat alert while at cruise altitude FL350. Non-normal checklist executed and pack turned off. On-ground inspection revealed clogged precooler heat exchanger core. Unit backflushed and tested."
    },
    {
        "ACN": "ASRS-193208",
        "Subsystem_Hint": "Landing Gear",
        "Narrative": "During landing rollout on B787-9, right main gear anti-skid caution light illuminated. Aircraft taxiing to gate under own power. Maintenance inspection identified damaged wheel speed sensor wiring harness. Sensor harness replaced."
    },
    {
        "ACN": "ASRS-181055",
        "Subsystem_Hint": "Electrical",
        "Narrative": "IDG 2 generator disconnected automatically during taxi out due to internal high oil temperature warning. Flight returned to gate. Integrated Drive Generator replaced per MEL, causing 3.2 hour schedule delay."
    },
    {
        "ACN": "ASRS-196021",
        "Subsystem_Hint": "Fuel Systems",
        "Narrative": "Preflight fuel balancing check showed fuel crossfeed valve failed to open when commanded from overhead panel. Actuator motor tested faulty and replaced by avionics crew prior to dispatch."
    },
    {
        "ACN": "ASRS-188319",
        "Subsystem_Hint": "Pneumatics",
        "Narrative": "Engine 1 bleed air valve failed closed during engine start checklist. APU bleed air used for crossbleed start after valve solenoids were inspected and reset by line maintenance."
    },
    {
        "ACN": "ASRS-190442",
        "Subsystem_Hint": "APU",
        "Narrative": "Auxiliary Power Unit auto-shutdown during passenger boarding due to low oil pressure fault code. APU oil level serviced and pressure transducer replaced, delaying departure by 1.1 hours."
    },
    {
        "ACN": "ASRS-183790",
        "Subsystem_Hint": "Avionics",
        "Narrative": "TCAS processor displayed fail flag intermittently during climb phase. Flight crew notified ATC and maintained manual visual separation. TCAS computer unit swapped by avionics line crew upon arrival."
    },
    {
        "ACN": "ASRS-194511",
        "Subsystem_Hint": "Hydraulics",
        "Narrative": "System B hydraulic quantity dropped to 62% during cruise. Flight crew monitored pressure with no further loss. Post-flight leak check identified weeping fitting on elevator power control unit."
    },
    {
        "ACN": "ASRS-186204",
        "Subsystem_Hint": "Fire Protection",
        "Narrative": "Cargo compartment loop A fire detector fault annunciated during pre-flight BIT test. Maintenance troubleshooting revealed corroded connector pin at forward cargo bay junction box."
    },
    {
        "ACN": "ASRS-191833",
        "Subsystem_Hint": "Propulsion",
        "Narrative": "N1 vibration monitor on Engine 1 registered elevated readings during high-altitude cruise. Fan blade inspection revealed fan blade dampener wear. Engine balanced and approved for service."
    },
    {
        "ACN": "ASRS-187592",
        "Subsystem_Hint": "Oxygen Systems",
        "Narrative": "Flight deck crew oxygen pressure noted below minimum dispatch threshold during pre-departure checks. Cylinder refilled and pressure regulator seal replaced under 1.0 hour delay."
    }
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
# Main App Layout with Tabs
# ---------------------------------------------------------
flights_df, telemetry_source = fetch_live_flights(opensky_user, opensky_pass)

# KPI Cards Header
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Tracked Flights", len(flights_df), delta="CONUS Airspace")
kpi2.metric("Avg Fleet Altitude", f"{int(flights_df['Altitude_m'].mean()):,} m")
kpi3.metric("Avg Fleet Speed", f"{int(flights_df['Velocity_m_s'].mean())} m/s")
kpi4.metric("Telemetry Stream", "ONLINE", delta=telemetry_source)

st.markdown("<br>", unsafe_allow_html=True)

# Organize Content in Executive Tabs
tab1, tab2, tab3 = st.tabs(["📡 Live Operations Radar", "🛡️ Safety & Risk Intelligence", "📊 Fleet Analytics"])

# TAB 1: RADAR & FLIGHT TRACKING
with tab1:
    col_map, col_details = st.columns([3, 1])

    with col_map:
        if not flights_df.empty:
            # Map rendering using your exact scatter layout
            fig = px.scatter_map(
                flights_df,
                lat="lat",
                lon="lon",
                hover_name="Callsign",
                hover_data=["Country", "Altitude_m", "Velocity_m_s"],
                zoom=3.4,
                center={"lat": 39.8283, "lon": -98.5795},
                height=520
            )

            fig.update_layout(
                map_style="carto-darkmatter",
                margin={"r": 0, "t": 0, "l": 0, "b": 0}
            )

            fig.update_traces(
                marker=dict(
                    size=14,
                    color="#00FF66",  # Your green radar marker
                    opacity=0.95
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No live flight telemetry available. Click Sync Fresh Telemetry.")

    with col_details:
        st.subheader("✈️ Flight Drilldown")

        if not flights_df.empty:
            selected_callsign = st.selectbox(
                "Select Flight Callsign:", 
                flights_df["Callsign"].unique()
            )
            selected_flight = flights_df[flights_df["Callsign"] == selected_callsign].iloc[0]

            st.markdown(f"**Origin Country:** `{selected_flight['Country']}`")
            st.markdown(f"**Current Altitude:** `{selected_flight['Altitude_m']} m`")
            st.markdown(f"**Airspeed:** `{selected_flight['Velocity_m_s']} m/s`")
            st.markdown(f"**Coordinates:** `{selected_flight['lat']:.2f}, {selected_flight['lon']:.2f}`")
        else:
            st.info("No active flights to display.")

        if st.button("🔄 Sync Fresh Telemetry", use_container_width=True):
            st.cache_data.clear()
            st.rerun()



# TAB 2: AI RISK EXTRACTION
with tab2:
    asrs_df = load_nasa_asrs_data()
    
    col_sel, col_audit = st.columns([1, 1.5])
    
    with col_sel:
        st.subheader("1. Select NASA Safety Report")
        selected_acn = st.selectbox("NASA ASRS Incident ACN:", asrs_df["ACN"].tolist())
        selected_row = asrs_df[asrs_df["ACN"] == selected_acn].iloc[0]
        custom_log = st.text_area("Narrative Text (Inspect or Modify):", selected_row["Narrative"], height=160)
        analyze_btn = st.button("🚀 Extract Risk Metrics via Gemini AI", use_container_width=True, type="primary")

    with col_audit:
        st.subheader("2. Executive Decision Support Output")
        
        if analyze_btn:
            if not gemini_api_key:
                st.error("⚠️ Please enter a Gemini API Key in the sidebar.")
            else:
                with st.spinner("Analyzing safety logs with Gemini 3.8 Flash..."):
                    risk = analyze_log_with_gemini(custom_log, gemini_api_key)
                
                if risk:
                    st.success("Risk Extraction Complete")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Subsystem Impacted", risk.get("Subsystem", "N/A"))
                    m2.metric("Severity Score", f"{risk.get('Severity_Score', 0)} / 10")
                    m3.metric("Projected Delay", f"{risk.get('Estimated_Delay_Hours', 0)} Hours")
                    
                    st.progress(risk.get('Severity_Score', 0) / 10)
                    
                    st.markdown("#### **Project Manager Action Plan:**")
                    st.warning(f"**Action Required:** {risk.get('AI_Summary', 'N/A')}")
                    st.info(f"**FAA Compliance Review Required:** {'YES' if risk.get('FAA_Compliance_Flag') else 'NO'}")
        else:
            st.info("Select a narrative and click 'Extract Risk Metrics' to run Gemini AI analysis.")

# TAB 3: SUBSYSTEM RISK ANALYTICS
with tab3:
    st.subheader("📊 Fleet-Wide Maintenance Risk Distribution")
    asrs_df = load_nasa_asrs_data()
    
    if "Subsystem_Hint" in asrs_df.columns:
        subsystem_counts = asrs_df["Subsystem_Hint"].value_counts().reset_index()
        subsystem_counts.columns = ["Subsystem", "Incident_Count"]
        
        fig_bar = px.bar(
            subsystem_counts,
            x="Subsystem",
            y="Incident_Count",
            color="Subsystem",
            title="NASA ASRS Incidents by Aircraft Subsystem",
            template="plotly_dark"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("SkyGuard AI Capstone | Developed with Streamlit, Google Gemini 3.8, and OpenSky Network.")
