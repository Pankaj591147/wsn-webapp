import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import zipfile
import io
import plotly.express as px

# ----------------- CONFIGURATION -----------------
st.set_page_config(
    page_title="WSN Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- STYLING -----------------
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #16213e;
    }
    /* Metric styling */
    .stMetric {
        background-color: #0f3460;
        border-radius: 10px;
        padding: 15px;
    }
    .stMetric > label {
        color: #a0a0a0;
    }
    .stMetric > div {
        color: #e94560;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #e94560;
        background-color: transparent;
        color: #e94560;
    }
    .stButton>button:hover {
        border-color: #e94560;
        background-color: #e94560;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- CACHED DATA LOADING -----------------
@st.cache_resource
def load_object_from_zip(zip_path, file_in_zip):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            file_bytes = z.read(file_in_zip)
            return joblib.load(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Fatal Error: Could not load '{file_in_zip}' from '{zip_path}'. Error: {e}")
        st.stop()

@st.cache_resource
def load_joblib_direct(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Fatal Error: Could not load '{file_path}'. Please ensure the file is in the repository. Error: {e}")
        st.stop()

# --- START OF CORRECTED CODE ---
# Load all necessary files, telling the code where to look for each one.
model = load_object_from_zip('wsn_modelC.zip', 'wsn_model.joblib') # Get model from zip
scaler = load_joblib_direct('scaler.joblib') # Load scaler as a separate file
expected_features = load_joblib_direct('feature_names.joblib') # Load features as a separate file
# --- END OF CORRECTED CODE ---


# ----------------- PAGE SELECTION -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Intrusion Detection", "About the Project"])


# ----------------- ABOUT PAGE -----------------
if page == "About the Project":
    st.title("🛡️ WSN Intrusion Detection System")
    st.markdown("### A Machine Learning Approach to Secure Wireless Sensor Networks")
    
    st.markdown("""
    This project leverages a **Random Forest classifier**, a powerful machine learning model, to detect and classify various types of intrusions in Wireless Sensor Networks (WSNs).
    
    **Why is this important?**
    WSNs are used in critical applications like environmental monitoring, healthcare, and military surveillance. However, their resource-constrained nature makes them highly vulnerable to attacks. This system provides an intelligent, data-driven defense mechanism.
    
    **Technology Stack:**
    - **Model:** Scikit-learn (Random Forest)
    - **Web Framework:** Streamlit
    - **Data Handling:** Pandas
    - **Deployment:** Streamlit Community Cloud & GitHub
    
    Navigate
