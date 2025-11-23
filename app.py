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
    /* Main background and text colors */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #2d2d5f 100%);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #00d9ff !important;
        font-weight: 600;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d9ff !important;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #9d4edd !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #00d9ff !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7209b7 0%, #560bad 100%) !important;
        color: white !important;
        border: 2px solid #9d4edd !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #9d4edd 0%, #7209b7 100%) !important;
        box-shadow: 0 0 20px rgba(157, 78, 221, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(157, 78, 221, 0.1) !important;
        border: 2px dashed #9d4edd !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: rgba(0, 217, 255, 0.1) !important;
        border-left: 4px solid #00d9ff !important;
        border-radius: 5px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(157, 78, 221, 0.2) !important;
        border-radius: 8px !important;
        color: #00d9ff !important;
        font-weight: 600 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #00d9ff 0%, #0096c7 100%) !important;
        color: #0f2027 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #48cae4 0%, #00d9ff 100%) !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.5) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(157, 78, 221, 0.1) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
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

# Load all necessary files
model = load_object_from_zip('wsn_modelC.zip', 'wsn_model.joblib')
scaler = load_joblib_direct('scaler.joblib')
expected_features = load_joblib_direct('feature_names.joblib')

# ----------------- PAGE SELECTION -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Intrusion Detection", "About the Project"])

# ----------------- ABOUT PAGE -----------------
if page == "About the Project":
    st.title("🛡️ WSN Intrusion Detection System")
    st.markdown("### A Machine Learning Approach to Secure Wireless Sensor Networks")
    
    st.markdown("""
    This project leverages a **Random Forest classifier** to detect intrusions in Wireless Sensor Networks (WSNs).
    Navigate to the **Live Intrusion Detection** page to upload your own WSN data and see the model in action!
    """)
    
    st.image("https://placehold.co/800x300/2c5364/00d9ff?text=WSN+Security+Concept", caption="Securing the sensory backbone of the IoT.")

# ----------------- DETECTION PAGE -----------------
elif page == "Live Intrusion Detection":
    st.title("🕵️ Live Network Traffic Analysis")
    st.markdown("Upload a CSV file containing WSN traffic data. The system will analyze each packet and classify it as **Normal** or a specific type of **Attack**.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a file with the same format as the training data.")
    
    if uploaded_file is not None:
        df_test = pd.read_csv(uploaded_file)
        
        with st.expander("Show Uploaded Data Preview"):
            st.dataframe(df_test.head())
        
        try:
            df_display = df_test.copy()
            
            if not all(feature in df_test.columns for feature in expected_features):
                st.error("CSV File Error: The uploaded file is missing required columns.")
                st.info(f"Required columns are: {expected_features}")
                st.stop()
            
            X_test = df_test[expected_features]
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            
            attack_labels = {3: 'Normal Traffic', 0: 'Blackhole Attack', 1: 'Flooding Attack', 2: 'Grayhole Attack', 4: 'Scheduling Attack'}
            df_display['Prediction'] = [attack_labels.get(p, 'Unknown') for p in predictions]
            df_display['Is Attack'] = df_display['Prediction'] != 'Normal Traffic'
