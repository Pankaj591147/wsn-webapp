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
    # (About page content remains the same)
    st.markdown("""
    This project leverages a **Random Forest classifier** to detect intrusions in Wireless Sensor Networks (WSNs).
    Navigate to the **Live Intrusion Detection** page to upload your own WSN data and see the model in action!
    """)
    st.image("https://placehold.co/800x300/1a1a2e/e94560?text=WSN+Security+Concept", caption="Securing the sensory backbone of the IoT.")


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

            total_packets = len(df_display)
            threats_detected = df_display['Is Attack'].sum()
            normal_packets = total_packets - threats_detected

            st.markdown("---")
            st.header("Analysis Dashboard")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Packets Analyzed", f"{total_packets:,}")
            col2.metric("Normal Packets", f"{normal_packets:,}")
            col3.metric("Threats Detected", f"{threats_detected:,}")

            st.markdown("---")
            st.header("Prediction Results & Visualization")

            col1_viz, col2_viz = st.columns([0.6, 0.4])

            with col1_viz:
                # --- START OF CORRECTED CODE ---
                st.subheader("Detailed Packet Analysis (Top 1000 Rows)")
                def highlight_attacks(row):
                    return ['background-color: #e94560; color: white' if row['Is Attack'] else '' for _ in row]
                
                # Only style and display the first 1000 rows
                st.dataframe(df_display.head(1000).style.apply(highlight_attacks, axis=1), height=350)

                # Prepare the full file for download
                @st.cache_data # Cache the conversion to make downloads faster
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(df_display)

                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv',
                )
                # --- END OF CORRECTED CODE ---

            with col2_viz:
                st.subheader("Prediction Summary")
                summary_df = df_display['Prediction'].value_counts().reset_index()
                summary_df.columns = ['Prediction Type', 'Count']
                
                fig = px.pie(
    summary_df,
    names='Prediction Type',
    values='Count',
    hole=0.5,
    color_discrete_map={
        'Normal Traffic': '#0aff99',         # Neon green
        'Blackhole Attack': '#ff0033',       # Deep red
        'Flooding Attack': '#ff6600',        # Dark orange
        'Grayhole Attack': '#aa88ff',        # Violet
        'Scheduling Attack': '#0099ff'       # Deep blue
    }
)

fig.update_layout(
    title_text='Distribution of Predictions',
    template='plotly_dark',
    legend_title_text='Traffic Type'
)

                fig.update_layout(title_text='Distribution of Predictions', template='plotly_dark', legend_title_text='Traffic Type')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.info("Awaiting CSV file upload...")
