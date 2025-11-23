import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import zipfile
import io

# --- Function to load model from a zip file ---
def load_model_from_zip(zip_path, file_in_zip):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            file_bytes = z.read(file_in_zip)
            return joblib.load(io.BytesIO(file_bytes))
    except FileNotFoundError:
        st.error(f"Error: The zip file '{zip_path}' was not found in the repository.")
        st.stop()
    except KeyError:
        st.error(f"Error: Could not find '{file_in_zip}' inside '{zip_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        st.stop()

# --- Load the model and feature names ---
# Load the model FROM THE ZIP FILE
model = load_model_from_zip('wsn_modelC.zip', 'wsn_model.joblib')

# Load the feature names DIRECTLY from the repository
try:
    expected_features = joblib.load('feature_names.joblib')
except FileNotFoundError:
    st.error("Error: 'feature_names.joblib' not found. Please ensure it is in the GitHub repository.")
    st.stop()


# --- Main App Logic ---
# Create a standard scaler
scaler = StandardScaler()

# App title
st.title('Wireless Sensor Network Intrusion Detection System')
st.write('Upload a CSV file with WSN traffic data to predict if it is normal or malicious.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df_test.head())

    try:
        # Keep a copy of the original data for display
        df_display = df_test.copy() 
        
        # Check if all expected columns are present in the uploaded file
        if not all(feature in df_test.columns for feature in expected_features):
            st.error("The uploaded CSV file is missing one or more required columns.")
            st.info(f"Required columns are: {expected_features}")
            st.stop()

        # Select and reorder the columns to match the model's training data exactly
        X_test = df_test[expected_features]
        
        # Scale the features
        X_test_scaled = scaler.fit_transform(X_test)

        # Make predictions
        predictions = model.predict(X_test_scaled)

        # Map predictions back to labels
        attack_labels = {
            0: 'Blackhole Attack',
            1: 'Flooding Attack',
            2: 'Grayhole Attack',
            3: 'Normal Traffic',
            4: 'Scheduling Attack'
        }
        
        predicted_labels = [attack_labels.get(p, 'Unknown') for p in predictions]
        
        # Add predictions to the display dataframe
        df_display['Predicted Attack Type'] = predicted_labels

        st.write("Prediction Results:")
        st.dataframe(df_display)

        st.write("Prediction Summary:")
        st.write(df_display['Predicted Attack Type'].value_counts())

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the uploaded CSV has the correct format.")
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
# (You can add custom CSS here if you want more advanced styling)
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
def load_model_from_zip(zip_path, file_in_zip):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            file_bytes = z.read(file_in_zip)
            return joblib.load(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Fatal Error: Could not load model from '{zip_path}'. Please ensure the file is in the repository. Error: {e}")
        st.stop()

@st.cache_resource
def load_joblib(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Fatal Error: Could not load '{file_path}'. Please ensure the file is in the repository. Error: {e}")
        st.stop()

# Load all necessary files once and cache them
model = load_model_from_zip('wsn_modelC.zip', 'wsn_model.joblib')
expected_features = load_joblib('feature_names.joblib')
scaler = load_joblib('scaler.joblib')

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

            # --- Display Metrics ---
            total_packets = len(df_display)
            threats_detected = df_display['Is Attack'].sum()
            normal_packets = total_packets - threats_detected

            st.markdown("---")
            st.header("Analysis Dashboard")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Packets Analyzed", f"{total_packets:,}")
            col2.metric("Normal Packets", f"{normal_packets:,}")
            col3.metric("Threats Detected", f"{threats_detected:,}")

            # --- Display Results ---
            st.markdown("---")
            st.header("Prediction Results & Visualization")

            col1_viz, col2_viz = st.columns([0.6, 0.4])

            with col1_viz:
                st.subheader("Detailed Packet Analysis")
                # Color-code the results
                def highlight_attacks(row):
                    return ['background-color: #e94560; color: white' if row['Is Attack'] else '' for _ in row]

                st.dataframe(df_display.style.apply(highlight_attacks, axis=1), height=400)

            with col2_viz:
                st.subheader("Prediction Summary")
                summary_df = df_display['Prediction'].value_counts().reset_index()
                summary_df.columns = ['Prediction Type', 'Count']
                
                # Create a cool donut chart
                fig = px.pie(summary_df, names='Prediction Type', values='Count', hole=0.5,
                             color_discrete_map={'Normal Traffic': '#16c79a',
                                                 'Blackhole Attack': '#e94560',
                                                 'Flooding Attack': '#ff8c00',
                                                 'Grayhole Attack': '#f08080',
                                                 'Scheduling Attack': '#ff6347'})
                fig.update_layout(
                    title_text='Distribution of Predictions',
                    template='plotly_dark',
                    legend_title_text='Traffic Type'
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.info("Awaiting CSV file upload...")
