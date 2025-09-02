import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import zipfile
import io

# --- START OF CHANGED CODE ---

# Function to load objects from a zip file (works for model and feature list)
def load_object_from_zip(zip_path, file_in_zip):
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_bytes = z.read(file_in_zip)
        return joblib.load(io.BytesIO(file_bytes))

# Add 'feature_names.joblib' to your zip file and upload it to GitHub
# For now, let's assume it's directly in the repo for simplicity
try:
    # Load the model
    model = joblib.load('wsn_model.joblib') # Reverting to direct load for clarity, use zip if needed
    # Load the feature names the model was trained on
    expected_features = joblib.load('feature_names.joblib') 
except Exception as e:
    st.error(f"Error loading model or feature list: {e}")
    st.info("Please ensure 'wsn_model.joblib' and 'feature_names.joblib' are in the GitHub repository.")
    st.stop()

# --- END OF CHANGED CODE ---

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
        # --- START OF CHANGED CODE ---
        # Keep a copy of the original data for display
        df_display = df_test.copy() 
        
        # Check if all expected columns are present in the uploaded file
        if not all(feature in df_test.columns for feature in expected_features):
            st.error("The uploaded CSV file is missing one or more required columns.")
            st.info(f"Required columns are: {expected_features}")
            st.stop()

        # Select and reorder the columns to match the model's training data exactly
        X_test = df_test[expected_features]
        # --- END OF CHANGED CODE ---
        
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
