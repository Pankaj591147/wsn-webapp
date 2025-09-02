import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import zipfile  # <-- Add this import
import io         # <-- Add this import

# --- START OF CHANGED CODE ---

# Function to load the model from a zip file
def load_model_from_zip(zip_path, file_in_zip):
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Read the file from the zip archive into memory
        file_bytes = z.read(file_in_zip)
        # Use io.BytesIO to treat the bytes as a file
        return joblib.load(io.BytesIO(file_bytes))

# Load the trained model from the zip file
# IMPORTANT: The first argument is the name of your zip file in the repo.
# The second is the name of the .joblib file *inside* the zip archive.
try:
    model = load_model_from_zip('wsn_modelC.zip', 'wsn_model.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop() # Stop the app if the model can't be loaded

# --- END OF CHANGED CODE ---


# Create a standard scaler (it should be the same as the one used for training)
scaler = StandardScaler()

# App title
st.title('Wireless Sensor Network Intrusion Detection System')
st.write('Upload a CSV file with WSN traffic data to predict if it is normal or malicious.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    df_test = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df_test.head())

    # Preprocess the uploaded data (must be consistent with training)
    try:
        X_test = df_test.drop(['id', 'Attack type'], axis=1, errors='ignore') 
        
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
        
        # Add predictions to the dataframe
        df_test['Predicted Attack Type'] = predicted_labels

        st.write("Prediction Results:")
        st.dataframe(df_test)

        # Display a summary
        st.write("Prediction Summary:")
        st.write(df_test['Predicted Attack Type'].value_counts())

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the uploaded CSV has the correct format and columns.")
