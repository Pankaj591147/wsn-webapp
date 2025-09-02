import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('wsn_model.joblib')

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
    # This assumes the uploaded CSV has the same columns as the training data
    try:
        X_test = df_test.drop(['id', 'Attack type'], axis=1, errors='ignore') # errors='ignore' will not raise an error if a column is not found
        
        # Scale the features
        X_test_scaled = scaler.fit_transform(X_test)

        # Make predictions
        predictions = model.predict(X_test_scaled)

        # Map predictions back to labels (you may need to adjust these based on your LabelEncoder)
        # 0: Blackhole, 1: Flooding, 2: Grayhole, 3: Normal, 4: Scheduling
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
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure the uploaded CSV has the correct format and columns.")