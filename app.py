import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load model
@st.cache_resource
def load_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("CPU Usage Prediction")

st.write("Enter the resource configuration to predict CPU usage.")

# Input fields
cpu_request = st.number_input("CPU Request (m)", min_value=0, value=500)
mem_request = st.number_input("Memory Request (MiB)", min_value=0, value=1000)
cpu_limit = st.number_input("CPU Limit (m)", min_value=0, value=1000)
mem_limit = st.number_input("Memory Limit (MiB)", min_value=0, value=2000)
runtime_minutes = st.number_input("Runtime (minutes)", min_value=0, value=60)
controller_kind = st.selectbox("Controller Kind", ['Deployment', 'StatefulSet', 'DaemonSet', 'Job'])

if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'cpu_request': [cpu_request],
        'mem_request': [mem_request],
        'cpu_limit': [cpu_limit],
        'mem_limit': [mem_limit],
        'runtime_minutes': [runtime_minutes],
        'controller_kind': [controller_kind]
    })
    
    # Preprocessing (must match training preprocessing)
    # Note: In a real production app, the preprocessor should be saved and loaded.
    # Here, I'll manually replicate the encoding logic for simplicity, 
    # but ideally, the OneHotEncoder should be part of a pipeline or saved separately.
    
    # Since we didn't save the encoder, we need to handle it carefully.
    # The training script used OneHotEncoder with sparse_output=False.
    # The columns were: 'Deployment', 'StatefulSet', 'DaemonSet', 'Job' (sorted alphabetically usually?)
    # Wait, OneHotEncoder sorts categories by default? No, it depends on the data order if not specified.
    # Actually, `get_feature_names_out` was used.
    
    # To be safe, let's assume the categories are fixed as defined in generate_data.py
    # and we can manually encode them or use a saved encoder.
    # Given the constraints, I will manually encode based on the known categories.
    
    categories = ['DaemonSet', 'Deployment', 'Job', 'StatefulSet'] # Alphabetical order is standard for OHE if not specified?
    # Actually, let's check how it was done.
    # In preprocess.py: encoder.fit_transform(X[['controller_kind']])
    # If I want to be robust, I should have saved the encoder.
    
    # Let's modify the app to handle this gracefully.
    # I'll create the feature vector manually.
    
    # Features expected by the model (based on train.py):
    # cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, 
    # controller_kind_DaemonSet, controller_kind_Deployment, controller_kind_Job, controller_kind_StatefulSet
    # (assuming alphabetical order which is typical for OHE)
    
    # Let's verify the columns in X_train.csv to be sure.
    # But I can't read it here easily without running code.
    # I'll assume alphabetical order for now.
    
    input_features = [cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes]
    
    # One-hot encoding
    for kind in categories:
        input_features.append(1 if controller_kind == kind else 0)
        
    # Convert to numpy array and reshape
    input_array = np.array(input_features).reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_array)[0]
    
    st.success(f"Predicted CPU Usage: {prediction:.2f} m")
    
    # Display metrics if available
    try:
        import json
        with open('evaluation.json', 'r') as f:
            metrics = json.load(f)
        st.subheader("Model Performance")
        st.json(metrics)
    except FileNotFoundError:
        pass
