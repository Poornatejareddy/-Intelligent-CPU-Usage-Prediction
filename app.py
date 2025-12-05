import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Page Config
st.set_page_config(
    page_title="CPU Usage Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    h2, h3 {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    import os
    model_path = 'models/model.pkl'
    
    if not os.path.exists(model_path):
        # Provide debugging information
        cwd = os.getcwd()
        models_dir_exists = os.path.exists('models')
        models_contents = os.listdir('models') if models_dir_exists else []
        
        error_msg = f"""
        ### Model file not found!
        
        **Expected path:** `{model_path}`  
        **Current directory:** `{cwd}`  
        **Models directory exists:** {models_dir_exists}  
        **Models directory contents:** {models_contents if models_contents else 'Empty or does not exist'}
        
        ### How to fix:
        1. **Local development:** Run the training pipeline:
           ```bash
           python src/preprocess.py
           python src/train.py
           python src/evaluate.py
           ```
        
        2. **Docker/Render deployment:** Ensure the Dockerfile runs the training pipeline during build.
        """
        raise FileNotFoundError(error_msg)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Header
st.title("⚡ Intelligent CPU Usage Prediction")
st.markdown("### Optimize your Kubernetes resources with AI-powered predictions")

# Sidebar for Inputs
with st.sidebar:
    st.header("🔧 Configuration")
    st.markdown("Adjust the resource parameters below:")
    
    with st.expander("CPU & Memory Settings", expanded=True):
        cpu_request = st.slider("CPU Request (m)", 0, 5000, 500, step=100)
        cpu_limit = st.slider("CPU Limit (m)", 0, 8000, 1000, step=100)
        mem_request = st.slider("Memory Request (MiB)", 0, 10000, 1000, step=100)
        mem_limit = st.slider("Memory Limit (MiB)", 0, 12000, 2000, step=100)
    
    with st.expander("Workload Details", expanded=True):
        runtime_minutes = st.number_input("Runtime (minutes)", min_value=0, value=60)
        controller_kind = st.selectbox("Controller Kind", ['Deployment', 'StatefulSet', 'DaemonSet', 'Job'])

    predict_btn = st.button("🚀 Predict Usage")

# Main Content Area
col1, col2 = st.columns([1, 2])

if predict_btn:
    with st.spinner("Analyzing workload patterns..."):
        time.sleep(0.5) # Simulating processing time for effect
        
        # Prepare input data
        # Manual One-Hot Encoding (matching training logic)
        categories = ['DaemonSet', 'Deployment', 'Job', 'StatefulSet']
        input_features = [cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes]
        for kind in categories:
            input_features.append(1 if controller_kind == kind else 0)
            
        input_array = np.array(input_features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(input_array)[0]
        prediction = max(0, prediction) # Ensure non-negative
        
        # Display Results
        with col1:
            st.markdown("### 📊 Prediction Result")
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin:0; font-size: 48px; color: #2980b9;">{prediction:.0f} m</h2>
                <p style="margin:0; color: #7f8c8d;">Predicted CPU Usage</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Utilization Analysis")
            utilization = (prediction / cpu_limit) * 100 if cpu_limit > 0 else 0
            st.progress(min(utilization / 100, 1.0))
            st.caption(f"Predicted usage is **{utilization:.1f}%** of the limit.")
            
            if utilization > 90:
                st.error("⚠️ High Risk of Throttling!")
            elif utilization > 75:
                st.warning("⚠️ High Utilization")
            else:
                st.success("✅ Healthy Utilization")

        with col2:
            st.markdown("### 📈 Resource Visualization")
            
            # Bar Chart: Request vs Limit vs Usage
            chart_data = pd.DataFrame({
                'Metric': ['CPU Request', 'Predicted Usage', 'CPU Limit'],
                'Value (m)': [cpu_request, prediction, cpu_limit]
            })
            
            # Custom color palette
            colors = ['#95a5a6', '#3498db', '#e74c3c']
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Metric', y='Value (m)', data=chart_data, palette=colors, ax=ax)
            
            # Add value labels
            for i, v in enumerate(chart_data['Value (m)']):
                ax.text(i, v + 50, f"{v:.0f}", ha='center', fontweight='bold')
                
            ax.set_ylabel("CPU (millicores)")
            ax.set_title("Resource Allocation vs Prediction")
            sns.despine()
            st.pyplot(fig)

else:
    # Default state / Welcome message
    with col1:
        st.info("👈 Adjust parameters in the sidebar and click **Predict** to see results.")
    
    with col2:
        # Show model performance if available
        try:
            import json
            with open('evaluation.json', 'r') as f:
                metrics = json.load(f)
            
            st.markdown("### 🏆 Model Performance")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("R² Score", f"{metrics['R2']:.3f}")
            m_col2.metric("MAE", f"{metrics['MAE']:.1f}")
            m_col3.metric("RMSE", f"{metrics['RMSE']:.1f}")
            
            # Show the actual vs predicted plot from training
            st.image("plots/actual_vs_predicted.png", caption="Model Evaluation: Actual vs Predicted", use_column_width=True)
            
        except FileNotFoundError:
            st.warning("Model metrics not found.")
