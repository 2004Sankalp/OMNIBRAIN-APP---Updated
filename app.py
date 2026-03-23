import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="OMNIBRAIN AI Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('heart_model.pkl')

model = load_model()

# Header
st.title("🫀 OMNIBRAIN: Multi-Class Heart Risk AI")
st.markdown("**Team 16** | Advanced severity detection of Coronary Artery Disease.")
st.divider()

st.subheader("Patient Vitals Input")

# Clean, text-based inputs matching the raw dataset
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50.0, max_value=250.0, value=120.0)
    chol = st.number_input("Serum Cholesterol", min_value=100.0, max_value=600.0, value=200.0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])

with col2:
    thalch = st.number_input("Max Heart Rate Achieved", min_value=60.0, max_value=220.0, value=150.0)
    exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("ST Segment Slope", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Major Vessels Colored by Flourosopy", [0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

st.divider()

# Prediction Logic
if st.button("Analyze Patient Severity", type="primary", use_container_width=True):
    
    # Pack the user inputs into a dataframe exact matching the original column names
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Predict the severity class (0 to 4)
    prediction = model.predict(input_data)[0]
    
    st.subheader("AI Diagnostic Assessment")
    
    # Multi-Class Output Mapping
    if prediction == 0:
        st.success("✅ **Class 0: Healthy Patient**")
        st.write("No significant narrowing of coronary arteries detected. Low risk.")
    elif prediction == 1:
        st.warning("⚠️ **Class 1: Mild Coronary Artery Disease**")
        st.write("Initial stages of vessel narrowing detected. Lifestyle changes and monitoring recommended.")
    elif prediction == 2:
        st.warning("🟠 **Class 2: Moderate Coronary Artery Disease**")
        st.write("Moderate blockage present. Recommend scheduling an appointment with a cardiologist.")
    elif prediction == 3:
        st.error("🚨 **Class 3: Severe Coronary Artery Disease**")
        st.write("High probability of severe vessel narrowing. Urgent medical consultation required.")
    elif prediction == 4:
        st.error("💀 **Class 4: Critical Cardiovascular Disease**")
        st.write("Extreme risk. Critical vessel blockage indicated. Immediate emergency medical intervention required.")
        # --- NEW GRAPH CODE STARTS HERE ---
    st.divider()
    st.subheader("📊 AI Confidence Breakdown")
    st.write("This chart displays the model's calculated probability for each severity level.")
    
    # Get the raw mathematical probabilities from the Random Forest model
    probabilities = model.predict_proba(input_data)[0]
    
    # Create a simple dataframe for the graph
    prob_df = pd.DataFrame({
        'Severity Level': ['Class 0 (Healthy)', 'Class 1 (Mild)', 'Class 2 (Moderate)', 'Class 3 (Severe)', 'Class 4 (Critical)'],
        'Probability (%)': probabilities * 100
    })
    
    # Display a native Streamlit bar chart
    st.bar_chart(prob_df.set_index('Severity Level'), color="#ff4b4b")
    # --- NEW GRAPH CODE ENDS HERE ---