import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Changed back to "centered" for that clean, focused look
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

# --- VISUAL DASHBOARD ---
st.subheader("📊 Vitals vs. Healthy Baselines")
met1, met2, met3 = st.columns(3)
met1.metric(label="Cholesterol", value=f"{chol} mg/dl", delta=f"{chol - 200} from baseline", delta_color="inverse")
met2.metric(label="Resting BP", value=f"{trestbps} mmHg", delta=f"{trestbps - 120} from baseline", delta_color="inverse")
met3.metric(label="Max Heart Rate", value=f"{thalch} bpm", delta=f"{thalch - 150} from baseline", delta_color="normal")

st.divider()

# Prediction Logic
if st.button("Analyze Patient Severity", type="primary", use_container_width=True):
    
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    danger_percentage = int((prediction / 4) * 100)
    
    st.subheader("AI Diagnostic Assessment")
    
    # 1. RESULT MESSAGE
    if prediction == 0:
        st.success("✅ **Class 0: Healthy Patient**\n\nNo significant narrowing of coronary arteries detected. Low risk.")
        color = "green"
    elif prediction == 1:
        st.warning("⚠️ **Class 1: Mild Coronary Artery Disease**\n\nInitial stages of vessel narrowing detected. Lifestyle changes and monitoring recommended.")
        color = "lightgreen"
    elif prediction == 2:
        st.warning("🟠 **Class 2: Moderate Coronary Artery Disease**\n\nModerate blockage present. Recommend scheduling an appointment with a cardiologist.")
        color = "orange"
    elif prediction == 3:
        st.error("🚨 **Class 3: Severe Coronary Artery Disease**\n\nHigh probability of severe vessel narrowing. Urgent medical consultation required.")
        color = "red"
    elif prediction == 4:
        st.error("💀 **Class 4: Critical Cardiovascular Disease**\n\nExtreme risk. Critical vessel blockage indicated. Immediate emergency medical intervention required.")
        color = "darkred"

    # 2. GAUGE METER
    st.write("### System Risk Level")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = danger_percentage,
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps' : [
                {'range': [0, 25], 'color': "rgba(0, 255, 0, 0.1)"},
                {'range': [25, 50], 'color': "rgba(255, 255, 0, 0.2)"},
                {'range': [50, 75], 'color': "rgba(255, 165, 0, 0.2)"},
                {'range': [75, 100], 'color': "rgba(255, 0, 0, 0.2)"}],
        }))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 3. PATIENT RADAR CHART
    st.divider()
    st.subheader("🕸️ Patient Vitals Radar")
    st.write("A 360-degree view of how the patient's vitals map out. (Larger area = higher severity flags).")
    categories = ['Blood Pressure', 'Cholesterol', 'Max HR', 'ST Depression']
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
          r=[trestbps/2, chol/3, thalch/2, oldpeak*25],
          theta=categories,
          fill='toself',
          name='Patient Profile',
          line_color=color
    ))
    fig_radar.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
      showlegend=False,
      height=400, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # 4. PROBABILITY BAR CHART
    st.divider()
    st.subheader("📊 AI Confidence Breakdown")
    st.write("This chart displays the model's calculated mathematical probability for each severity level.")
    prob_df = pd.DataFrame({
        'Class': ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'],
        'Probability (%)': probabilities * 100
    })
    fig_bar = px.bar(prob_df, x='Class', y='Probability (%)', color='Class',
                     color_discrete_sequence=['green', 'lightgreen', 'orange', 'red', 'darkred'])
    fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- MEDICAL GLOSSARY ---
st.divider()
with st.expander("ℹ️ Clinical Terminology Reference"):
    st.markdown("""
    * **Thallium Scan (thal):** A nuclear stress test. 'Reversable defect' indicates blood flow completely stops during exercise.
    * **Fluoroscopy (ca):** The number of major coronary arteries (0-3) showing severe blockages under X-ray dye.
    * **ST Depression (oldpeak):** How far the heartbeat wave drops below normal on an ECG after exercise. Drops > 2.0 indicate severe oxygen starvation.
    * **ST Segment Slope:** The angle of the heartbeat wave during exercise. 'Downsloping' is a critical warning sign of narrowing arteries.
    """)