import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="OMNIBRAIN AI Predictor", layout="wide") # Changed to 'wide' for bigger charts!

@st.cache_resource
def load_model():
    return joblib.load('heart_model.pkl')

model = load_model()

# Header
st.title("🫀 OMNIBRAIN: Multi-Class Heart Risk AI")
st.markdown("**Team 16** | Advanced severity detection of Coronary Artery Disease.")
st.divider()

st.subheader("Patient Vitals Input")

# Clean, text-based inputs
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
    
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    danger_percentage = int((prediction / 4) * 100)

    # --- TOP ROW: RESULTS & GAUGE CHART ---
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("AI Diagnostic Assessment")
        if prediction == 0:
            st.success("✅ **Class 0: Healthy Patient**\n\nNo significant narrowing of coronary arteries detected. Low risk.")
            color = "green"
        elif prediction == 1:
            st.warning("⚠️ **Class 1: Mild CAD**\n\nInitial stages of vessel narrowing detected.")
            color = "lightgreen"
        elif prediction == 2:
            st.warning("🟠 **Class 2: Moderate CAD**\n\nModerate blockage present. Recommend cardiology consult.")
            color = "orange"
        elif prediction == 3:
            st.error("🚨 **Class 3: Severe CAD**\n\nHigh probability of severe vessel narrowing.")
            color = "red"
        elif prediction == 4:
            st.error("💀 **Class 4: Critical Disease**\n\nExtreme risk. Immediate emergency intervention required.")
            color = "darkred"

    with res_col2:
        # PLOTLY INTERACTIVE GAUGE
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = danger_percentage,
            title = {'text': "Clinical Risk Severity (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps' : [
                    {'range': [0, 25], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [25, 50], 'color': "rgba(255, 255, 0, 0.2)"},
                    {'range': [50, 75], 'color': "rgba(255, 165, 0, 0.2)"},
                    {'range': [75, 100], 'color': "rgba(255, 0, 0, 0.2)"}],
            }))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- BOTTOM ROW: RADAR & PROBABILITY ---
    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("🕸️ Patient Vitals Radar")
        # Normalize data roughly so it fits on a clean radar chart 0-100 scale
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
          height=350, margin=dict(l=30, r=30, t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with chart_col2:
        st.subheader("📊 AI Confidence Breakdown")
        # Plotly Interactive Bar Chart
        prob_df = pd.DataFrame({
            'Class': ['0 (Healthy)', '1 (Mild)', '2 (Moderate)', '3 (Severe)', '4 (Critical)'],
            'Probability (%)': probabilities * 100
        })
        fig_bar = px.bar(prob_df, x='Class', y='Probability (%)', color='Class',
                         color_discrete_sequence=['green', 'lightgreen', 'orange', 'red', 'darkred'])
        fig_bar.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
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