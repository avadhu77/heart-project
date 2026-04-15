import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Heart Failure Prediction App")
st.write("Enter the patient details below to predict survival probability.")

# Creating columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=60)
    anaemia = st.selectbox("Anaemia (0=No, 1=Yes)", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", value=582)
    diabetes = st.selectbox("Diabetes (0=No, 1=Yes)", [0, 1])

with col2:
    ejection_fraction = st.number_input("Ejection Fraction", value=38)
    high_blood_pressure = st.selectbox("High Blood Pressure (0=No, 1=Yes)", [0, 1])
    platelets = st.number_input("Platelets", value=265000.0)
    serum_creatinine = st.number_input("Serum Creatinine", value=1.1)

with col3:
    serum_sodium = st.number_input("Serum Sodium", value=137)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
    smoking = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1])
    time = st.number_input("Follow-up Period (days)", value=100)

# Prediction Logic
if st.button("Predict Survival"):
    # Organize features in the exact order the model expects
    features = np.array([[
        age, anaemia, creatinine_phosphokinase, diabetes, 
        ejection_fraction, high_blood_pressure, platelets, 
        serum_creatinine, serum_sodium, sex, smoking, time
    ]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    if prediction[0] == 1:
        st.error(f"High Risk: The model predicts a potential heart failure event.")
    else:
        st.success(f"Low Risk: The model predicts a survival outcome.")
        
    st.write(f"Confidence Level: {np.max(probability) * 100:.2f}%")
