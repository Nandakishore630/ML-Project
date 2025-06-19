import streamlit as st
import numpy as np
import joblib
model = joblib.load('diabetes_model.pkl')


st.title("Diabetes Prediction App")

st.write("Enter the following details to predict diabetes:")

# Input features
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
age = st.number_input("Age", min_value=0)


if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is unlikely to have diabetes.")