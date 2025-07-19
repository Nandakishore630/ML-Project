import streamlit as st
import pandas as pd
import joblib

 
model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_columns.pkl")  # saved during training

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")
 
st.sidebar.header("Input Employee Details")

# Collect inputs (no default values)
age = st.sidebar.number_input("Age", min_value=18, max_value=65)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", ["Tech-support", "Craft-repair", "Other-service", "Sales","Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct","Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"])
hours_per_week = st.sidebar.number_input("Hours per week", min_value=1, max_value=80)
experience = st.sidebar.number_input("Years of Experience", min_value=0, max_value=40)


input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)


input_processed = pd.get_dummies(input_df)

input_processed = input_processed.reindex(columns=feature_names, fill_value=0)

 
if st.button("Predict Salary Class"):
    prediction = model.predict(input_processed)
    st.success(f"✅ Prediction: {prediction[0]}")
 
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    batch_processed = pd.get_dummies(batch_data)
    batch_processed = batch_processed.reindex(columns=feature_names, fill_value=0)

 
    batch_preds = model.predict(batch_processed)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
