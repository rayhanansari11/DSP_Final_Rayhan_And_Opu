import streamlit as st
import pandas as pd
import joblib

# Load the model and preprocessor
model = joblib.load("models/final_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

st.title("Medical Charges Prediction App")

# User input form
st.header("Input Patient Data")
age = st.number_input("Age", min_value=0, max_value=100, step=1, value=18)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict"):
    # Create input data
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
    
    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)
    
    # Predict
    prediction = model.predict(input_data_preprocessed)
    
    # Display result
    st.success(f"Predicted Medical Charges: ${prediction[0]:.2f}")
