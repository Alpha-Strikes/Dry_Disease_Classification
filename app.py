import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Loading the trained model
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = ['Gender', 'Age', 'Sleep duration', 'Sleep quality', 'Sleep disorder', 'Wake up during night', 'Feel sleepy during day', 'Caffeine consumption', 'Alcohol consumption', 'Smoking', 'Medical issue', 'Ongoing medication', 'Smart device before bed', 'Average screen time', 'Blue-light filter', 'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye']

x1 = st.number_input("Enter Gender (Male -> 0 | Female -> 1):")
x2 = st.number_input("Enter Age:")
x3 = st.number_input("How long do you sleep?:")
x4 = st.number_input("What is your Sleep Quality (1 to 5):", value=0.0)
x5 = st.number_input("Do you have a sleep disorder? (No -> 0 | Yes -> 1):")
x6 = st.number_input("Do you wake up during the night? (No -> 0 | Yes -> 1):")
x7 = st.number_input("Do you feel sleepy during the day? (No -> 0 | Yes -> 1):")
x8 = st.number_input("Have you consumed Caffeine lately? (No -> 0 | Yes -> 1):")
x9 = st.number_input("Do you consume Alcohol? (No -> 0 | Yes -> 1):")
x10 = st.number_input("Do you Smoke? (No -> 0 | Yes -> 1):")
x11 = st.number_input("Do you a medical issue? (No -> 0 | Yes -> 1):")
x12 = st.number_input("Are you ongoing medication? (No -> 0 | Yes -> 1):")
x13 = st.number_input("Do you use a smart device before bed? (No -> 0 | Yes -> 1):")
x14 = st.number_input("What is your average screen time?:")
x15 = st.number_input("Do you use a Blue-Light filter? (No -> 0 | Yes -> 1):")
x16 = st.number_input("Do you have a Discomforting Eye-Strain? (No -> 0 | Yes -> 1):")
x17 = st.number_input("Do you have Redness in your eyes? (No -> 0 | Yes -> 1):")
x18 = st.number_input("Are your eyes Itching or Irritating? (No -> 0 | Yes -> 1):")

if st.button("Predict"):
    new_data = np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18]])
    new_data_df = pd.DataFrame(new_data, columns=feature_names)

    new_data_scaled = scaler.transform(new_data_df)

    prediction = model.predict(new_data_scaled)

    output = "Dry Eye Disease" if prediction == 1 else "No Dry Eye Disease"
    # print("Prediction:", "Dry Eye Disease" if prediction == 1 else "No Dry Eye Disease")
    st.success(f"The predicted output is: {output}")