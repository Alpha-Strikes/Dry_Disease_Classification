import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Loading the trained model
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = ['Gender', 'Age', 'Sleep duration', 'Sleep quality', 'Sleep disorder', 'Wake up during night', 'Feel sleepy during day', 'Caffeine consumption', 'Alcohol consumption', 'Smoking', 'Medical issue', 'Ongoing medication', 'Smart device before bed', 'Average screen time', 'Blue-light filter', 'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye']

x1 = st.number_input("Enter x1 (numeric value):", value=0.0)
x2 = st.number_input("Enter x2 (numeric value):", value=0.0)
x3 = st.number_input("Enter x3 (numeric value):", value=0.0)
x4 = st.number_input("Enter x4 (numeric value):", value=0.0)
x5 = st.number_input("Enter x5 (numeric value):", value=0.0)
x6 = st.number_input("Enter x6 (numeric value):", value=0.0)
x7 = st.number_input("Enter x7 (numeric value):", value=0.0)
x8 = st.number_input("Enter x8 (numeric value):", value=0.0)
x9 = st.number_input("Enter x9 (numeric value):", value=0.0)
x10 = st.number_input("Enter x10 (numeric value):", value=0.0)
x11 = st.number_input("Enter x11 (numeric value):", value=0.0)
x12 = st.number_input("Enter x12 (numeric value):", value=0.0)
x13 = st.number_input("Enter x12 (numeric value):", value=0.0)
x14 = st.number_input("Enter x14 (numeric value):", value=0.0)
x15 = st.number_input("Enter x15 (numeric value):", value=0.0)
x16 = st.number_input("Enter x16 (numeric value):", value=0.0)
x17 = st.number_input("Enter x17 (numeric value):", value=0.0)

if st.button("Predict"):
    new_data = np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17]])
    new_data_df = pd.DataFrame(new_data, columns=feature_names)

    new_data_scaled = scaler.transform(new_data_df)

    prediction = model.predict(new_data_scaled)

    output = "Dry Eye Disease" if prediction == 1 else "No Dry Eye Disease"
    # print("Prediction:", "Dry Eye Disease" if prediction == 1 else "No Dry Eye Disease")
    st.success(f"The predicted output is: {output}")