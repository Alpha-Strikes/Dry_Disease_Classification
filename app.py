import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Loading the trained model
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = ['Gender', 'Age', 'Sleep duration', 'Sleep quality', 'Sleep disorder', 'Wake up during night', 'Feel sleepy during day', 'Caffeine consumption', 'Alcohol consumption', 'Smoking', 'Medical issue', 'Ongoing medication', 'Smart device before bed', 'Average screen time', 'Blue-light filter', 'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye']

new_data = np.array([[1, 21, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 1, 1, 0, 0]])
new_data_df = pd.DataFrame(new_data, columns=feature_names)

new_data_scaled = scaler.transform(new_data_df)

prediction = model.predict(new_data_scaled)

print("Prediction:", "Dry Eye Disease" if prediction == 1 else "No Dry Eye Disease")