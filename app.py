import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


# Loading the trained model
model = joblib.load("decision_tree_model.pkl")

new_data = np.array([[1, 21, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 1, 1, 0, 0]])
new_data_df = pd.DataFrame(new_data, columns=feature_names)

scaler = MinMaxScaler()
new_data_scaled = scaler.transform(new_data_df)

prediction = model.predict(new_data_scaled)

print("Prediction:", "Dry Eye Disease" if prediction == 1 else "No Dry Eye Disease")