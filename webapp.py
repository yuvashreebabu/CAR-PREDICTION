import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Streamlit UI
st.title("Sales Prediction Web App")
st.write("Enter the advertising budget to predict sales")

# Input fields
TV = st.number_input("TV Advertising Budget ($)", min_value=0.0, format="%.2f")
Radio = st.number_input("Radio Advertising Budget ($)", min_value=0.0, format="%.2f")
Newspaper = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, format="%.2f")

# Predict function
def predict_sales(tv, radio, newspaper):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)
    return prediction[0]

# Prediction Button
if st.button("Predict Sales"):
    prediction = predict_sales(TV, Radio, Newspaper)
    st.success(f"Predicted Sales: {prediction:.2f} units")
