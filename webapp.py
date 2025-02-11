import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("car_price_model.pkl")

# Title of the Web App
st.title("Car Price Prediction App")
st.write("Enter the details of the car to predict its selling price.")

# User input fields
year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, step=1, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000, value=50000)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1, value=20.0)
engine = st.number_input("Engine Capacity (CC)", min_value=500, step=100, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, step=5.0, value=80.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=14, step=1, value=5)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# Encode categorical variables
fuel_mapping = {"Petrol": 3, "Diesel": 1, "CNG": 0, "LPG": 2, "Electric": 4}
seller_mapping = {"Dealer": 0, "Individual": 1, "Trustmark Dealer": 2}
transmission_mapping = {"Manual": 1, "Automatic": 0}
owner_mapping = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2, "Fourth & Above Owner": 3, "Test Drive Car": 4}

fuel_encoded = fuel_mapping[fuel]
seller_encoded = seller_mapping[seller_type]
transmission_encoded = transmission_mapping[transmission]
owner_encoded = owner_mapping[owner]

# Make prediction
if st.button("Predict Price"):
    input_data = np.array([[year, km_driven, mileage, engine, max_power, seats, fuel_encoded, seller_encoded, transmission_encoded, owner_encoded]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Selling Price: ₹{prediction[0]:,.2f}")
