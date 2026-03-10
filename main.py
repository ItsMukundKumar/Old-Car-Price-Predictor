import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Old Car Price Predictor")
st.write("Predict the selling price of a used car")

# Dropdown options from encoders
car_names = encoders["Car_Name"].classes_
fuel_types = encoders["Fuel_Type"].classes_
seller_types = encoders["Seller_Type"].classes_
transmissions = encoders["Transmission"].classes_

# UI layout
col1, col2 = st.columns(2)

with col1:
    car_name = st.selectbox("Car Name", car_names)
    year = st.number_input("Year", 1990, 2025, 2015)
    present_price = st.number_input("Present Price (Lakhs)", min_value=0.0, value=5.0)

with col2:
    kms_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)
    fuel_type = st.selectbox("Fuel Type", fuel_types)
    seller_type = st.selectbox("Seller Type", seller_types)
    transmission = st.selectbox("Transmission", transmissions)

# Predict button
if st.button("Predict Selling Price"):

    car_encoded = encoders["Car_Name"].transform([car_name])[0]
    fuel_encoded = encoders["Fuel_Type"].transform([fuel_type])[0]
    seller_encoded = encoders["Seller_Type"].transform([seller_type])[0]
    trans_encoded = encoders["Transmission"].transform([transmission])[0]

    input_data = np.array([[car_encoded,
                            year,
                            present_price,
                            kms_driven,
                            fuel_encoded,
                            seller_encoded,
                            trans_encoded]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")