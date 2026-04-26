import streamlit as st
import numpy as np
import joblib


# Load model files
model = joblib.load("svr_model.pkl")

scaler = joblib.load("scaler.pkl")

pca = joblib.load("pca.pkl")


# Title
st.title("Traffic Signal Time Prediction")


# Inputs
traffic_density = st.number_input("Traffic Density")

vehicle_count = st.number_input("Vehicle Count")

average_speed = st.number_input("Average Speed")

lane_count = st.number_input("Lane Count")

peak_hour = st.number_input("Peak Hour")


# Prediction
if st.button("Predict"):

    data = np.array([[
        traffic_density,
        vehicle_count,
        average_speed,
        lane_count,
        peak_hour
    ]])

    data_scaled = scaler.transform(data)

    data_pca = pca.transform(data_scaled)

    prediction = model.predict(data_pca)

    st.success(
        f"Predicted Green Signal Time: {round(prediction[0],2)} seconds"
    )
