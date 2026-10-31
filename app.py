
import streamlit as st
import pandas as pd
import pickle

# Load trained ML model
with open("tree_health_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Tree Health Monitoring",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 IoT-Based Tree Health Monitoring")
st.write("Environmental Stress Classification using Random Forest ML")

st.subheader("Enter Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    soil = st.slider("Soil Moisture (%)", 0, 100, 65)
    gas = st.slider("Gas Level", 50, 600, 120)

with col2:
    temp = st.slider("Temperature (°C)", 15, 45, 27)
    humidity = st.slider("Humidity (%)", 20, 95, 70)

if st.button("Predict Tree Condition"):
    input_data = pd.DataFrame({
        "Soil_Moisture": [soil],
        "Gas_Level": [gas],
        "Temperature": [temp],
        "Humidity": [humidity]
    })

    prediction = model.predict(input_data)[0]

    alerts = {
        "Healthy": "Normal environmental condition",
        "Water Stress": "Check soil moisture level",
        "Heat Stress": "Check temperature and humidity",
        "Environmental Risk": "Check surrounding air quality"
    }

    st.subheader("Prediction Result")
    st.success(prediction)
    st.info(alerts[prediction])

st.caption(
    "Demonstration using synthetic training data. "
    "Not a validated tree disease diagnosis system."
)
