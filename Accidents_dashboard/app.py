import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("rf_accident_pipeline.pkl")

# =======================
# Page Config
# =======================
st.set_page_config(page_title="🚦 Smart Road Accident Predictor", layout="centered")

st.title("🚦 Smart Road Accident Predictor")
st.markdown("### Predict the likelihood of a road accident based on live conditions 🚗💨")

# =======================
# Sidebar - User Inputs
# =======================
st.sidebar.header("📊 Input Parameters")

traffic_volume = st.sidebar.number_input("Traffic Volume", min_value=0, value=3000)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
is_weekend = st.sidebar.selectbox("Is Weekend?", [0, 1])
temp = st.sidebar.number_input("Temperature (K)", value=295)
rain_1h = st.sidebar.number_input("Rain in last hour (mm)", value=0.0)
snow_1h = st.sidebar.number_input("Snow in last hour (mm)", value=0.0)
clouds_all = st.sidebar.number_input("Cloud coverage (%)", value=50)
weather_main = st.sidebar.selectbox(
    "Weather", 
    ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Snow", "Squall", "Thunderstorm"]
)

threshold = st.sidebar.slider("⚖️ Risk Threshold", 0.1, 0.9, 0.5)

# =======================
# Prediction
# =======================
if st.button("🔍 Predict Accident"):
    input_df = pd.DataFrame([{
        "traffic_volume": traffic_volume,
        "Hour": hour,
        "Is_Weekend": is_weekend,
        "temp": temp,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "weather_main": weather_main
    }])
    
    # Get probability
    prob = pipeline.predict_proba(input_df)[0][1]  # probability of accident
    prediction = int(prob >= threshold)
    
    # =======================
    # Display Results
    # =======================
    st.subheader("📌 Prediction Result")
    st.metric(label="Likelihood of Accident (%)", value=f"{prob*100:.2f}%")

    if prob < 0.3:
        st.success("🟢 Low Risk: Roads are generally safe")
    elif prob < 0.6:
        st.warning("🟡 Medium Risk: Caution advised")
    else:
        st.error("🔴 High Risk: Accident likely! Drive carefully 🚨")

    st.progress(int(prob*100))

# =======================
# About Section
# =======================
st.markdown("---")
st.markdown("### ℹ️ About this Project")
st.write("""
This Smart Road Analytics project predicts **road accident risk** using 
traffic, weather, and time-based factors.  
Built with **Python, Scikit-learn, and Streamlit**.
""")
st.markdown("👨‍💻 Developed by **Maneesh**")
