import streamlit as st
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import google.generativeai as genai


# Load ML model
with open("rfmodel_8.sav", "rb") as model_file:
    model = pickle.load(model_file)

# model=joblib.load("rfmodel_8.sav")

# Configure Gemini API
genai.configure(api_key="AIzaSyAHsBQL8k1dfejBFVd7AMIECr3F6m4xPPY")
model_gemini = genai.GenerativeModel("gemini-1.5-pro")

# ESP32 server URL
ESP32_URL = "http://192.168.175.163/"

# Set Streamlit page config
st.set_page_config(page_title="Health Status Predictor", layout="centered")
st.title("🩺 IoT and ML Based Health Monitoring Web App")

st.markdown(
    """
    <style>
    /* Target all label texts */
    label[data-testid="stWidgetLabel"] > div {
        color: #2E8B57;  /* SeaGreen - change this to your preferred color */
        font-weight: bold;
        font-size: 18px;
    }

    /* Style the input box */
    input[type="number"] {
        background-color: #f5fff5;
        color: #000000;
        border: 2px solid #2E8B57;
        border-radius: 8px;
        padding: 8px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <style>
    /* Change the color of the title */
    .stApp h1 {
        color: #2E8B57;  /* SeaGreen or any hex code you like */
        text-align: center;
    }


    </style>
    """,
    unsafe_allow_html=True
)

import base64

# Read and encode the image
with open("bg3.jpg", "rb") as file:
    bg_image = base64.b64encode(file.read()).decode()

# Inject CSS with background image + opacity
st.markdown(
    f"""
    <style>
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        opacity: 0.2;  /* ⬅️ Change this to control image transparency */
        
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Function to fetch and parse ESP32 sensor values
def fetch_esp32_data():
    try:
        res = requests.get(ESP32_URL, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Parse each line
        paragraphs = soup.find_all("p")
        temp = float(paragraphs[0].text.split(":")[1].replace("°C", "").strip().replace('"', ''))
        pulse = int(paragraphs[1].text.split(":")[1].replace("BPM", "").strip().replace('"', ''))
        spo2 = int(paragraphs[2].text.split(":")[1].replace("%", "").strip().replace('"', ''))

        return pulse, temp, spo2

    except Exception as e:
        st.warning(f"⚠️ Error fetching values from ESP32: {e}")
        return None, None, None

# Fetch sensor values
pulse_val, temp_val, spo2_val = fetch_esp32_data()

# Display inputs (values from ESP if available)
pulse = st.number_input("Heart Rate (BPM)", min_value=0, value=pulse_val if pulse_val else 75, step=1, key="pulse")
temperature = st.number_input("Temperature (°C)", min_value=0.0, value=temp_val if temp_val else 36.5, step=0.1, key="temp")
spo2 = st.number_input("SpO₂ (%)", min_value=0, value=spo2_val if spo2_val else 98, step=1, key="spo2")

# Predict button
if st.button("Check Now"):
    # Format for model
    input_data = np.array([[pulse, temperature, spo2]])
    prediction = model.predict(input_data)[0]
    status = "Healthy" if prediction == 0 else "Unhealthy"

    # Show result
    st.markdown(f"### 🧠 Predicted Health Status: {'✅ Healthy' if prediction == 0 else '⚠️ Unhealthy'}")

    # AI prompt for health report
    prompt = f"""
    Generate a brief health report based on the vitals:
    - Heart Rate: {pulse} BPM
    - Temperature: {temperature} °C
    - SpO₂: {spo2}%
    - Health Status: {status}

    Include a short health summary, advice, and if needed, a light prescription or recommendations.
    """

    gemini_response = model_gemini.generate_content(prompt)
    report_text = gemini_response.text

    # Display report
    st.subheader("📄 Health Report")
    st.markdown(report_text)

    # Download as .txt
    buffer = BytesIO()
    buffer.write(report_text.encode())
    buffer.seek(0)

    st.download_button(
        label="⬇️ Download Report",
        data=buffer,
        file_name="health_report.txt",
        mime="text/plain"
    )
