import streamlit as st
import pandas as pd
import numpy as np
import requests
from twilio.rest import Client
from datetime import datetime
import re
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import pickle
import os
# ThingSpeak API
CHANNEL_ID = "2885781"
READ_API_KEY = "LP39XB8DD3ARDXDK"


# Twilio Configuration
TWILIO_PHONE_NUMBER = "whatsapp:+14155238886"  # Your Twilio WhatsApp number
AUTH_TOKEN = "1602c5b71508bcc28efc1e79657f2ea4"  # Your Twilio Auth Token
SID = "AC3d305967e3e0b493a29623a00409c805"  # Your Twilio SID
client = Client(SID, AUTH_TOKEN)

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'password' not in st.session_state:
    st.session_state.password = "admin123"
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'language' not in st.session_state:
    st.session_state.language = "English"

# Language support
def t(text):
    tamil_dict = {
        "Login": "உள்நுழைய",
        "Username": "பயனர் பெயர்",
        "Password": "கடவுச்சொல்",
        "Login Successful": "உள்நுழைவு வெற்றிகரமாக முடிந்தது",
        "Invalid Credentials": "தவறான தகவல்கள்",
        "Firework Industry Lightning Prediction Dahboard": "பட்டாசு தொழில் மின்காற்று முன்னறிவிப்பு டாஷ்போர்டு",
        "Navigation": "வழிசெலுத்தல்",
        "Home": "முகப்பு",
        "Sensor Visualization": "அணுகி தரவுகள்",
        "Threshold Alerts": "அவதானிப்பு அலாரங்கள்",
        "Settings": "அமைப்புகள்",
        "Logout": "வெளியேறு",
        "Welcome to the Smart Agriculture Monitoring System!": "スマート வேளாண்மை கண்காணிப்பு முறைமைக்கு வரவேற்கிறோம்!",
        "Set Threshold Alerts": "அவதானிப்பு அலாரங்களை அமைக்கவும்",
        "Check Alerts": "அலாரங்களை சரிபார்க்கவும்",
        "Language": "மொழி",
        "Theme": "தீம்",
        "Change Password": "கடவுச்சொல் மாற்றவும்",
        "Current Password": "தற்போதைய கடவுச்சொல்",
        "New Password": "புதிய கடவுச்சொல்",
        "Update Password": "கடவுச்சொல் புதுப்பிக்கவும்",
        "Apply Settings": "அமைப்புகளை செயல்படுத்து",
        "Download CSV": "CSV பதிவிறக்குக",
        "Sensor Data": "அணுகி தரவுகள்",
        "Date": "தேதி",
        "Time": "நேரம்",
    }
    return tamil_dict.get(text, text) if st.session_state.language == "Tamil" else text


pairs = [
    (r"Hi|Hello|Hey", ["Hello! How can I assist you with the Lightning Prediction System today?"]),

    (r"Tell me about the Lightning Prediction System", [
        "This system predicts potential lightning hazards in firework industries using sensors and automates safety responses like activating a water pump."]),

    (r"What is the flame sensor?", [
        "The flame sensor detects fire or flame in the surroundings and triggers alerts or activates the water pump to reduce the risk."]),

    (r"What is the rain sensor?", [
        "The rain detection sensor identifies rainfall, helping in weather prediction and real-time monitoring."]),

    (r"What is the DHT sensor?", [
        "The DHT sensor measures temperature and humidity, which are key factors in lightning and weather prediction."]),

    (r"What does the sound sensor do?", [
        "The sound sensor detects loud thunder-like noises, which can help indicate possible lightning activity."]),

    (r"What is the water level sensor for?", [
        "The water level sensor monitors water buildup, especially after rain, and helps in flood or overflow detection."]),

    (r"What does the voltage sensor do?", [
        "The voltage sensor monitors sudden electrical spikes which could be caused by lightning or power surges."]),

    (r"What is the current sensor used for?", [
        "The current sensor measures electrical current to identify abnormal patterns possibly caused by lightning."]),

    (r"What is the ESP8266 used for?", [
        "The ESP8266 module connects the system to the internet for real-time data transmission and remote monitoring."]),

    (r"What is the Arduino's role?", [
        "The Arduino processes data from sensors and controls components like the water pump based on sensor inputs."]),

    (r"What is the use of the water pump motor?", [
        "The water pump is activated automatically in case of fire detection to extinguish flames or prevent spread."]),

    (r"What powers the system?", [
        "The system is powered by a 12V transformer, supplying adequate power to sensors and the water pump."]),

    (r"Can I get real-time sensor data?", [
        "Yes, real-time sensor data is available on the dashboard, including temperature, humidity, sound levels, flame, voltage, and more."]),

    (r"How does the system prevent lightning damage?", [
        "The system monitors weather and electrical indicators to predict lightning and activates preventive responses such as alerts and water pump activation."]),

    (r"Can I receive alerts?", [
        "Yes! You will get alerts when any sensor crosses a predefined threshold – for example, flame detection or sudden voltage spikes."]),

    (r"Can I download sensor data?", [
        "Yes, you can download all logged sensor data from the dashboard for analysis and reporting."]),

    (r"Bye|Goodbye", ["Goodbye! Stay safe from lightning hazards!"]),

    (r"(.*)", [
        "Sorry, I didn't understand that. Could you please rephrase or ask something related to the lightning prediction system?"]),
]


# Function to get chatbot response
def chatbot_response(user_input):
    for pattern, responses in pairs:
        if re.match(pattern, user_input, re.I):
            return responses[0]
    return "Sorry, I didn't understand that. Can you please clarify?"

# Fetch data from ThingSpeak
def fetch_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=10"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()["feeds"]
        df = pd.DataFrame(data)
        df = df.rename(columns={
            'field1': 'Temperature',
            'field2': 'Humidity',
            'field3': 'Rain',
            'field4': 'Flame',
            'field5': 'Water Level',
            'field6': 'Thunder Sound',
            'field7': 'Voltage Drop',
            'field8': 'Current Drop'
        })
        df['created_at'] = pd.to_datetime(df['created_at'])
        df.set_index('created_at', inplace=True)
        return df
    else:
        st.error("❌ Failed to fetch data from ThingSpeak")
        return None

# Send Twilio Alert
def send_whatsapp_alert(to, message):
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to
        )
        st.success(f"Alert sent to {to}: {message}")
    except Exception as e:
        st.error(f"Failed to send message: {e}")

# Login Page
def login_page():
    st.title(t("Login"))
    username = st.text_input(t("Username"))
    password = st.text_input(t("Password"), type="password")
    if st.button(t("Login")):
        if username == "admin" and password == st.session_state.password:
            st.session_state.logged_in = True
            st.success(t("Login Successful"))
            st.rerun()
        else:
            st.error(t("Invalid Credentials"))

# App Pages
def data_analytics():
    st.title("📊 Data Analytics")

    # Fetch real-time data from ThingSpeak
    df = fetch_data()

    if df is not None:
        # Convert all relevant columns to numeric
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
        df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')
        df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce')
        df['Flame'] = pd.to_numeric(df['Flame'], errors='coerce')
        df['Water Level'] = pd.to_numeric(df['Water Level'], errors='coerce')
        df['Thunder Sound'] = pd.to_numeric(df['Thunder Sound'], errors='coerce')
        df['Voltage Drop'] = pd.to_numeric(df['Voltage Drop'], errors='coerce')
        df['Current Drop'] = pd.to_numeric(df['Current Drop'], errors='coerce')

        # Drop rows with missing data in the main fields
        df = df.dropna(subset=['Temperature', 'Humidity', 'Rain', 'Flame', 'Water Level'])

        st.subheader("📈 Sensor Data Analytics")

        # Updated analysis options
        analysis_option = st.selectbox(
            "Choose Analysis Type",
            [
                "Temperature Trends",
                "Humidity Trends",
                "Rain Level Analysis",
                "Flame Sensor Events",
                "Water Level Monitoring",
                "Thunder Sound Events",
                "Voltage Drop Analysis",
                "Current Drop Analysis"
            ]
        )

        if analysis_option == "Temperature Trends":
            st.subheader("🌡️ Temperature Trends")
            fig, ax = plt.subplots()
            df['Temperature'].plot(ax=ax, color='red')
            ax.set_title("Temperature Over Time")
            ax.set_ylabel("Temperature (°C)")
            st.pyplot(fig)

        elif analysis_option == "Humidity Trends":
            st.subheader("💧 Humidity Trends")
            fig, ax = plt.subplots()
            df['Humidity'].plot(ax=ax, color='blue')
            ax.set_title("Humidity Over Time")
            ax.set_ylabel("Humidity (%)")
            st.pyplot(fig)

        elif analysis_option == "Rain Level Analysis":
            st.subheader("🌧️ Rain Level Analysis")
            fig, ax = plt.subplots()
            df['Rain'].plot(ax=ax, color='purple')
            ax.set_title("Rain Level Over Time")
            ax.set_ylabel("Rain Level")
            st.pyplot(fig)

        elif analysis_option == "Flame Sensor Events":
            st.subheader("🔥 Flame Sensor Events")
            flame_events = df[df['Flame'] > 0]
            st.write(f"Detected {len(flame_events)} flame events.")
            st.dataframe(flame_events)

        elif analysis_option == "Water Level Monitoring":
            st.subheader("💦 Water Level Monitoring")
            fig, ax = plt.subplots()
            df['Water Level'].plot(ax=ax, color='cyan')
            ax.set_title("Water Level Over Time")
            ax.set_ylabel("Water Level")
            st.pyplot(fig)

        elif analysis_option == "Thunder Sound Events":
            st.subheader("⚡ Thunder Sound Events")
            fig, ax = plt.subplots()
            df['Thunder Sound'].plot(ax=ax, color='orange')
            ax.set_title("Thunder Sound Intensity Over Time")
            ax.set_ylabel("Sound Level")
            st.pyplot(fig)

        elif analysis_option == "Voltage Drop Analysis":
            st.subheader("🔋 Voltage Drop Analysis")
            fig, ax = plt.subplots()
            df['Voltage Drop'].plot(ax=ax, color='magenta')
            ax.set_title("Voltage Drop Over Time")
            ax.set_ylabel("Voltage Drop (V)")
            st.pyplot(fig)

        elif analysis_option == "Current Drop Analysis":
            st.subheader("🔌 Current Drop Analysis")
            fig, ax = plt.subplots()
            df['Current Drop'].plot(ax=ax, color='brown')
            ax.set_title("Current Drop Over Time")
            ax.set_ylabel("Current Drop (A)")
            st.pyplot(fig)

        # Option to download the analytics data
        if st.button("Download Analytics Data"):
            st.download_button(
                label="📥 Download Analytics Data",
                data=df.to_csv().encode('utf-8'),
                file_name="firework_lightning_analytics.csv",
                mime="text/csv"
            )


def about_section():
    st.title("🔍 About This Project")

    st.subheader("Project Overview")
    st.write("""
        This project is titled **"Firework Industry Lightning Prediction and Protection System"**. It is designed to monitor and predict lightning and fire-related hazards in fireworks manufacturing environments using a variety of IoT sensors.

        The system integrates multiple sensors to track weather parameters such as temperature, humidity, rainfall, sound of thunder, voltage drops, current fluctuations, and flame detection. The collected real-time data is transmitted to the cloud via ThingSpeak using the ESP8266 module.

        Based on critical thresholds, the system activates alerts and safety mechanisms such as water pump motors to mitigate fire hazards. This proactive system provides real-time monitoring, analytics, and alerts for early warning and response, ensuring safety in the high-risk firework industry.
    """)

    st.subheader("🔧 Components Used")
    st.write("""
        - **Arduino Uno**: Main microcontroller for data collection and control.
        - **ESP8266 Wi-Fi Module**: Sends real-time sensor data to the ThingSpeak cloud.

        **Sensors:**
        - **DHT11 Sensor**: Measures temperature and humidity.
        - **Rain Detection Sensor**: Detects presence and intensity of rainfall.
        - **Sound Sensor**: Detects thunder-related sound vibrations.
        - **Water Level Sensor**: Monitors water levels for safety and pump control.
        - **Flame Sensor**: Detects presence of flame/fire.
        - **Voltage Sensor**: Detects voltage drop that may be caused by lightning or electrical surges.
        - **Current Sensor**: Detects current variation, helpful for identifying abnormal load behavior.

        **Actuators and Power:**
        - **Water Pump Motor**: Activated during flame detection for emergency fire suppression.
        - **12V Transformer**: Provides regulated power supply for the circuit and components.
    """)

    st.subheader("⚙️ Working of the System")
    st.write("""
        The system continuously monitors the environment using the integrated sensors. The Arduino Uno collects sensor readings and sends the data to the ThingSpeak IoT platform via ESP8266 for visualization and remote monitoring.

        When thresholds are exceeded (e.g., flame detection, loud thunder, abnormal voltage/current levels), the system triggers real-time alerts and can activate a water pump motor as a response mechanism.

        This project provides a preventive solution by predicting potential lightning activity and responding to fire threats in real time, enhancing safety in fireworks production facilities.
    """)


def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    else:
        st.warning("ML model not found. Please train the model and place 'model.pkl' in the same folder.")
        return None

# Load the pre-trained AI model
#model = joblib.load('irrigation_model.pkl')

# AI Prediction Section in Streamlit

# Fetch Data
data = fetch_data()
def ai_prediction():
    st.title("🔮 Lightning & Thunder AI Weather Prediction")

    st.subheader("🧠 AI-Based 1-Hour Prediction & Alerts")
    model = load_model()

    if model and not data.empty:
        latest = data.iloc[-1]

        # Convert to numeric to avoid type errors
        for col in ["Temperature", "Humidity", "Rain", "Flame", "Water Level", "Thunder Sound", "Voltage Drop", "Current Drop"]:
            latest[col] = pd.to_numeric(latest[col], errors='coerce')

        # Real-time Prediction
        st.markdown("### 🔍 Real-time Prediction")
        input_data = latest[["Temperature", "Humidity", "Rain", "Flame", "Water Level", "Thunder Sound", "Voltage Drop", "Current Drop"]].values.reshape(1, -1)
        prediction = model.predict(input_data)[0]
        st.success(f"🤖 AI Prediction Output: **{prediction}**")

        # Fire Alert
        flame = float(latest["Flame"])
        if flame > 500:
            st.error("🚨 FIRE ALERT DETECTED!")
        else:
            st.info("✅ No fire detected currently.")

        # Thunderstorm Alert
        humidity = float(latest["Humidity"])
        rain = float(latest["Rain"])
        sound = float(latest["Thunder Sound"])
        if humidity > 70 and rain > 5 and sound > 500:
            st.warning("⚡ Thunderstorm possible in next 1 hour.")
        else:
            st.success("☀️ No thunderstorm expected in the next hour.")

        # Manual Input Section
        st.markdown("---")
        st.markdown("### ✍️ Manual Sensor Input for AI Prediction")
        col1, col2, col3, col4 = st.columns(4)
        temperature = col1.number_input("Temperature", value=30.0)
        humidity = col2.number_input("Humidity", value=50.0)
        rain = col3.number_input("Rain", value=0.0)
        flame = col4.number_input("Flame", value=200.0)
        water = col1.number_input("Water Level", value=30.0)
        sound = col2.number_input("Thunder Sound", value=100.0)
        voltage = col3.number_input("Voltage Drop", value=220.0)
        current = col4.number_input("Current Drop", value=0.5)

        if st.button("📡 Predict with Manual Input"):
            manual_data = np.array([[temperature, humidity, rain, flame, water, sound, voltage, current]])
            manual_pred = model.predict(manual_data)[0]
            st.success(f"📊 AI Prediction Output: **{manual_pred}**")

            if flame > 500:
                st.error("🚨 FIRE ALERT DETECTED!")
            else:
                st.info("✅ No fire based on input.")

            if humidity > 70 and rain > 5 and sound > 500:
                st.warning("⚡ Thunderstorm may occur in next 1 hour.")
            else:
                st.success("☀️ No thunderstorm expected.")

    else:
        st.warning("No model or data available for prediction.")

def electricity_consumption():
    st.title("⚡ Electricity Consumption Monitor")

    df = fetch_data()
    if df is None or df.empty:
        st.warning("No data available.")
        return

    # Filter last 10 entries for recent analysis
    recent_data = df.tail(10)

    water_pump_on_time = 0  # seconds
    interval = 15  # assuming each reading is 15 seconds apart

    for _, row in recent_data.iterrows():
        try:
            flame = float(row['Flame'])
        except:
            flame = 0

        if flame > 0:
            water_pump_on_time += 10  # 10 seconds if flame is detected

    # Convert seconds to hours
    pump_time_hr = water_pump_on_time / 3600

    # Power Consumption Calculation
    pump_power = 50  # in Watts
    voltage = 12     # system voltage in Volts
    current = pump_power / voltage  # in Amperes

    pump_kwh = (pump_power * pump_time_hr) / 1000  # kWh

    # Cost Calculation
    unit_rate = 8  # ₹ per kWh
    cost = pump_kwh * unit_rate

    # Display results
    st.subheader("🚿 Water Pump Motor (Triggered by Flame Detection)")
    st.write(f"Total ON Time: {water_pump_on_time} seconds")
    st.write(f"Estimated Power Consumed: {pump_kwh:.4f} kWh")
    st.write(f"Estimated Current Usage: {current:.2f} A")
    st.write(f"Estimated Cost: ₹{cost:.2f}")

    st.success(f"💰 Total Estimated Electricity Cost: ₹{cost:.2f}")


def dashboard():
    st.set_page_config(page_title="Lightning Prediction Dashboard", layout="wide")
    if st.session_state.theme == "Dark":
        st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

    st.sidebar.title("🌾 " + t("Firework Factory Lightning Protection System"))
    menu = [
        f"🌱 {t('Home')}",
        f"📊 {t('Sensor Visualization')}",
        f"🚨 {t('Threshold Alerts')}",
        f"📊 {t('Data Analytics')}",  # New section
        f"🔮 {t('AI Prediction')}",  # Add the AI Prediction section
        f"⚡  {t('Electricity Consumption')}",
        f"🔍 {t('About This Project')}",
        f"⚙️ {t('Settings')}",
        f"🔓 {t('Logout')}"
    ]

    choice = st.sidebar.radio(t("Navigation"), menu)

    if choice.endswith(t("Home")):
        st.title("🌱 " + t("Home"))

        # Product Description
        st.subheader("🚜 " + t("Product Description"))
        st.write(
            "Lightning Protection system is designed to monitor and protect environments using real-time sensor data. It tracks temperature, humidity, rain, flame, water level, thunder sound, voltage, and current. When a fire is detected, the system automatically activates a water pump to prevent damage. Data is sent to ThingSpeak and displayed on a Streamlit dashboard, along with alerts and AI-based weather predictions. Ideal for safety-critical applications like firework industries.")
        # Contact Details
        st.subheader("📞 " + t("Contact Us"))
        st.write("""
            **Support Team:**
            - Email: projectiot2k25@gmail.com
            - Phone: +91 8667633858
            - Address: Jayaraj Annapackiam CSI College Of Engineering, Margoschis Nagar, Nazareth, Tamilnadu, India 628 617.
        """)

        # Real-time Sensor Data
        df = fetch_data()
        if df is not None:
            st.subheader("📊 " + t("Real-Time Sensor Data"))

            # Horizontal layout for real-time monitoring
            cols = st.columns(4)
            latest_data = df.iloc[-1]  # Get the latest data from the dataframe

            sensors = [
                "Temperature", "Humidity", "Rain", "Flame",
                "Water Level", "Thunder Sound", "Voltage Drop", "Current Drop"
            ]
            for i, sensor in enumerate(sensors):
                with cols[i % 4]:
                    value = latest_data[sensor]
                    st.metric(sensor, f"{value}")

        # Display date and time of data retrieval
        st.write("---")
        st.metric("Date", datetime.now().strftime("%Y-%m-%d"))
        st.metric("Time", datetime.now().strftime("%H:%M:%S"))

        # Chatbot section
        st.subheader("🤖 " + t("Chat with our Assistant"))
        user_input = st.text_input("Ask me anything about the Lighting Protection System:")
        if user_input:
            response = chatbot_response(user_input)
            st.write(f"Bot: {response}")

    elif choice.endswith(t("Sensor Visualization")):
        st.title("📊 " + t("Sensor Visualization"))
        df = fetch_data()
        if df is not None:
            sensors = ['Temperature', 'Humidity', 'Rain', 'Flame', 'Water Level', 'Thunder Sound', 'Voltage Drop',
                       'Current Drop']
            cols = st.columns(4)
            for i, sensor in enumerate(sensors):
                with cols[i % 4]:
                    if st.button(f"📈 {sensor}"):
                        st.session_state.selected_sensor = sensor
                        st.session_state.df = df
                        st.rerun()

            st.write("---")
            if st.button(t("Download CSV")):
                st.download_button(
                    label="📥 " + t("Download CSV"),
                    data=df.to_csv().encode('utf-8'),
                    file_name="sensor_data.csv",
                    mime="text/csv"
                )

            if 'selected_sensor' in st.session_state:
                selected_sensor = st.session_state.selected_sensor
                df = st.session_state.df
                st.subheader(f"{selected_sensor} Data")
                st.line_chart(df[selected_sensor])
                latest_value = df[selected_sensor].iloc[-1]

                if selected_sensor == "Flame" and latest_value > 1:
                    st.error(f"🔥 {selected_sensor} is too high! Value: {latest_value}")
                elif selected_sensor == "Thunder Sound" and latest_value < 30:
                    st.warning(f"💧 {selected_sensor} is low. Value: {latest_value}")
                else:
                    st.success(f"✅ {selected_sensor} value is normal: {latest_value}")

    elif choice.endswith(t("Data Analytics")):
        data_analytics()  # Call the function to display the data analytics section

    elif choice.endswith(t("AI Prediction")):
        ai_prediction()

    elif choice.endswith(t("Electricity Consumption")):
        electricity_consumption()

    elif choice.endswith(t("About This Project")):
        about_section()

    elif choice.endswith(t("Threshold Alerts")):
        st.title("🚨" + ("Threshold Alerts"))

        # Set Alert Thresholds
        st.markdown("### 🔧 Set Alert Thresholds")
        temperature_threshold = st.number_input("Temperature Threshold", min_value=0.0, max_value=100.0, value=37.0)
        humidity_threshold = st.number_input("Humidity Threshold", min_value=0.0, max_value=100.0, value=80.0)
        flame_threshold = st.number_input("Flame Threshold", min_value=0.0, max_value=1.0, value=1.0)

        # Contact Information
        st.markdown("### 📞 Set Up WhatsApp Alert")
        to_whatsapp = st.text_input("Enter WhatsApp Number to Send Alerts", value="whatsapp:+919344789554")
        message = st.text_area("Enter Alert Message",
                               value="Alert! Sensor threshold exceeded. Please check the system.")

        if st.button("Send Test Alert"):
            send_whatsapp_alert(to_whatsapp, message)

        data = fetch_data()
        # Real-time Alert Logic
        if not data.empty:
            latest = data.iloc[-1]

            try:
                flame = float(latest["Flame"])
                temp = float(latest["Temperature"])
                humidity = float(latest["Humidity"])
            except (ValueError, TypeError):
                st.warning("Sensor data contains invalid values.")
                return

            # Fire Alert
            if flame > flame_threshold:
                send_whatsapp_alert(to_whatsapp, f"🚨 FIRE ALERT DETECTED! Current Flame: {flame}")

            # Temperature Alert
            if temp > temperature_threshold:
                send_whatsapp_alert(to_whatsapp, f"⚠️ High Temperature Alert! Current Temp: {temp}°C")

            # Humidity Alert
            if humidity > humidity_threshold:
                send_whatsapp_alert(to_whatsapp, f"⚠️ High Humidity Alert! Current Humidity: {humidity}%")

        else:
            st.warning("No data available for alert evaluation.")


    elif choice.endswith(t("About The Project")):
        about_section()

    elif choice.endswith(t("Settings")):
        st.title("⚙️ " + t("Settings"))
        st.subheader(t("Change Theme"))
        theme = st.radio(t("Theme"), ["Light", "Dark"])
        st.session_state.theme = theme

        st.subheader(t("Change Language"))
        language = st.radio(t("Language"), ["English", "Tamil"])
        st.session_state.language = language

        st.subheader(t("Change Password"))
        current_password = st.text_input(t("Current Password"), type="password")
        new_password = st.text_input(t("New Password"), type="password")
        if st.button(t("Update Password")):
            if current_password == st.session_state.password:
                st.session_state.password = new_password
                st.success(t("Password Updated"))
            else:
                st.error(t("Incorrect Current Password"))




    elif choice.endswith(t("Logout")):
        st.session_state.logged_in = False
        st.success(t("Logged out successfully"))
        st.rerun()


# Main Program
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()


