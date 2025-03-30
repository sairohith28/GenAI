# Required Libraries
import autogen
import pandas as pd
import requests
import json
import logging
from datetime import datetime
import streamlit as st
from gtts import gTTS
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM API Configuration
LLM_API_URL = "http://203.112.158.104:5006/v1/chat/completions"
LLM_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"

# Function to Call LLM API
def call_llm_api(prompt, system_message="You are an expert AI assistant", max_tokens=150):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": "Bearer apex@#1",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return f"Error: {str(e)}"

# Load Datasets
daily_reminders = pd.read_csv("/Users/hitty/hitty_code/hackathon/daily_reminder.csv")
health_data = pd.read_csv("/Users/hitty/hitty_code/hackathon/health_monitoring.csv")
safety_data = pd.read_csv("/Users/hitty/hitty_code/hackathon/safety_monitoring.csv")

# AutoGen Agent Configurations
config_list = [
    {
        "model": LLM_MODEL,
        "api_key": "apex@#1",  # Dummy key since we use Bearer token
        "base_url": LLM_API_URL,
        "api_type": "custom",  # Custom endpoint
    }
]

# Health Monitoring Agent
health_agent = autogen.AssistantAgent(
    name="HealthAgent",
    system_message="You are a health monitoring expert. Analyze health data and alert if thresholds are breached.",
    llm_config={"config_list": config_list}
)

# Safety Monitoring Agent
safety_agent = autogen.AssistantAgent(
    name="SafetyAgent",
    system_message="You are a safety expert. Detect falls or unusual inactivity and notify caregivers.",
    llm_config={"config_list": config_list}
)

# Reminder Agent
reminder_agent = autogen.AssistantAgent(
    name="ReminderAgent",
    system_message="You are a reminder assistant. Send reminders for daily activities and track acknowledgments.",
    llm_config={"config_list": config_list}
)

# Coordinator Agent
coordinator_agent = autogen.AssistantAgent(
    name="CoordinatorAgent",
    system_message="You coordinate between health, safety, and reminder agents. Notify caregivers/family if needed.",
    llm_config={"config_list": config_list}
)

# Wellness Buddy Agent (Creative Addition)
wellness_agent = autogen.AssistantAgent(
    name="WellnessBuddy",
    system_message="You are a friendly companion. Provide motivational quotes or fun facts to keep the elderly engaged.",
    llm_config={"config_list": config_list}
)

# User Proxy Agent to Simulate Elderly User
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False
)

# Function to Generate Voice Reminder
def generate_voice_reminder(text, filename="reminder.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# Multi-Agent Workflow
def run_multi_agent_system(user_id):
    # Health Monitoring
    health_subset = health_data[health_data["Device-ID/User-ID"] == user_id].iloc[-1]
    health_prompt = f"Analyze this health data: Heart Rate: {health_subset['Heart Rate']}, BP: {health_subset['Blood Pressure']}, Glucose: {health_subset['Glucose Levels']}. Thresholds breached? {health_subset['Heart Rate Below/Above Threshold (Yes/No)']}, {health_subset['Blood Pressure Below/Above Threshold (Yes/No)']}, {health_subset['Glucose Levels Below/Above Threshold (Yes/No)']}"
    health_response = call_llm_api(health_prompt, system_message=health_agent.system_message)

    # Safety Monitoring
    safety_subset = safety_data[safety_data["Device-ID/User-ID"] == user_id].iloc[-1]
    safety_prompt = f"Analyze this safety data: Movement: {safety_subset['Movement Activity']}, Fall Detected: {safety_subset['Fall Detected (Yes/No)']}, Inactivity Duration: {safety_subset['Post-Fall Inactivity Duration (Seconds)']} seconds. Should an alert be triggered?"
    safety_response = call_llm_api(safety_prompt, system_message=safety_agent.system_message)

    # Reminder Management
    reminder_subset = daily_reminders[daily_reminders["Device-ID/User-ID"] == user_id].iloc[-1]
    reminder_prompt = f"Check this reminder: Type: {reminder_subset['Reminder Type']}, Scheduled Time: {reminder_subset['Scheduled Time']}, Sent: {reminder_subset['Reminder Sent (Yes/No)']}, Acknowledged: {reminder_subset['Acknowledged (Yes/No)']}. Should a reminder be sent now?"
    reminder_response = call_llm_api(reminder_prompt, system_message=reminder_agent.system_message)

    # Wellness Buddy
    wellness_prompt = "Provide a motivational quote or fun fact for an elderly person."
    wellness_response = call_llm_api(wellness_prompt, system_message=wellness_agent.system_message)

    # Coordinator Decision
    coordinator_prompt = f"Health: {health_response}\nSafety: {safety_response}\nReminder: {reminder_response}\nShould caregivers be notified?"
    coordinator_response = call_llm_api(coordinator_prompt, system_message=coordinator_agent.system_message)

    return health_response, safety_response, reminder_response, wellness_response, coordinator_response

# Streamlit UI
def main():
    st.set_page_config(page_title="ElderCare AI", layout="wide")
    st.title("🌟 ElderCare AI: Your Smart Companion for Independent Living")
    st.markdown("A multi-agent system to monitor health, ensure safety, and manage daily routines for the elderly.")

    # Sidebar for User Selection
    user_id = st.sidebar.selectbox("Select User/Device ID", daily_reminders["Device-ID/User-ID"].unique())

    # Real-Time Dashboard
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Health Monitoring")
        health_response, safety_response, reminder_response, wellness_response, coordinator_response = run_multi_agent_system(user_id)
        st.write(health_response)
        # Visualization
        fig, ax = plt.subplots()
        sns.lineplot(data=health_data[health_data["Device-ID/User-ID"] == user_id], x="Timestamp", y="Heart Rate", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("🛡️ Safety Monitoring")
        st.write(safety_response)

    st.subheader("⏰ Daily Reminders")
    st.write(reminder_response)
    if "send a reminder" in reminder_response.lower():
        reminder_text = f"Reminder: Time for your {daily_reminders[daily_reminders['Device-ID/User-ID'] == user_id].iloc[-1]['Reminder Type']}!"
        st.audio(generate_voice_reminder(reminder_text))

    st.subheader("😊 Wellness Buddy")
    st.write(wellness_response)

    st.subheader("📢 Coordinator Updates")
    st.write(coordinator_response)

    # Gamification: Points for Acknowledging Reminders
    if st.button("Acknowledge Reminder"):
        st.success("🎉 +10 Points Earned!")

    # Export Report
    if st.button("Generate Caregiver Report"):
        report = f"Health: {health_response}\nSafety: {safety_response}\nReminders: {reminder_response}\nCoordinator: {coordinator_response}"
        st.download_button("Download Report", report, file_name=f"report_{user_id}.txt")

if __name__ == "__main__":
    main()