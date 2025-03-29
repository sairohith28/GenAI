import autogen
import requests
import json
from typing import Dict, List
import pandas as pd
from datetime import datetime

# Configuration for your hosted LLM (minimal, since we rely on custom function)
llm_config = {
    "config_list": [{
        "model": "unsloth/Qwen2.5-1.5B-Instruct",
        "api_key": "apex@#1"  # Kept for compatibility, but we'll override with custom function
    }],
    "timeout": 120,
    "max_tokens": 500,
    "temperature": 0.7,
}

# Custom function to call your hosted LLM with authentication
def call_custom_llm(messages: List[Dict]) -> str:
    payload = {
        "model": "unsloth/Qwen2.5-1.5B-Instruct",
        "messages": messages,
        "max_tokens": 500
    }
    
    headers = {
    "api-key": "apex@#1",
    "Content-Type": "application/json"
}
    
    try:
        response = requests.post(
            "http://203.112.158.104:5006/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling LLM: {str(e)} - Status Code: {e.response.status_code if e.response else 'No response'} - Response: {e.response.text if e.response else 'No response'}"

# Load datasets with error handling
def load_csv_with_fallback(file_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Warning: {file_name} not found. Using empty DataFrame as fallback.")
        return pd.DataFrame()

health_data = load_csv_with_fallback("health_monitoring.csv")
safety_data = load_csv_with_fallback("safety_monitoring.csv")
reminder_data = load_csv_with_fallback("daily_remainder.csv")

# Define agents with custom LLM function
health_agent = autogen.AssistantAgent(
    name="HealthMonitoringAgent",
    system_message="You are a health monitoring specialist. Analyze health data and generate alerts if values exceed thresholds.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False,  # Disable code execution to force custom LLM
    function_map={"call_llm": call_custom_llm}
)

safety_agent = autogen.AssistantAgent(
    name="SafetyMonitoringAgent",
    system_message="You are a safety monitoring expert. Analyze movement data, detect falls, and monitor unusual behavior patterns.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False,
    function_map={"call_llm": call_custom_llm}
)

reminder_agent = autogen.AssistantAgent(
    name="ReminderAgent",
    system_message="You are a personal assistant for elderly care. Manage daily reminders for medication, appointments, and activities.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False,
    function_map={"call_llm": call_custom_llm}
)

coordinator_agent = autogen.AssistantAgent(
    name="CoordinatorAgent",
    system_message="You are the central coordinator for elderly care. Receive inputs from Health, Safety, and Reminder agents, make decisions, and recommend actions.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config=False,
    function_map={"call_llm": call_custom_llm}
)

# Group Chat setup
group_chat = autogen.GroupChat(
    agents=[health_agent, safety_agent, reminder_agent, coordinator_agent],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)

# Helper functions for data processing
def process_health_data(user_id: str) -> str:
    if health_data.empty or user_id not in health_data["Device-ID/User-ID"].values:
        return "No health data available"
    user_data = health_data[health_data["Device-ID/User-ID"] == user_id].iloc[-1]
    alerts = []
    if user_data["Heart Rate Below/Above Threshold (Yes/No)"] == "Yes":
        alerts.append(f"Abnormal heart rate: {user_data['Heart Rate']}")
    if user_data["Blood Pressure Below/Above Threshold (Yes/No)"] == "Yes":
        alerts.append(f"Abnormal BP: {user_data['Blood Pressure']}")
    if user_data["Glucose Levels Below/Above Threshold (Yes/No)"] == "Yes":
        alerts.append(f"Abnormal glucose: {user_data['Glucose Levels']}")
    if user_data["SpO₂ Below Threshold (Yes/No)"] == "Yes":
        alerts.append(f"Low oxygen: {user_data['Oxygen Saturation (SpO₂%)']}")
    return "\n".join(alerts) if alerts else "All health parameters normal"

def process_safety_data(user_id: str) -> str:
    if safety_data.empty or user_id not in safety_data["Device-ID/User-ID"].values:
        return "No safety data available"
    user_data = safety_data[safety_data["Device-ID/User-ID"] == user_id].iloc[-1]
    if user_data["Fall Detected (Yes/No)"] == "Yes":
        return f"Fall detected at {user_data['Location']} with impact force {user_data['Impact Force Level']}"
    elif user_data["Post-Fall Inactivity Duration (Seconds)"] > 300:
        return f"Unusual inactivity detected at {user_data['Location']} for {user_data['Post-Fall Inactivity Duration (Seconds)']} seconds"
    return "No safety concerns detected"

def process_reminder_data(user_id: str) -> str:
    if reminder_data.empty or user_id not in reminder_data["Device-ID/User-ID"].values:
        return "No reminder data available"
    user_data = reminder_data[reminder_data["Device-ID/User-ID"] == user_id].iloc[-1]
    if user_data["Reminder Sent (Yes/No)"] == "Yes" and user_data["Acknowledged (Yes/No)"] == "No":
        return f"Missed {user_data['Reminder Type']} reminder scheduled for {user_data['Scheduled Time']}"
    return "All reminders acknowledged"

# Main execution function
def monitor_elderly(user_id: str):
    initial_message = f"""
    Coordinator: Starting monitoring for user {user_id} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    HealthMonitoringAgent: Please analyze latest health data
    SafetyMonitoringAgent: Please check safety status
    ReminderAgent: Please verify reminder compliance
    """
    
    health_status = process_health_data(user_id)
    safety_status = process_safety_data(user_id)
    reminder_status = process_reminder_data(user_id)
    
    try:
        manager.initiate_chat(
            coordinator_agent,
            message=f"""
            {initial_message}
            HealthMonitoringAgent: {health_status}
            SafetyMonitoringAgent: {safety_status}
            ReminderAgent: {reminder_status}
            Coordinator: Please analyze the combined data and recommend actions
            """
        )
        final_response = group_chat.messages[-1]["content"]
    except Exception as e:
        final_response = f"Error during agent collaboration: {str(e)}"
    
    return final_response

# Example usage
if __name__ == "__main__":
    user_id = "D1000"
    result = monitor_elderly(user_id)
    print(f"Monitoring result for {user_id}:\n{result}")