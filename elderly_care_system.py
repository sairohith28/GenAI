import os
import pandas as pd
import requests
import json
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy and pandas numeric types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return super().default(obj)

# LLM API Configuration
LLM_API_URL = "http://203.112.158.104:5006/v1/chat/completions"  # Corrected endpoint
LLM_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"

# Function to call the LLM API
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

# Data loading and preprocessing functions
def load_data():
    """Load and preprocess the datasets"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        health_data = pd.read_csv(os.path.join(base_path, "health_monitoring.csv"))
        safety_data = pd.read_csv(os.path.join(base_path, "safety_monitoring.csv"))
        reminder_data = pd.read_csv(os.path.join(base_path, "daily_reminder.csv"))
        
        # Convert timestamps to datetime objects
        health_data['Timestamp'] = pd.to_datetime(health_data['Timestamp'])
        safety_data['Timestamp'] = pd.to_datetime(safety_data['Timestamp'])
        reminder_data['Timestamp'] = pd.to_datetime(reminder_data['Timestamp'])
        reminder_data['Scheduled Time'] = pd.to_datetime(reminder_data['Scheduled Time']).dt.time
        
        return health_data, safety_data, reminder_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

# Agent Configurations
config_list = [
    {
        "model": LLM_MODEL,
        "api_base": LLM_API_URL,
        "api_key": "apex@#1",  # Used in the Authorization header
        "api_type": "custom"   # Indicate a custom endpoint
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,

}

# Define Agent Classes
class HealthMonitorAgent(AssistantAgent):
    """Agent responsible for health monitoring and alerts"""
    
    def __init__(self, name, llm_config, health_data):
        super().__init__(name=name, llm_config=llm_config)
        self.health_data = health_data
        self.user_thresholds = self._initialize_user_thresholds()
    
    def _initialize_user_thresholds(self):
        """Initialize health thresholds for each user"""
        users = self.health_data["Device-ID/User-ID"].unique()
        thresholds = {}
        
        for user in users:
            thresholds[user] = {
                "heart_rate_min": 60,
                "heart_rate_max": 100,
                "blood_pressure_systolic_min": 90,
                "blood_pressure_systolic_max": 130,
                "blood_pressure_diastolic_min": 60,
                "blood_pressure_diastolic_max": 80,
                "glucose_min": 70,
                "glucose_max": 140,
                "spo2_min": 94
            }
        
        return thresholds
    
    def monitor_health(self, user_id, current_time=None):
        """Monitor health data for a specific user"""
        if current_time is None:
            current_time = datetime.now()
        
        user_data = self.health_data[self.health_data["Device-ID/User-ID"] == user_id]
        
        if user_data.empty:
            return {"status": f"No health data available for user {user_id}"}
        
        recent_data = user_data.sort_values("Timestamp", ascending=False).iloc[0]
        
        alerts = []
        thresholds = self.user_thresholds[user_id]
        
        heart_rate = recent_data["Heart Rate"]
        if heart_rate < thresholds["heart_rate_min"] or heart_rate > thresholds["heart_rate_max"]:
            alerts.append(f"Heart rate is {heart_rate}, outside normal range ({thresholds['heart_rate_min']}-{thresholds['heart_rate_max']})")
        
        if "Blood Pressure" in recent_data:
            bp_parts = recent_data["Blood Pressure"].split("/")
            if len(bp_parts) == 2:
                systolic = int(bp_parts[0].split()[0])
                diastolic = int(bp_parts[1].split()[0])
                
                if systolic < thresholds["blood_pressure_systolic_min"] or systolic > thresholds["blood_pressure_systolic_max"]:
                    alerts.append(f"Systolic blood pressure is {systolic}, outside normal range ({thresholds['blood_pressure_systolic_min']}-{thresholds['blood_pressure_systolic_max']})")
                
                if diastolic < thresholds["blood_pressure_diastolic_min"] or diastolic > thresholds["blood_pressure_diastolic_max"]:
                    alerts.append(f"Diastolic blood pressure is {diastolic}, outside normal range ({thresholds['blood_pressure_diastolic_min']}-{thresholds['blood_pressure_diastolic_max']})")
        
        glucose = recent_data["Glucose Levels"]
        if glucose < thresholds["glucose_min"] or glucose > thresholds["glucose_max"]:
            alerts.append(f"Glucose level is {glucose}, outside normal range ({thresholds['glucose_min']}-{thresholds['glucose_max']})")
        
        spo2 = recent_data["Oxygen Saturation (SpO₂%)"]
        if spo2 < thresholds["spo2_min"]:
            alerts.append(f"Oxygen saturation is {spo2}%, below threshold {thresholds['spo2_min']}%")
        
        # In HealthMonitorAgent.monitor_health
        if alerts:
            return {
                "user_id": user_id,
                "timestamp": recent_data["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "alerts": alerts,
                "recommendation": self._generate_health_recommendation(alerts),
                "heart_rate": recent_data["Heart Rate"],
                "blood_pressure": recent_data["Blood Pressure"],
                "glucose": recent_data["Glucose Levels"],
                "spo2": recent_data["Oxygen Saturation (SpO₂%)"]
            }
        else:
            return {
                "user_id": user_id,
                "timestamp": recent_data["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "status": "All health metrics within normal range",
                "heart_rate": recent_data["Heart Rate"],
                "blood_pressure": recent_data["Blood Pressure"],
                "glucose": recent_data["Glucose Levels"],
                "spo2": recent_data["Oxygen Saturation (SpO₂%)"]
            }
    
    def _generate_health_recommendation(self, alerts):
        """Generate recommendations based on health alerts"""
        alert_text = "\n".join(alerts)
        prompt = f"Based on the following health alerts, provide a brief recommendation for the caregiver:\n{alert_text}"
        return call_llm_api(prompt, system_message="You are a medical assistant. Provide brief, actionable recommendations.")

class SafetyMonitorAgent(AssistantAgent):
    """Agent responsible for safety monitoring and fall detection"""
    
    def __init__(self, name, llm_config, safety_data):
        super().__init__(name=name, llm_config=llm_config)
        self.safety_data = safety_data
        self.inactivity_thresholds = {user: 3600 for user in safety_data["Device-ID/User-ID"].unique()}  # 1 hour
    
    def monitor_safety(self, user_id, current_time=None):
        """Monitor safety data for a specific user"""
        if current_time is None:
            current_time = datetime.now()
        
        user_data = self.safety_data[self.safety_data["Device-ID/User-ID"] == user_id]
        
        if user_data.empty:
            return {"status": f"No safety data available for user {user_id}"}
        
        recent_data = user_data.sort_values("Timestamp", ascending=False).iloc[0]
        
        if recent_data["Fall Detected (Yes/No)"] == "Yes":
            return {
                "user_id": user_id,
                "timestamp": recent_data["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "alert": "Fall detected",
                "location": recent_data["Location"],
                "impact_force": recent_data["Impact Force Level"],
                "inactivity_duration": recent_data["Post-Fall Inactivity Duration (Seconds)"],
                "recommendation": "Immediate attention required. Contact emergency services if no response."
            }
        
        time_diff = (current_time - recent_data["Timestamp"]).total_seconds()
        if time_diff > self.inactivity_thresholds[user_id] and recent_data["Movement Activity"] == "No Movement":
            return {
                "user_id": user_id,
                "timestamp": recent_data["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "alert": "Extended inactivity detected",
                "location": recent_data["Location"],
                "inactivity_duration": f"{time_diff/60:.1f} minutes",
                "recommendation": "Check on the individual or try to make contact."
            }
        
        return {
            "user_id": user_id,
            "timestamp": recent_data["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "status": "No safety concerns detected",
            "last_activity": recent_data["Movement Activity"],
            "location": recent_data["Location"]
        }

class ReminderAgent(AssistantAgent):
    """Agent responsible for managing reminders"""
    
    def __init__(self, name, llm_config, reminder_data):
        super().__init__(name=name, llm_config=llm_config)
        self.reminder_data = reminder_data
    
    def get_upcoming_reminders(self, user_id, look_ahead_hours=2, current_time=None):
        """Get upcoming reminders for a specific user"""
        if current_time is None:
            current_time = datetime.now()
        
        user_reminders = self.reminder_data[self.reminder_data["Device-ID/User-ID"] == user_id]
        
        if user_reminders.empty:
            return {"status": f"No reminders available for user {user_id}"}
        
        current_time_only = current_time.time()
        upcoming_reminders = []
        
        for _, reminder in user_reminders.iterrows():
            scheduled_time = reminder["Scheduled Time"]
            
            current_seconds = current_time_only.hour * 3600 + current_time_only.minute * 60 + current_time_only.second
            scheduled_seconds = scheduled_time.hour * 3600 + scheduled_time.minute * 60 + scheduled_time.second
            
            if scheduled_seconds < current_seconds:
                time_diff = (24 * 3600 - current_seconds) + scheduled_seconds
            else:
                time_diff = scheduled_seconds - current_seconds
            
            if time_diff <= look_ahead_hours * 3600:
                upcoming_reminders.append({
                    "type": reminder["Reminder Type"],
                    "scheduled_time": scheduled_time.strftime("%H:%M"),
                    "time_until": f"{time_diff/60:.0f} minutes",
                    "already_sent": reminder["Reminder Sent (Yes/No)"] == "Yes",
                    "acknowledged": reminder["Acknowledged (Yes/No)"] == "Yes"
                })
        
        if not upcoming_reminders:
            return {
                "user_id": user_id,
                "status": f"No reminders scheduled in the next {look_ahead_hours} hours"
            }
        
        return {
            "user_id": user_id,
            "upcoming_reminders": upcoming_reminders,
            "recommendation": self._generate_reminder_message(upcoming_reminders)
        }
    
    def _generate_reminder_message(self, reminders):
        """Generate a personalized reminder message"""
        if not reminders:
            return "No upcoming reminders."
        
        next_reminder = min(reminders, key=lambda x: int(x["time_until"].split()[0]))
        prompt = f"Create a friendly, brief reminder message for an elderly person about their upcoming {next_reminder['type']} at {next_reminder['scheduled_time']} (in {next_reminder['time_until']})."
        return call_llm_api(prompt, system_message="You are a helpful assistant for elderly care. Keep messages clear, concise, and encouraging.")

class CoordinatorAgent(GroupChatManager):
    """Agent that coordinates between the specialized agents"""
    
    def __init__(self, name, llm_config, groupchat):
        super().__init__(name=name, groupchat=groupchat, llm_config=llm_config)

class CaregiverAgent(UserProxyAgent):
    """Agent representing the caregiver interface"""
    
    def __init__(self, name, llm_config):
        super().__init__(name=name, human_input_mode="NEVER", llm_config=llm_config)
        
    def notify_caregiver(self, alert_type, user_id, details):
        """Simulate notifying the caregiver"""
        message = f"ALERT: {alert_type} for user {user_id}\nDetails: {json.dumps(details, indent=2, cls=NumpyEncoder)}"
        logger.info(f"Caregiver notification sent: {message}")
        return {"status": "Notification sent", "message": message}

class ElderCareSystem:
    """Main system manager class"""
    
    def __init__(self):
        self.health_data, self.safety_data, self.reminder_data = load_data()
        
        if self.health_data is None or self.safety_data is None or self.reminder_data is None:
            logger.error("Failed to load data. System initialization aborted.")
            raise Exception("Data loading failed")
        
        self.health_agent = HealthMonitorAgent("Health_Monitor", llm_config, self.health_data)
        self.safety_agent = SafetyMonitorAgent("Safety_Monitor", llm_config, self.safety_data)
        self.reminder_agent = ReminderAgent("Reminder_System", llm_config, self.reminder_data)
        self.caregiver_agent = CaregiverAgent("Caregiver", llm_config)
        
        self.agents = [
            self.health_agent,
            self.safety_agent,
            self.reminder_agent,
            self.caregiver_agent
        ]
        
        self.groupchat = GroupChat(agents=self.agents, messages=[], max_round=10)
        self.coordinator = CoordinatorAgent("Coordinator", llm_config, self.groupchat)
    
    def run_comprehensive_check(self, user_id):
        """Run a comprehensive check on a user across all systems"""
        try:
            results = {
                "user_id": user_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "health_status": self.health_agent.monitor_health(user_id),
                "safety_status": self.safety_agent.monitor_safety(user_id),
                "reminders": self.reminder_agent.get_upcoming_reminders(user_id)
            }
            
            alerts = []
            if isinstance(results["health_status"], dict) and "alerts" in results["health_status"]:
                alerts.append({"type": "Health Alert", "details": results["health_status"]})
            
            if isinstance(results["safety_status"], dict) and "alert" in results["safety_status"]:
                alerts.append({"type": "Safety Alert", "details": results["safety_status"]})
            
            notifications = []
            for alert in alerts:
                notification = self.caregiver_agent.notify_caregiver(alert["type"], user_id, alert["details"])
                notifications.append(notification)
            
            if notifications:
                results["notifications"] = notifications
            
            # Convert any NumPy types to Python standard types before returning
            return json.loads(json.dumps(results, cls=NumpyEncoder))
        except Exception as e:
            logger.error(f"Error in comprehensive check: {e}")
            raise
    
    def run_group_analysis(self, user_id, query):
        """Run a collaborative analysis using the group chat"""
        # Gather data from each agent
        health_status = self.health_agent.monitor_health(user_id)
        safety_status = self.safety_agent.monitor_safety(user_id)
        reminders = self.reminder_agent.get_upcoming_reminders(user_id)

        # Construct a better structured prompt for the LLM
        initial_prompt = f"""
        You are an elderly care system AI assistant providing analysis for caregivers.
        
        Analyze the health and safety data for elderly user {user_id} with this query: {query}
        
        DATA:
        Health Status: {json.dumps(health_status, indent=2, cls=NumpyEncoder)}
        Safety Status: {json.dumps(safety_status, indent=2, cls=NumpyEncoder)}
        Upcoming Reminders: {json.dumps(reminders, indent=2, cls=NumpyEncoder)}
        
        RESPONSE FORMAT:
        Your response MUST be formatted in clean, well-structured Markdown with these sections:
        
        # Summary
        [Brief overview of user's current status - 2-3 sentences]
        
        ## Health Analysis
        - Heart Rate: [value] ([normal/high/low]) - Normal range: 60-100 bpm
        - Blood Pressure: [value] ([normal/high/low]) - Normal range: 90-130/60-80 mmHg
        - Glucose: [value] ([normal/high/low]) - Normal range: 70-140 mg/dL
        - SpO2: [value] ([normal/low]) - Normal range: >94%
        
        ## Safety Status
        - Current location: [location]
        - Recent activity: [activity details]
        - Movement status: [active/inactive] - [details]
        
        ## Upcoming Reminders
        - [List upcoming reminders with times]
        
        ## Recommendations
        1. [First specific recommendation]
        2. [Second specific recommendation]
        3. [Additional recommendations as needed]
        
        Make sure ALL sections are included in your response, even if some data is missing.
        Use bullet points and numbered lists for clarity.
        Focus on providing actionable insights and clear explanations.
        """

        # Directly call the LLM API instead of using autogen's group chat
        try:
            analysis = call_llm_api(
                prompt=initial_prompt,
                system_message="You are a professional healthcare coordinator specializing in elderly care. Always provide well-structured responses using proper Markdown formatting with headers, bullet points, and numbered lists. Focus on key insights and actionable recommendations.",
                max_tokens=800  # Increased to allow for a detailed response
            )
            return {
                "user_id": user_id,
                "query": query,
                "analysis": analysis,
                "full_conversation": [{"role": "coordinator", "content": analysis}]
            }
        except Exception as e:
            logger.error(f"Error in group analysis: {e}")
            return {
                "user_id": user_id,
                "query": query,
                "error": f"Analysis failed: {str(e)}"
            }

if __name__ == "__main__":
    system = ElderCareSystem()
    user_id = "D1001"
    results = system.run_comprehensive_check(user_id)
    print(json.dumps(results, indent=2, cls=NumpyEncoder))