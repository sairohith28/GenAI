import os
import pandas as pd
import requests
import json
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

#libraries to retrieve the data from env
from dotenv import load_dotenv 
load_dotenv()
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
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
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
        health_data = pd.read_csv("/Users/hitty/hitty_code/hackathon/health_monitoring.csv")
        safety_data = pd.read_csv("/Users/hitty/hitty_code/hackathon/safety_monitoring.csv")
        reminder_data = pd.read_csv("/Users/hitty/hitty_code/hackathon/daily_reminder.csv")
        
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
        "api_key": "apex@#1",
        "api_type": "custom"
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

# Define Agent Base Class for Common Functionality
class ElderCareAgent(AssistantAgent):
    """Base agent class with common functionality for all elder care agents"""
    
    def __init__(self, name, system_message, llm_config):
        super().__init__(name=name, llm_config=llm_config, system_message=system_message)
        self.recent_observations = []
        self.knowledge_base = {}
        self.agent_interactions = defaultdict(list)
        
    def record_observation(self, observation):
        """Record an observation or insight"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.recent_observations.append({"timestamp": timestamp, "content": observation})
        if len(self.recent_observations) > 10:  # Keep only recent observations
            self.recent_observations.pop(0)
    
    def record_interaction(self, agent_name, message):
        """Record interaction with another agent"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.agent_interactions[agent_name].append({
            "timestamp": timestamp,
            "message": message
        })
    
    def update_knowledge(self, key, value):
        """Update agent's knowledge base"""
        self.knowledge_base[key] = value
    
    def generate_insight(self, context):
        """Generate an insight based on recent observations"""
        if not self.recent_observations:
            return "No recent observations to analyze."
        
        observations = "\n".join([f"{obs['timestamp']}: {obs['content']}" for obs in self.recent_observations])
        prompt = f"Based on these recent observations, provide a concise insight:\n{observations}\n\nContext: {context}"
        
        return call_llm_api(prompt, 
                           system_message=f"You are the {self.name} agent providing insights based on observations.", 
                           max_tokens=200)

# Define Specialized Agent Classes
class HealthMonitorAgent(ElderCareAgent):
    """Agent responsible for health monitoring and alerts"""
    
    def __init__(self, name, llm_config, health_data):
        system_message = """You are a specialized Health Monitoring Agent for elderly care.
        Your responsibilities include:
        1. Monitoring vital signs (heart rate, blood pressure, glucose levels, oxygen saturation)
        2. Detecting abnormal health patterns and providing alerts
        3. Offering health insights and recommendations based on monitored data
        4. Coordinating with other agents when health issues need attention
        
        Be proactive in identifying health concerns and suggesting preventive measures.
        """
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.health_data = health_data
        self.user_thresholds = self._initialize_user_thresholds()
        self.user_health_trends = defaultdict(list)  # Track health trends by user
        
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
    
    def _record_health_trend(self, user_id, metrics):
        """Record health metrics for trend analysis"""
        if len(self.user_health_trends[user_id]) > 10:  # Keep only recent readings
            self.user_health_trends[user_id].pop(0)
        self.user_health_trends[user_id].append({
            "timestamp": datetime.now(),
            "metrics": metrics
        })
    
    def analyze_health_trends(self, user_id):
        """Analyze trends in health metrics over time"""
        if user_id not in self.user_health_trends or len(self.user_health_trends[user_id]) < 2:
            return "Insufficient data for trend analysis."
        
        trends = self.user_health_trends[user_id]
        metrics = ["heart_rate", "blood_pressure", "glucose", "spo2"]
        trend_analysis = {}
        
        for metric in metrics:
            if metric not in trends[0]["metrics"]:
                continue
                
            values = [t["metrics"].get(metric) for t in trends if metric in t["metrics"]]
            if not values or None in values:
                continue
                
            # For blood pressure, extract systolic value for trend
            if metric == "blood_pressure":
                systolic_values = []
                for bp in values:
                    if isinstance(bp, str) and "/" in bp:
                        try:
                            systolic = int(bp.split("/")[0].strip().split()[0])
                            systolic_values.append(systolic)
                        except:
                            continue
                values = systolic_values
            
            if not isinstance(values[0], (int, float)):
                continue
                
            # Calculate trend direction
            if len(values) >= 2:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                first_avg = sum(first_half)/len(first_half)
                second_avg = sum(second_half)/len(second_half)
                
                if second_avg > first_avg * 1.05:
                    trend_analysis[metric] = "increasing"
                elif second_avg < first_avg * .95:
                    trend_analysis[metric] = "decreasing"
                else:
                    trend_analysis[metric] = "stable"
        
        # Generate trend insight
        insight_prompt = f"Analyze these health trends for user {user_id}: {json.dumps(trend_analysis)}"
        trend_insight = call_llm_api(insight_prompt, 
                                    system_message="You are a health analyst providing a brief analysis of health trends. Keep it concise and factual.",
                                    max_tokens=150)
        
        self.record_observation(f"Health trend analysis for {user_id}: {trend_insight}")
        return trend_analysis
    
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
        
        # Record metrics for trend analysis
        self._record_health_trend(user_id, {
            "heart_rate": heart_rate,
            "blood_pressure": recent_data["Blood Pressure"],
            "glucose": glucose,
            "spo2": spo2
        })
        
        # Generate result
        if alerts:
            result = {
                "user_id": user_id,
                "timestamp": recent_data["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "alerts": alerts,
                "recommendation": self._generate_health_recommendation(alerts),
                "heart_rate": recent_data["Heart Rate"],
                "blood_pressure": recent_data["Blood Pressure"],
                "glucose": recent_data["Glucose Levels"],
                "spo2": recent_data["Oxygen Saturation (SpO₂%)"],
                "trend_analysis": self.analyze_health_trends(user_id)
            }
            self.record_observation(f"Health alert for {user_id}: {alerts}")
            return result
        else:
            result = {
                "user_id": user_id,
                "timestamp": recent_data["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "status": "All health metrics within normal range",
                "heart_rate": recent_data["Heart Rate"],
                "blood_pressure": recent_data["Blood Pressure"],
                "glucose": recent_data["Glucose Levels"],
                "spo2": recent_data["Oxygen Saturation (SpO₂%)"],
                "trend_analysis": self.analyze_health_trends(user_id)
            }
            self.record_observation(f"Healthy status for {user_id}: All metrics normal")
            return result
    
    def _generate_health_recommendation(self, alerts):
        """Generate recommendations based on health alerts"""
        alert_text = "\n".join(alerts)
        prompt = f"""Based on the following health alerts, provide a detailed recommendation for the caregiver:
        {alert_text}
        
        Provide:
        1. An assessment of the severity
        2. Specific actions the caregiver should take
        3. Any follow-up monitoring recommendations
        4. Lifestyle or dietary advice if relevant
        """
        
        return call_llm_api(prompt, 
                           system_message="You are a medical assistant providing detailed, actionable recommendations for caregivers of elderly patients.",
                           max_tokens=300)

class SafetyMonitorAgent(ElderCareAgent):
    """Agent responsible for safety monitoring and fall detection"""
    
    def __init__(self, name, llm_config, safety_data):
        system_message = """You are a specialized Safety Monitoring Agent for elderly care.
        Your responsibilities include:
        1. Monitoring movement patterns and detecting falls
        2. Identifying unusual periods of inactivity
        3. Tracking location within the home to ensure safety
        4. Alerting caregivers about potential safety issues
        
        Be vigilant in identifying safety concerns and suggesting preventive measures.
        """
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.safety_data = safety_data
        self.inactivity_thresholds = {user: 3600 for user in safety_data["Device-ID/User-ID"].unique()}  # 1 hour
        self.location_history = defaultdict(list)
        self.movement_patterns = defaultdict(list)
        
    def _update_location_history(self, user_id, location, timestamp):
        """Track user's location history"""
        if len(self.location_history[user_id]) > 20:  # Keep only recent locations
            self.location_history[user_id].pop(0)
            
        self.location_history[user_id].append({
            "timestamp": timestamp,
            "location": location
        })
    
    def _update_movement_patterns(self, user_id, activity, timestamp):
        """Track user's movement patterns"""
        if len(self.movement_patterns[user_id]) > 20:  # Keep only recent movements
            self.movement_patterns[user_id].pop(0)
            
        self.movement_patterns[user_id].append({
            "timestamp": timestamp,
            "activity": activity
        })
    
    def analyze_movement_patterns(self, user_id):
        """Analyze patterns in user movement"""
        if user_id not in self.movement_patterns or len(self.movement_patterns[user_id]) < 3:
            return "Insufficient data for pattern analysis."
        
        patterns = self.movement_patterns[user_id]
        activities = [p["activity"] for p in patterns]
        
        # Count activities
        activity_counts = {}
        for activity in activities:
            if activity in activity_counts:
                activity_counts[activity] += 1
            else:
                activity_counts[activity] = 1
        
        # Generate movement insight
        dominant_activity = max(activity_counts, key=activity_counts.get)
        activity_ratio = activity_counts[dominant_activity] / len(activities)
        
        if activity_ratio > 0.7 and dominant_activity == "No Movement":
            insight = f"Concern: {dominant_activity} is the dominant activity ({activity_counts[dominant_activity]} of {len(activities)} records, {activity_ratio:.0%}). This suggests unusually low activity levels."
            self.record_observation(f"Movement concern for {user_id}: Extended periods of inactivity detected")
        else:
            insight = f"Normal: Activity distribution shows typical pattern with {dominant_activity} being most common ({activity_ratio:.0%})."
            
        return {
            "dominant_activity": dominant_activity,
            "activity_distribution": activity_counts,
            "insight": insight
        }
    
    def analyze_location_patterns(self, user_id):
        """Analyze patterns in user location"""
        if user_id not in self.location_history or len(self.location_history[user_id]) < 3:
            return "Insufficient data for location analysis."
        
        locations = [l["location"] for l in self.location_history[user_id]]
        
        # Count locations
        location_counts = {}
        for location in locations:
            if location in location_counts:
                location_counts[location] += 1
            else:
                location_counts[location] = 1
        
        # Generate location insight
        most_frequented = max(location_counts, key=location_counts.get)
        location_ratio = location_counts[most_frequented] / len(locations)
        
        if location_ratio > 0.8:
            insight = f"Note: {most_frequented} is highly frequented ({location_counts[most_frequented]} of {len(locations)} records, {location_ratio:.0%})."
        else:
            insight = f"Normal: Location distribution shows typical pattern with {most_frequented} being most common ({location_ratio:.0%})."
            
        return {
            "most_frequented": most_frequented,
            "location_distribution": location_counts,
            "insight": insight
        }
    
    def monitor_safety(self, user_id, current_time=None):
        """Monitor safety data for a specific user"""
        if current_time is None:
            current_time = datetime.now()
        
        user_data = self.safety_data[self.safety_data["Device-ID/User-ID"] == user_id]
        
        if user_data.empty:
            return {"status": f"No safety data available for user {user_id}"}
        
        recent_data = user_data.sort_values("Timestamp", ascending=False).iloc[0]
        data_timestamp = recent_data["Timestamp"]
        
        # Update tracking data
        self._update_location_history(user_id, recent_data["Location"], data_timestamp)
        self._update_movement_patterns(user_id, recent_data["Movement Activity"], data_timestamp)
        
        # Fall detection logic
        if recent_data["Fall Detected (Yes/No)"] == "Yes":
            result = {
                "user_id": user_id,
                "timestamp": data_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "alert": "Fall detected",
                "location": recent_data["Location"],
                "impact_force": recent_data["Impact Force Level"],
                "inactivity_duration": recent_data["Post-Fall Inactivity Duration (Seconds)"],
                "recommendation": self._generate_fall_recommendation(recent_data),
                "movement_analysis": self.analyze_movement_patterns(user_id),
                "location_analysis": self.analyze_location_patterns(user_id)
            }
            self.record_observation(f"Fall detected for {user_id} in {recent_data['Location']}")
            return result
        
        # Inactivity detection logic
        time_diff = (current_time - data_timestamp).total_seconds()
        if time_diff > self.inactivity_thresholds[user_id] and recent_data["Movement Activity"] == "No Movement":
            result = {
                "user_id": user_id,
                "timestamp": data_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "alert": "Extended inactivity detected",
                "location": recent_data["Location"],
                "inactivity_duration": f"{time_diff/60:.1f} minutes",
                "recommendation": self._generate_inactivity_recommendation(time_diff, recent_data["Location"]),
                "movement_analysis": self.analyze_movement_patterns(user_id),
                "location_analysis": self.analyze_location_patterns(user_id)
            }
            self.record_observation(f"Extended inactivity for {user_id}: {time_diff/60:.1f} minutes in {recent_data['Location']}")
            return result
        
        # Normal status
        result = {
            "user_id": user_id,
            "timestamp": data_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "No safety concerns detected",
            "last_activity": recent_data["Movement Activity"],
            "location": recent_data["Location"],
            "movement_analysis": self.analyze_movement_patterns(user_id),
            "location_analysis": self.analyze_location_patterns(user_id)
        }
        return result
    
    def _generate_fall_recommendation(self, fall_data):
        """Generate recommendations for a detected fall"""
        prompt = f"""
        An elderly person has fallen with the following details:
        - Location: {fall_data["Location"]}
        - Impact Force: {fall_data["Impact Force Level"]}
        - Inactivity Duration: {fall_data["Post-Fall Inactivity Duration (Seconds)"]} seconds
        
        Provide immediate response instructions for the caregiver, including:
        1. Assessment priorities
        2. First aid considerations
        3. When emergency services should be called
        4. Follow-up care recommendations
        """
        
        return call_llm_api(prompt, 
                           system_message="You are an emergency care specialist providing critical guidance for caregivers responding to an elderly person's fall.",
                           max_tokens=300)
    
    def _generate_inactivity_recommendation(self, inactivity_duration, location):
        """Generate recommendations for extended inactivity"""
        inactivity_minutes = inactivity_duration / 60
        
        prompt = f"""
        An elderly person has been inactive for {inactivity_minutes:.1f} minutes in their {location}.
        
        Provide guidance for the caregiver, including:
        1. How to check on the person appropriately
        2. What to look for when assessing the situation
        3. Potential concerns related to extended inactivity
        4. When to escalate to medical attention
        """
        
        return call_llm_api(prompt, 
                           system_message="You are a care specialist providing guidance for checking on an elderly person after extended inactivity.",
                           max_tokens=250)

class ReminderAgent(ElderCareAgent):
    """Agent responsible for managing reminders"""
    
    def __init__(self, name, llm_config, reminder_data):
        system_message = """You are a specialized Reminder Agent for elderly care.
        Your responsibilities include:
        1. Managing medication reminders and schedules
        2. Tracking appointments and important events
        3. Creating personalized reminders for daily activities
        4. Ensuring reminders are acknowledged and followed
        
        Be supportive and encouraging in your reminder messages, while ensuring important tasks are completed.
        """
        super().__init__(name=name, system_message=system_message, llm_config=llm_config)
        self.reminder_data = reminder_data
        self.reminder_history = defaultdict(list)
        self.missed_reminder_patterns = defaultdict(int)
    
    def _update_reminder_history(self, user_id, reminder_type, acknowledged, scheduled_time):
        """Track reminder acknowledgment history"""
        if len(self.reminder_history[user_id]) > 20:  # Keep only recent reminder responses
            self.reminder_history[user_id].pop(0)
            
        self.reminder_history[user_id].append({
            "timestamp": datetime.now(),
            "type": reminder_type,
            "scheduled_time": scheduled_time,
            "acknowledged": acknowledged
        })
        
        # Track missed reminders
        if not acknowledged:
            self.missed_reminder_patterns[user_id] += 1
        else:
            self.missed_reminder_patterns[user_id] = max(0, self.missed_reminder_patterns[user_id] - 1)
    
    def analyze_reminder_adherence(self, user_id):
        """Analyze patterns in reminder acknowledgment"""
        if user_id not in self.reminder_history or len(self.reminder_history[user_id]) < 3:
            return "Insufficient data for reminder adherence analysis."
        
        history = self.reminder_history[user_id]
        acknowledged = sum(1 for r in history if r["acknowledged"])
        adherence_rate = acknowledged / len(history)
        
        # Analyze reminder types
        reminder_types = {}
        for r in history:
            r_type = r["type"]
            if r_type not in reminder_types:
                reminder_types[r_type] = {"total": 0, "acknowledged": 0}
            reminder_types[r_type]["total"] += 1
            if r["acknowledged"]:
                reminder_types[r_type]["acknowledged"] += 1
        
        # Find problematic reminder types
        problematic_types = []
        for r_type, stats in reminder_types.items():
            if stats["total"] >= 2 and stats["acknowledged"] / stats["total"] < 0.5:
                problematic_types.append(r_type)
        
        if adherence_rate < 0.7:
            status = "Concern"
            insight = f"Low reminder adherence rate ({adherence_rate:.0%}). "
            if problematic_types:
                insight += f"Problem areas: {', '.join(problematic_types)}."
            self.record_observation(f"Reminder adherence concern for {user_id}: {adherence_rate:.0%} compliance rate")
        else:
            status = "Good"
            insight = f"Healthy reminder adherence rate ({adherence_rate:.0%})."
        
        return {
            "adherence_rate": adherence_rate,
            "status": status,
            "insight": insight,
            "problematic_types": problematic_types,
            "reminder_type_stats": reminder_types
        }
    
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
                acknowledged = reminder["Acknowledged (Yes/No)"] == "Yes"
                
                # Update reminder history for completed reminders
                if reminder["Reminder Sent (Yes/No)"] == "Yes":
                    self._update_reminder_history(
                        user_id, 
                        reminder["Reminder Type"], 
                        acknowledged, 
                        scheduled_time.strftime("%H:%M")
                    )
                
                upcoming_reminders.append({
                    "type": reminder["Reminder Type"],
                    "scheduled_time": scheduled_time.strftime("%H:%M"),
                    "time_until": f"{time_diff/60:.0f} minutes",
                    "already_sent": reminder["Reminder Sent (Yes/No)"] == "Yes",
                    "acknowledged": acknowledged
                })
        
        if not upcoming_reminders:
            return {
                "user_id": user_id,
                "status": f"No reminders scheduled in the next {look_ahead_hours} hours",
                "adherence_analysis": self.analyze_reminder_adherence(user_id)
            }
        
        adherence_analysis = self.analyze_reminder_adherence(user_id)
        
        # Generate personalized message based on adherence patterns
        if adherence_analysis != "Insufficient data for reminder adherence analysis." and \
           isinstance(adherence_analysis, dict) and \
           adherence_analysis["status"] == "Concern":
            personalized_note = self._generate_adherence_improvement_message(adherence_analysis)
        else:
            personalized_note = None
        
        result = {
            "user_id": user_id,
            "upcoming_reminders": upcoming_reminders,
            "recommendation": self._generate_reminder_message(upcoming_reminders),
            "adherence_analysis": adherence_analysis
        }
        
        if personalized_note:
            result["personalized_note"] = personalized_note
            
        return result
    
    def _generate_reminder_message(self, reminders):
        """Generate a personalized reminder message"""
        if not reminders:
            return "No upcoming reminders."
        
        next_reminder = min(reminders, key=lambda x: int(x["time_until"].split()[0]))
        prompt = f"""Create a friendly, clear reminder message for an elderly person about their upcoming {next_reminder['type']} at {next_reminder['scheduled_time']} (in {next_reminder['time_until']}).
        
        The reminder should be:
        1. Easy to understand
        2. Encouraging and positive
        3. Clear about the specific task
        4. Include a gentle note about importance if it's a medical reminder
        """
        
        return call_llm_api(prompt, 
                           system_message="You are a helpful assistant for elderly care. Create clear, friendly, and encouraging reminder messages.",
                           max_tokens=200)
    
    def _generate_adherence_improvement_message(self, adherence_analysis):
        """Generate a message to help improve reminder adherence"""
        problematic_types = adherence_analysis.get("problematic_types", [])
        problematic_str = ", ".join(problematic_types) if problematic_types else "general reminders"
        
        prompt = f"""
        Create a supportive message for a caregiver to help an elderly person improve their adherence to {problematic_str}.
        Their current adherence rate is {adherence_analysis["adherence_rate"]:.0%}.
        
        The message should include:
        1. Constructive suggestions to improve adherence
        2. Potential root causes to investigate
        3. Positive reinforcement techniques
        4. When to consider simplifying the reminder schedule
        """
        
        return call_llm_api(prompt, 
                           system_message="You are a behavioral specialist focusing on improving adherence to medical and daily routines for elderly individuals.",
                           max_tokens=250)

class ContextCoordinatorAgent(GroupChatManager):
    """Agent that coordinates context and communication between specialized agents"""
    
    def __init__(self, name, llm_config, groupchat):
        super().__init__(name=name, groupchat=groupchat, llm_config=llm_config)
        self.user_context = defaultdict(dict)
        self.recent_decisions = []
        self.priority_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        
    def update_user_context(self, user_id, context_data):
        """Update context information for a user"""
        self.user_context[user_id].update(context_data)
    
    def get_user_context(self, user_id):
        """Get current context for a user"""
        return self.user_context.get(user_id, {})
    
    def coordinate_response(self, user_id, health_data, safety_data, reminder_data):
        """Coordinate response across agents based on collected data"""
        priority_level = "low"
        recommended_actions = []
        
        # Assess health alerts
        if "alerts" in health_data:
            priority_level = "medium"
            for alert in health_data["alerts"]:
                if "blood pressure" in alert.lower() or "heart rate" in alert.lower():
                    priority_level = "high"
                recommended_actions.append(f"Address health alert: {alert}")
        
        # Assess safety alerts
        if "alert" in safety_data:
            if safety_data["alert"] == "Fall detected":
                priority_level = "critical"
                recommended_actions.append("Immediate response to fall detection")
            elif "inactivity" in safety_data["alert"].lower():
                priority_level = max(priority_level, "high", key=lambda x: self.priority_levels[x])
                recommended_actions.append(f"Check on extended inactivity in {safety_data['location']}")
        
        # Assess reminder adherence
        if "adherence_analysis" in reminder_data and isinstance(reminder_data["adherence_analysis"], dict):
            adherence = reminder_data["adherence_analysis"]
            if adherence.get("status") == "Concern" and adherence.get("adherence_rate", 1) < 0.6:
                priority_level = max(priority_level, "medium", key=lambda x: self.priority_levels[x])
                recommended_actions.append("Address poor reminder adherence")
        
        # Generate coordinated response
        context = {
            "user_id": user_id,
            "priority_level": priority_level,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommended_actions": recommended_actions
        }
        
        # Record decision
        self.recent_decisions.append(context)
        if len(self.recent_decisions) > 10:
            self.recent_decisions.pop(0)
        
        # Update user context
        self.update_user_context(user_id, context)
        
        return context
    
    def generate_comprehensive_guidance(self, user_id, health_data, safety_data, reminder_data, query=None):
        """Generate comprehensive guidance based on all available data"""
        context = self.coordinate_response(user_id, health_data, safety_data, reminder_data)
        
        # Create detailed prompt for LLM
        prompt = f"""
        Generate comprehensive guidance for user {user_id} with priority level: {context['priority_level'].upper()}
        
        HEALTH DATA:
        {json.dumps(health_data, indent=2, cls=NumpyEncoder)}
        
        SAFETY DATA:
        {json.dumps(safety_data, indent=2, cls=NumpyEncoder)}
        
        REMINDER DATA:
        {json.dumps(reminder_data, indent=2, cls=NumpyEncoder)}
        
        COORDINATOR ASSESSMENT:
        - Priority: {context['priority_level'].upper()}
        - Recommended Actions: {', '.join(context['recommended_actions']) if context['recommended_actions'] else 'None'}
        
        USER QUERY: {query if query else "No specific query provided"}
        
        Based on all this information, please provide:
        
        1. A concise overview of the user's current status
        2. Specific recommendations for caregivers
        3. Health and safety action items in order of priority
        4. Suggestions for follow-up monitoring
        """
        
        return call_llm_api(prompt, 
                           system_message=f"""You are the Coordination Center for an elderly care system. 
                           Your role is to integrate information from health monitoring, safety systems, and reminder systems
                           to provide holistic, prioritized guidance for caregivers. Format your response in clear sections with
                           markdown formatting for readability. Be concise but thorough.""",
                           max_tokens=800)

class CaregiverAgent(UserProxyAgent):
    """Agent representing the caregiver interface"""
    
    def __init__(self, name, llm_config):
        super().__init__(name=name, human_input_mode="NEVER", llm_config=llm_config)
        self.notification_history = defaultdict(list)
        self.alert_counts = defaultdict(lambda: defaultdict(int))
        
    def notify_caregiver(self, alert_type, user_id, details):
        """Simulate notifying the caregiver"""
        timestamp = datetime.now()
        
        # Format message based on alert type and details
        if alert_type == "Health Alert":
            formatted_message = self._format_health_alert(user_id, details)
        elif alert_type == "Safety Alert":
            formatted_message = self._format_safety_alert(user_id, details)
        elif alert_type == "Reminder Alert":
            formatted_message = self._format_reminder_alert(user_id, details)
        else:
            formatted_message = f"ALERT: {alert_type} for user {user_id}\nDetails: {json.dumps(details, indent=2, cls=NumpyEncoder)}"
        
        # Track notification
        self.notification_history[user_id].append({
            "timestamp": timestamp,
            "type": alert_type,
            "message": formatted_message
        })
        
        # Count alerts by type
        self.alert_counts[user_id][alert_type] += 1
        
        # Log notification
        logger.info(f"Caregiver notification sent: {formatted_message}")
        
        # Return confirmation
        return {
            "status": "Notification sent", 
            "message": formatted_message,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _format_health_alert(self, user_id, details):
        """Format health alert message"""
        alerts = details.get("alerts", [])
        alert_list = "\n- ".join(alerts) if alerts else "No specific alerts"
        
        message = f"""HEALTH ALERT: User {user_id}
        
Timestamp: {details.get('timestamp', 'Unknown')}

Alerts:
- {alert_list}

Vitals:
- Heart Rate: {details.get('heart_rate', 'N/A')} bpm
- Blood Pressure: {details.get('blood_pressure', 'N/A')}
- Glucose: {details.get('glucose', 'N/A')} mg/dL
- SpO2: {details.get('spo2', 'N/A')}%

Recommendation:
{details.get('recommendation', 'No recommendation available')}
        """
        return message
    
    def _format_safety_alert(self, user_id, details):
        """Format safety alert message"""
        if details.get('alert') == "Fall detected":
            message = f"""URGENT SAFETY ALERT: Fall Detected for User {user_id}
            
Timestamp: {details.get('timestamp', 'Unknown')}
Location: {details.get('location', 'Unknown')}
Impact Force: {details.get('impact_force', 'Unknown')}
Inactivity Duration: {details.get('inactivity_duration', 'Unknown')} seconds

IMMEDIATE ACTION REQUIRED:
{details.get('recommendation', 'Check on user immediately')}
            """
        elif "inactivity" in details.get('alert', '').lower():
            message = f"""SAFETY ALERT: Extended Inactivity for User {user_id}
            
Timestamp: {details.get('timestamp', 'Unknown')}
Location: {details.get('location', 'Unknown')}
Inactivity Duration: {details.get('inactivity_duration', 'Unknown')}

Recommendation:
{details.get('recommendation', 'Check on user to ensure well-being')}
            """
        else:
            message = f"""SAFETY ALERT: User {user_id}
            
Timestamp: {details.get('timestamp', 'Unknown')}
Alert: {details.get('alert', 'Unknown safety issue')}
Location: {details.get('location', 'Unknown')}

Details: {json.dumps(details, indent=2, cls=NumpyEncoder)}
            """
        return message
    
    def _format_reminder_alert(self, user_id, details):
        """Format reminder alert message"""
        upcoming = details.get('upcoming_reminders', [])
        reminders_list = "\n- ".join([f"{r['type']} at {r['scheduled_time']} (in {r['time_until']})" for r in upcoming]) if upcoming else "None"
        
        message = f"""REMINDER ALERT: User {user_id}
        
Upcoming Reminders:
- {reminders_list}

Adherence Analysis: 
{details.get('adherence_analysis', {}).get('insight', 'No analysis available')}

Recommendation:
{details.get('recommendation', 'No recommendation available')}
        """
        return message
    
    def get_notification_summary(self, user_id, hours=24):
        """Get a summary of notifications for a user within the specified time period"""
        if user_id not in self.notification_history:
            return {"status": f"No notifications for user {user_id}"}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_notifications = [n for n in self.notification_history[user_id] if n["timestamp"] >= cutoff_time]
        
        if not recent_notifications:
            return {"status": f"No notifications for user {user_id} in the past {hours} hours"}
        
        # Count by type
        type_counts = {}
        for notification in recent_notifications:
            alert_type = notification["type"]
            if alert_type in type_counts:
                type_counts[alert_type] += 1
            else:
                type_counts[alert_type] = 1
        
        # Generate summary
        return {
            "user_id": user_id,
            "period_hours": hours,
            "total_count": len(recent_notifications),
            "type_breakdown": type_counts,
            "most_recent": recent_notifications[-1],
            "oldest": recent_notifications[0]
        }

class ElderCareSystem:
    """Main system manager class"""
    
    def __init__(self):
        self.health_data, self.safety_data, self.reminder_data = load_data()
        
        if self.health_data is None or self.safety_data is None or self.reminder_data is None:
            logger.error("Failed to load data. System initialization aborted.")
            raise Exception("Data loading failed")
        
        # Initialize specialized agents
        self.health_agent = HealthMonitorAgent("Health_Monitor", llm_config, self.health_data)
        self.safety_agent = SafetyMonitorAgent("Safety_Monitor", llm_config, self.safety_data)
        self.reminder_agent = ReminderAgent("Reminder_System", llm_config, self.reminder_data)
        self.caregiver_agent = CaregiverAgent("Caregiver", llm_config)
        
        # Configure the agent group
        self.agents = [
            self.health_agent,
            self.safety_agent,
            self.reminder_agent,
            self.caregiver_agent
        ]
        
        # Set up group chat and coordinator
        self.groupchat = GroupChat(agents=self.agents, messages=[], max_round=10)
        self.coordinator = ContextCoordinatorAgent("Coordinator", llm_config, self.groupchat)
        
        # System metrics
        self.system_start_time = datetime.now()
        self.request_counts = defaultdict(int)
        self.performance_metrics = {
            "avg_response_time": 0.5,  # Initial default value in seconds
            "total_requests": 0,
            "alerts_generated": 0
        }
    
    def run_comprehensive_check(self, user_id):
        """Run a comprehensive check on a user across all systems"""
        try:
            start_time = time.time()
            self.request_counts[user_id] += 1
            self.performance_metrics["total_requests"] += 1
            
            # Collect data from each specialized agent
            health_status = self.health_agent.monitor_health(user_id)
            safety_status = self.safety_agent.monitor_safety(user_id)
            reminders = self.reminder_agent.get_upcoming_reminders(user_id)
            
            # Organize results
            results = {
                "user_id": user_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "health_status": health_status,
                "safety_status": safety_status,
                "reminders": reminders,
                "request_count": self.request_counts[user_id]
            }
            
            # Process alerts
            alerts = []
            if isinstance(results["health_status"], dict) and "alerts" in results["health_status"]:
                alerts.append({"type": "Health Alert", "details": results["health_status"]})
            
            if isinstance(results["safety_status"], dict) and "alert" in results["safety_status"]:
                alerts.append({"type": "Safety Alert", "details": results["safety_status"]})
            
            if len(alerts) > 0:
                self.performance_metrics["alerts_generated"] += len(alerts)
                
                # Get coordinator's assessment
                coordinator_assessment = self.coordinator.coordinate_response(
                    user_id, health_status, safety_status, reminders
                )
                results["coordinator_assessment"] = coordinator_assessment
                
                # Send notifications to caregiver
                notifications = []
                for alert in alerts:
                    notification = self.caregiver_agent.notify_caregiver(alert["type"], user_id, alert["details"])
                    notifications.append(notification)
                
                if notifications:
                    results["notifications"] = notifications
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics["avg_response_time"] = (
                (self.performance_metrics["avg_response_time"] * 
                (self.performance_metrics["total_requests"] - 1) + processing_time) / 
                self.performance_metrics["total_requests"]
            )
            
            # Add performance metrics to results
            results["system_metrics"] = {
                "processing_time": processing_time,
                "uptime": (datetime.now() - self.system_start_time).total_seconds() / 3600,  # hours
                "avg_response_time": self.performance_metrics["avg_response_time"]
            }
            
            # Convert any NumPy types to Python standard types before returning
            return json.loads(json.dumps(results, cls=NumpyEncoder))
        except Exception as e:
            logger.error(f"Error in comprehensive check: {e}")
            raise
    
    def run_group_analysis(self, user_id, query):
        """Run a collaborative analysis using the group chat"""
        try:
            # Gather data from each agent
            health_status = self.health_agent.monitor_health(user_id)
            safety_status = self.safety_agent.monitor_safety(user_id)
            reminders = self.reminder_agent.get_upcoming_reminders(user_id)
            
            # Use the coordinator to generate comprehensive guidance
            analysis = self.coordinator.generate_comprehensive_guidance(
                user_id, health_status, safety_status, reminders, query
            )
            
            return {
                "user_id": user_id,
                "query": query,
                "analysis": analysis,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_sources": {
                    "health_data": "Latest readings from monitoring devices",
                    "safety_data": "Movement and location sensors",
                    "reminder_data": "Scheduled activities and medication"
                }
            }
        except Exception as e:
            logger.error(f"Error in group analysis: {e}")
            return {
                "user_id": user_id,
                "query": query,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def get_system_health(self):
        """Get system health metrics"""
        uptime = (datetime.now() - self.system_start_time).total_seconds() / 3600  # hours
        
        return {
            "status": "operational",
            "uptime_hours": uptime,
            "total_requests": self.performance_metrics["total_requests"],
            "alerts_generated": self.performance_metrics["alerts_generated"],
            "avg_response_time": self.performance_metrics["avg_response_time"],
            "active_agents": [
                {"name": "Health Monitor", "status": "active"},
                {"name": "Safety Monitor", "status": "active"},
                {"name": "Reminder System", "status": "active"},
                {"name": "Context Coordinator", "status": "active"},
                {"name": "Caregiver Interface", "status": "active"}
            ]
        }

if __name__ == "__main__":
    system = ElderCareSystem()
    user_id = "D1001"
    results = system.run_comprehensive_check(user_id)
    print(json.dumps(results, indent=2, cls=NumpyEncoder))