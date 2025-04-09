from flask import Flask, request, jsonify
import json
import logging
from elderly_care_system import ElderCareSystem
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime

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
        return super().default(obj)

app = Flask(__name__)
CORS(app)
app.json_encoder = NumpyEncoder  # Use our custom encoder

# Initialize the system
system = ElderCareSystem()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/user/<user_id>/comprehensive-check', methods=['GET'])
def comprehensive_check(user_id):
    try:
        results = system.run_comprehensive_check(user_id)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in comprehensive check: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/health-status', methods=['GET'])
def health_status(user_id):
    try:
        status = system.health_agent.monitor_health(user_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in health status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/safety-status', methods=['GET'])
def safety_status(user_id):
    try:
        status = system.safety_agent.monitor_safety(user_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in safety status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/reminders', methods=['GET', 'POST'])
def reminders(user_id):
    if request.method == 'GET':
        try:
            look_ahead = request.args.get('look_ahead_hours', default=2, type=int)
            reminders = system.reminder_agent.get_upcoming_reminders(user_id, look_ahead_hours=look_ahead)
            return jsonify(reminders)
        except Exception as e:
            logger.error(f"Error in reminders: {e}")
            return jsonify({"error": str(e)}), 500
    
    if request.method == 'POST':
        try:
            data = request.json
            reminder_type = data.get('reminder_type')
            scheduled_time = data.get('scheduled_time')
            description = data.get('description', '')
            
            # Add new reminder to the reminder_data DataFrame
            new_reminder = pd.DataFrame({
                'Device-ID/User-ID': [user_id],
                'Timestamp': [datetime.now()],
                'Reminder Type': [reminder_type],
                'Scheduled Time': [scheduled_time],
                'Reminder Sent (Yes/No)': ['No'],
                'Acknowledged (Yes/No)': ['No']
            })
            
            system.reminder_data = pd.concat([system.reminder_data, new_reminder], ignore_index=True)
            system.reminder_data.to_csv('/Users/hitty/hitty_code/hackathon/daily_reminder.csv', index=False)
            
            return jsonify({"status": "Reminder added successfully"})
        except Exception as e:
            logger.error(f"Error adding reminder: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/analysis', methods=['POST'])
def collaborative_analysis(user_id):
    try:
        data = request.json
        query = data.get('query', '')
        analysis = system.run_group_analysis(user_id, query)
        # Convert any NumPy types before returning the response
        return jsonify(json.loads(json.dumps(analysis, cls=NumpyEncoder)))
    except Exception as e:
        logger.error(f"Error in collaborative analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/notify-caregiver', methods=['POST'])
def notify_caregiver():
    try:
        data = request.json
        user_id = data.get('user_id')
        alert_type = data.get('alert_type')
        details = data.get('details', {})
        
        notification = system.caregiver_agent.notify_caregiver(alert_type, user_id, details)
        return jsonify(notification)
    except Exception as e:
        logger.error(f"Error in caregiver notification: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)