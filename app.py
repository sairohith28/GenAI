from flask import Flask, request, jsonify
import json
import logging
from elderly_care_system import ElderCareSystem
from flask_cors import CORS  # Add CORS support

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the system
system = ElderCareSystem()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/api/user/<user_id>/comprehensive-check', methods=['GET'])
def comprehensive_check(user_id):
    """Run a comprehensive check on a user"""
    try:
        results = system.run_comprehensive_check(user_id)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in comprehensive check: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/health-status', methods=['GET'])
def health_status(user_id):
    """Get the health status for a user"""
    try:
        status = system.health_agent.monitor_health(user_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in health status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/safety-status', methods=['GET'])
def safety_status(user_id):
    """Get the safety status for a user"""
    try:
        status = system.safety_agent.monitor_safety(user_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in safety status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/reminders', methods=['GET'])
def reminders(user_id):
    """Get upcoming reminders for a user"""
    try:
        look_ahead = request.args.get('look_ahead_hours', default=2, type=int)
        reminders = system.reminder_agent.get_upcoming_reminders(user_id, look_ahead_hours=look_ahead)
        return jsonify(reminders)
    except Exception as e:
        logger.error(f"Error in reminders: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<user_id>/analysis', methods=['POST'])
def collaborative_analysis(user_id):
    """Run a collaborative analysis using the group chat"""
    try:
        data = request.json
        query = data.get('query', '')
        analysis = system.run_group_analysis(user_id, query)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error in collaborative analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/notify-caregiver', methods=['POST'])
def notify_caregiver():
    """Manually notify a caregiver"""
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
    app.run(host='0.0.0.0', port=5000)  # Bind to 0.0.0.0 to be accessible on your LAN