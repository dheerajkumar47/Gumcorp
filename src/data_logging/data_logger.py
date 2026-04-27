import csv
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import cv2
import numpy as np

class TrackingLogger:
    def __init__(self, config_path: str = "../../config/settings.yaml"):
        """
        Initialize the tracking data logger.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.logging_enabled = self.config['logging']['enabled']
        self.log_level = self.config['logging']['log_level']
        self.log_file = self.config['logging']['log_file']
        self.csv_log_file = self.config['logging']['csv_log_file']
        
        # Setup file logging
        if self.logging_enabled:
            self._setup_file_logging()
            self._setup_csv_logging()
        
        # Data storage for current session
        self.session_data = []
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'logging': {
                    'enabled': True,
                    'log_level': 'INFO',
                    'log_file': '../../logs/factory_tracking.log',
                    'csv_log_file': '../../logs/tracking_data.csv'
                }
            }
    
    def _setup_file_logging(self):
        """Setup file-based logging."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger('FactoryTracking')
        self.logger.setLevel(getattr(logging, self.log_level))
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(getattr(logging, self.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_csv_logging(self):
        """Setup CSV-based logging for tracking data."""
        csv_dir = os.path.dirname(self.csv_log_file)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_log_file):
            with open(self.csv_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'employee_id',
                    'x_position',
                    'y_position',
                    'activity',
                    'camera_id',
                    'confidence'
                ])
    
    def log_detection(self, timestamp: float, employee_id: int, 
                    position: tuple, activity: str,
                    camera_id: str = 'camera_1', confidence: float = 1.0):
        """
        Log a detection event.
        
        Args:
            timestamp: Unix timestamp
            employee_id: Unique employee identifier
            position: (x, y) position on floor plan
            activity: Current activity classification
            camera_id: Camera that detected the employee
            confidence: Detection confidence score
        """
        if not self.logging_enabled:
            return
        
        # Log to CSV
        try:
            with open(self.csv_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.fromtimestamp(timestamp).isoformat(),
                    employee_id,
                    position[0],
                    position[1],
                    activity,
                    camera_id,
                    confidence
                ])
        except Exception as e:
            self.logger.error(f"Failed to log to CSV: {e}")
        
        # Store in session data
        self.session_data.append({
            'timestamp': timestamp,
            'employee_id': employee_id,
            'position': position,
            'activity': activity,
            'camera_id': camera_id,
            'confidence': confidence
        })
        
        self.logger.info(
            f"Detection: Employee {employee_id} at ({position[0]:.1f}, {position[1]:.1f}) "
            f"- {activity} (conf: {confidence:.2f})"
        )
    
    def log_event(self, event_type: str, message: str, level: str = 'INFO'):
        """
        Log a general event.
        
        Args:
            event_type: Type of event (e.g., 'SYSTEM', 'TRACKING', 'ERROR')
            message: Event message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        if not self.logging_enabled:
            return
        
        log_message = f"[{event_type}] {message}"
        getattr(self.logger, level.lower())(log_message)
    
    def log_presence(self, employee_id: int, event_type: str, 
                    timestamp: float, details: Dict = None):
        """
        Log presence events (entry, exit).
        
        Args:
            employee_id: Employee identifier
            event_type: Type of presence event ('ENTERED', 'EXITED')
            timestamp: Unix timestamp
            details: Additional details about the event
        """
        if not self.logging_enabled:
            return
        
        message = f"Employee {employee_id} {event_type} at {datetime.fromtimestamp(timestamp).isoformat()}"
        if details:
            message += f" - Details: {json.dumps(details)}"
        
        self.logger.info(message)
    
    def save_session_summary(self, output_path: str = None):
        """
        Save a summary of the current session.
        
        Args:
            output_path: Path to save the summary JSON file
        """
        if not self.logging_enabled or not self.session_data:
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"../../logs/session_summary_{timestamp}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            self.logger.info(f"Session summary saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save session summary: {e}")
    
    def get_session_data(self) -> List[Dict]:
        """Get all session data collected so far."""
        return self.session_data.copy()
    
    def export_to_json(self, output_path: str):
        """
        Export session data to JSON format.
        
        Args:
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            self.logger.info(f"Data exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")