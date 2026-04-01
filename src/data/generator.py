"""Data generation and processing for workplace safety monitoring."""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SafetyDataGenerator:
    """Generate synthetic workplace safety sensor data."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize the data generator with configuration.
        
        Args:
            config: Configuration object containing data generation parameters.
        """
        self.config = config
        self.random_seed = config.data.synthetic.random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
    def generate_sensor_data(
        self, 
        n_samples: int,
        start_time: Optional[datetime] = None,
        include_anomalies: bool = True
    ) -> pd.DataFrame:
        """Generate synthetic sensor data with optional anomalies.
        
        Args:
            n_samples: Number of data points to generate.
            start_time: Starting timestamp for the data.
            include_anomalies: Whether to include anomalous readings.
            
        Returns:
            DataFrame with sensor readings and metadata.
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
            
        timestamps = [
            start_time + timedelta(minutes=i * 5) 
            for i in range(n_samples)
        ]
        
        data = {
            'timestamp': timestamps,
            'sensor_id': np.random.choice(['SENSOR_001', 'SENSOR_002', 'SENSOR_003'], n_samples),
            'location': np.random.choice(['Factory_A', 'Factory_B', 'Warehouse'], n_samples),
            'shift': np.random.choice(['Day', 'Night', 'Evening'], n_samples),
        }
        
        # Generate normal sensor readings
        sensor_configs = self.config.data.sensors
        
        for sensor_name, sensor_config in sensor_configs.items():
            normal_min, normal_max = sensor_config.normal_range
            
            # Generate normal readings with some variation
            normal_readings = np.random.normal(
                loc=(normal_min + normal_max) / 2,
                scale=(normal_max - normal_min) / 6,  # 3-sigma rule
                size=n_samples
            )
            
            # Clip to reasonable bounds
            normal_readings = np.clip(normal_readings, normal_min * 0.5, normal_max * 1.5)
            
            data[sensor_name.lower()] = normal_readings
            
        df = pd.DataFrame(data)
        
        # Add anomalies if requested
        if include_anomalies:
            df = self._add_anomalies(df, sensor_configs)
            
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Generated {len(df)} sensor readings with {df['is_anomaly'].sum()} anomalies")
        
        return df
    
    def _add_anomalies(self, df: pd.DataFrame, sensor_configs: Dict) -> pd.DataFrame:
        """Add anomalous readings to the dataset.
        
        Args:
            df: DataFrame with normal sensor readings.
            sensor_configs: Configuration for each sensor type.
            
        Returns:
            DataFrame with anomalies added.
        """
        contamination = self.config.data.synthetic.contamination
        n_anomalies = int(len(df) * contamination)
        
        # Randomly select rows to make anomalous
        anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
        
        df['is_anomaly'] = 0
        df.loc[anomaly_indices, 'is_anomaly'] = 1
        
        # Create different types of anomalies
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drift', 'noise'])
            
            if anomaly_type == 'spike':
                # Sudden extreme values
                sensor_name = np.random.choice(list(sensor_configs.keys()))
                sensor_config = sensor_configs[sensor_name]
                critical_threshold = sensor_config.critical_threshold
                
                # Generate spike in either direction
                if np.random.random() > 0.5:
                    spike_value = critical_threshold * np.random.uniform(1.2, 2.0)
                else:
                    spike_value = critical_threshold * np.random.uniform(0.1, 0.3)
                    
                df.loc[idx, sensor_name.lower()] = spike_value
                
            elif anomaly_type == 'drift':
                # Gradual drift over multiple readings
                drift_length = min(5, len(df) - idx)
                sensor_name = np.random.choice(list(sensor_configs.keys()))
                sensor_config = sensor_configs[sensor_name]
                
                base_value = df.loc[idx, sensor_name.lower()]
                drift_factor = np.random.uniform(0.5, 1.5)
                
                for i in range(drift_length):
                    if idx + i < len(df):
                        df.loc[idx + i, sensor_name.lower()] *= drift_factor
                        
            elif anomaly_type == 'noise':
                # High noise levels
                sensor_name = np.random.choice(list(sensor_configs.keys()))
                noise_factor = np.random.uniform(2.0, 5.0)
                
                df.loc[idx, sensor_name.lower()] *= noise_factor
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better anomaly detection.
        
        Args:
            df: DataFrame with sensor readings.
            
        Returns:
            DataFrame with additional derived features.
        """
        # Rolling statistics
        window_size = 10
        
        for col in ['temperature', 'gas_level', 'vibration', 'noise_level', 'humidity']:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
                df[f'{col}_zscore'] = (df[col] - df[f'{col}_rolling_mean']) / df[f'{col}_rolling_std']
                
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Safety score (composite metric)
        safety_features = ['temperature', 'gas_level', 'vibration', 'noise_level', 'humidity']
        available_features = [f for f in safety_features if f in df.columns]
        
        if available_features:
            # Normalize features to 0-1 scale
            normalized_features = []
            for feature in available_features:
                feature_min = df[feature].min()
                feature_max = df[feature].max()
                normalized = (df[feature] - feature_min) / (feature_max - feature_min)
                normalized_features.append(normalized)
            
            # Calculate safety score (lower is safer)
            df['safety_score'] = np.mean(normalized_features, axis=0)
        
        return df
    
    def generate_incident_data(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """Generate incident reports based on sensor anomalies.
        
        Args:
            sensor_data: DataFrame with sensor readings and anomalies.
            
        Returns:
            DataFrame with incident reports.
        """
        anomaly_data = sensor_data[sensor_data['is_anomaly'] == 1].copy()
        
        incidents = []
        for idx, row in anomaly_data.iterrows():
            incident = {
                'incident_id': f"INC_{idx:06d}",
                'timestamp': row['timestamp'],
                'location': row['location'],
                'sensor_id': row['sensor_id'],
                'severity': self._determine_severity(row),
                'description': self._generate_description(row),
                'resolved': np.random.choice([True, False], p=[0.7, 0.3]),
                'resolution_time': None,
                'cost_estimate': np.random.uniform(100, 5000),
            }
            
            if incident['resolved']:
                resolution_delay = timedelta(hours=np.random.uniform(1, 24))
                incident['resolution_time'] = row['timestamp'] + resolution_delay
                
            incidents.append(incident)
        
        return pd.DataFrame(incidents)
    
    def _determine_severity(self, row: pd.Series) -> str:
        """Determine incident severity based on sensor readings.
        
        Args:
            row: Single row of sensor data.
            
        Returns:
            Severity level string.
        """
        severity_levels = self.config.alerts.severity_levels
        
        # Calculate overall risk score
        risk_score = 0
        sensor_configs = self.config.data.sensors
        
        for sensor_name, sensor_config in sensor_configs.items():
            if sensor_name.lower() in row.index:
                value = row[sensor_name.lower()]
                threshold = sensor_config.critical_threshold
                
                if value > threshold:
                    risk_score += (value / threshold) - 1
        
        # Map risk score to severity
        if risk_score >= severity_levels.critical:
            return 'Critical'
        elif risk_score >= severity_levels.high:
            return 'High'
        elif risk_score >= severity_levels.medium:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_description(self, row: pd.Series) -> str:
        """Generate incident description based on sensor readings.
        
        Args:
            row: Single row of sensor data.
            
        Returns:
            Description string.
        """
        descriptions = []
        sensor_configs = self.config.data.sensors
        
        for sensor_name, sensor_config in sensor_configs.items():
            if sensor_name.lower() in row.index:
                value = row[sensor_name.lower()]
                threshold = sensor_config.critical_threshold
                unit = sensor_config.unit
                
                if value > threshold:
                    descriptions.append(
                        f"{sensor_name} reading {value:.1f}{unit} exceeds threshold {threshold}{unit}"
                    )
        
        if descriptions:
            return "; ".join(descriptions)
        else:
            return "Multiple sensor readings indicate unsafe conditions"
