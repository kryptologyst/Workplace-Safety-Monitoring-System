"""Unit tests for the workplace safety monitoring system."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from omegaconf import OmegaConf

from src.data.generator import SafetyDataGenerator
from src.models.anomaly_detection import SafetyAnomalyDetector, ThresholdBasedDetector
from src.eval.metrics import SafetyEvaluationMetrics


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return OmegaConf.create({
        'data': {
            'synthetic': {
                'n_samples': 100,
                'contamination': 0.1,
                'random_seed': 42
            },
            'sensors': {
                'temperature': {
                    'normal_range': [15, 35],
                    'critical_threshold': 60,
                    'unit': '°C'
                },
                'gas_level': {
                    'normal_range': [0, 10],
                    'critical_threshold': 25,
                    'unit': 'ppm'
                },
                'vibration': {
                    'normal_range': [0, 0.5],
                    'critical_threshold': 1.0,
                    'unit': 'G'
                }
            }
        },
        'models': {
            'isolation_forest': {
                'contamination': 0.1,
                'random_state': 42,
                'n_estimators': 10
            },
            'one_class_svm': {
                'nu': 0.1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'local_outlier_factor': {
                'n_neighbors': 5,
                'contamination': 0.1
            }
        },
        'evaluation': {
            'k_values': [5, 10]
        },
        'alerts': {
            'severity_levels': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.95
            }
        }
    })


@pytest.fixture
def sample_data(sample_config):
    """Generate sample data for testing."""
    generator = SafetyDataGenerator(sample_config)
    return generator.generate_sensor_data(n_samples=100, include_anomalies=True)


class TestSafetyDataGenerator:
    """Test cases for SafetyDataGenerator."""
    
    def test_initialization(self, sample_config):
        """Test generator initialization."""
        generator = SafetyDataGenerator(sample_config)
        assert generator.config == sample_config
        assert generator.random_seed == 42
    
    def test_generate_sensor_data(self, sample_config):
        """Test sensor data generation."""
        generator = SafetyDataGenerator(sample_config)
        data = generator.generate_sensor_data(n_samples=50)
        
        assert len(data) == 50
        assert 'timestamp' in data.columns
        assert 'temperature' in data.columns
        assert 'gas_level' in data.columns
        assert 'vibration' in data.columns
        assert 'is_anomaly' in data.columns
        
        # Check data types
        assert data['timestamp'].dtype == 'datetime64[ns]'
        assert data['is_anomaly'].dtype == 'int64'
    
    def test_generate_incident_data(self, sample_config, sample_data):
        """Test incident data generation."""
        generator = SafetyDataGenerator(sample_config)
        incidents = generator.generate_incident_data(sample_data)
        
        assert len(incidents) == sample_data['is_anomaly'].sum()
        assert 'incident_id' in incidents.columns
        assert 'severity' in incidents.columns
        assert 'description' in incidents.columns
    
    def test_anomaly_generation(self, sample_config):
        """Test anomaly generation."""
        generator = SafetyDataGenerator(sample_config)
        data = generator.generate_sensor_data(n_samples=100, include_anomalies=True)
        
        # Check anomaly rate is approximately correct
        anomaly_rate = data['is_anomaly'].mean()
        assert 0.05 <= anomaly_rate <= 0.15  # Allow some variance
    
    def test_derived_features(self, sample_config):
        """Test derived feature generation."""
        generator = SafetyDataGenerator(sample_config)
        data = generator.generate_sensor_data(n_samples=50)
        
        # Check for derived features
        derived_features = [col for col in data.columns if 'rolling' in col or 'zscore' in col]
        assert len(derived_features) > 0
        
        # Check time-based features
        assert 'hour' in data.columns
        assert 'day_of_week' in data.columns
        assert 'is_weekend' in data.columns


class TestSafetyAnomalyDetector:
    """Test cases for SafetyAnomalyDetector."""
    
    def test_initialization(self, sample_config):
        """Test detector initialization."""
        detector = SafetyAnomalyDetector(sample_config)
        assert detector.config == sample_config
        assert not detector.is_fitted
        assert len(detector.models) == 0
    
    def test_fit(self, sample_config, sample_data):
        """Test model fitting."""
        detector = SafetyAnomalyDetector(sample_config)
        X = sample_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        y = sample_data['is_anomaly']
        
        detector.fit(X, y)
        
        assert detector.is_fitted
        assert len(detector.models) > 0
        assert len(detector.feature_names) == 3
    
    def test_predict(self, sample_config, sample_data):
        """Test prediction."""
        detector = SafetyAnomalyDetector(sample_config)
        X = sample_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        y = sample_data['is_anomaly']
        
        detector.fit(X, y)
        predictions = detector.predict(X)
        
        assert len(predictions) > 0
        for model_name, pred in predictions.items():
            assert len(pred) == len(X)
            assert all(p in [0, 1] for p in pred)
    
    def test_predict_proba(self, sample_config, sample_data):
        """Test probability prediction."""
        detector = SafetyAnomalyDetector(sample_config)
        X = sample_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        y = sample_data['is_anomaly']
        
        detector.fit(X, y)
        probabilities = detector.predict_proba(X)
        
        assert len(probabilities) > 0
        for model_name, prob in probabilities.items():
            assert len(prob) == len(X)
            assert all(0 <= p <= 1 for p in prob)
    
    def test_ensemble_predict(self, sample_config, sample_data):
        """Test ensemble prediction."""
        detector = SafetyAnomalyDetector(sample_config)
        X = sample_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        y = sample_data['is_anomaly']
        
        detector.fit(X, y)
        pred, prob = detector.ensemble_predict(X)
        
        assert len(pred) == len(X)
        assert len(prob) == len(X)
        assert all(p in [0, 1] for p in pred)
        assert all(0 <= p <= 1 for p in prob)
    
    def test_evaluate(self, sample_config, sample_data):
        """Test model evaluation."""
        detector = SafetyAnomalyDetector(sample_config)
        X = sample_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        y = sample_data['is_anomaly']
        
        detector.fit(X, y)
        metrics = detector.evaluate(X, y)
        
        assert len(metrics) > 0
        assert 'ensemble_precision' in metrics
        assert 'ensemble_recall' in metrics
        assert 'ensemble_f1' in metrics


class TestThresholdBasedDetector:
    """Test cases for ThresholdBasedDetector."""
    
    def test_initialization(self, sample_config):
        """Test threshold detector initialization."""
        detector = ThresholdBasedDetector(sample_config)
        assert detector.config == sample_config
        assert len(detector.thresholds) == 3
    
    def test_predict(self, sample_config, sample_data):
        """Test threshold-based prediction."""
        detector = ThresholdBasedDetector(sample_config)
        X = sample_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        
        predictions = detector.predict(X)
        
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, sample_config, sample_data):
        """Test threshold-based probability prediction."""
        detector = ThresholdBasedDetector(sample_config)
        X = sample_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        
        probabilities = detector.predict_proba(X)
        
        assert len(probabilities) == len(X)
        assert all(0 <= p <= 1 for p in probabilities)


class TestSafetyEvaluationMetrics:
    """Test cases for SafetyEvaluationMetrics."""
    
    def test_initialization(self, sample_config):
        """Test evaluator initialization."""
        evaluator = SafetyEvaluationMetrics(sample_config)
        assert evaluator.config == sample_config
        assert evaluator.k_values == [5, 10]
    
    def test_calculate_all_metrics(self, sample_config):
        """Test comprehensive metric calculation."""
        evaluator = SafetyEvaluationMetrics(sample_config)
        
        # Create sample data
        y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8])
        
        metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        assert 'precision_at_5' in metrics
        assert 'precision_at_10' in metrics
        assert 'expected_calibration_error' in metrics
        assert 'brier_score' in metrics
        assert 'alert_rate' in metrics
        assert 'detection_rate' in metrics
        assert 'false_alarm_rate' in metrics
    
    def test_create_evaluation_report(self, sample_config):
        """Test evaluation report creation."""
        evaluator = SafetyEvaluationMetrics(sample_config)
        
        results = {
            'model1': {'f1_score': 0.8, 'auc_roc': 0.9, 'detection_rate': 0.85},
            'model2': {'f1_score': 0.7, 'auc_roc': 0.8, 'detection_rate': 0.75}
        }
        
        report = evaluator.create_evaluation_report(results)
        
        assert len(report) == 2
        assert 'rank' in report.columns
        assert 'composite_score' in report.columns


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self, sample_config):
        """Test complete end-to-end pipeline."""
        # Generate data
        generator = SafetyDataGenerator(sample_config)
        sensor_data = generator.generate_sensor_data(n_samples=100)
        
        # Prepare features
        X = sensor_data[['temperature', 'gas_level', 'vibration']].fillna(0)
        y = sensor_data['is_anomaly']
        
        # Train models
        detector = SafetyAnomalyDetector(sample_config)
        detector.fit(X, y)
        
        # Make predictions
        pred, prob = detector.ensemble_predict(X)
        
        # Evaluate
        evaluator = SafetyEvaluationMetrics(sample_config)
        metrics = evaluator.calculate_all_metrics(y, pred, prob)
        
        # Check results
        assert len(pred) == len(y)
        assert len(prob) == len(y)
        assert len(metrics) > 0
        assert 'f1_score' in metrics
    
    def test_data_consistency(self, sample_config):
        """Test data consistency across components."""
        generator = SafetyDataGenerator(sample_config)
        sensor_data = generator.generate_sensor_data(n_samples=50)
        
        # Check that all required columns exist
        required_columns = ['timestamp', 'temperature', 'gas_level', 'vibration', 'is_anomaly']
        for col in required_columns:
            assert col in sensor_data.columns
        
        # Check data ranges
        assert sensor_data['temperature'].min() >= 0
        assert sensor_data['gas_level'].min() >= 0
        assert sensor_data['vibration'].min() >= 0
        assert sensor_data['is_anomaly'].isin([0, 1]).all()


if __name__ == "__main__":
    pytest.main([__file__])
