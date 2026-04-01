"""Main training script for workplace safety monitoring system."""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.data.generator import SafetyDataGenerator
from src.models.anomaly_detection import SafetyAnomalyDetector, ThresholdBasedDetector
from src.eval.metrics import SafetyEvaluationMetrics
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    return config


def prepare_data(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and prepare training data.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (sensor_data, incident_data).
    """
    logger.info("Generating synthetic safety data...")
    
    generator = SafetyDataGenerator(config)
    
    # Generate sensor data
    n_samples = config.data.synthetic.n_samples
    sensor_data = generator.generate_sensor_data(
        n_samples=n_samples,
        include_anomalies=True
    )
    
    # Generate incident data
    incident_data = generator.generate_incident_data(sensor_data)
    
    logger.info(f"Generated {len(sensor_data)} sensor readings and {len(incident_data)} incidents")
    
    return sensor_data, incident_data


def prepare_features(sensor_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and targets for training.
    
    Args:
        sensor_data: DataFrame with sensor readings.
        
    Returns:
        Tuple of (features, targets).
    """
    logger.info("Preparing features and targets...")
    
    # Select sensor features
    sensor_features = ['temperature', 'gas_level', 'vibration', 'noise_level', 'humidity']
    available_features = [f for f in sensor_features if f in sensor_data.columns]
    
    # Add derived features
    derived_features = [col for col in sensor_data.columns 
                       if any(suffix in col for suffix in ['_rolling_mean', '_rolling_std', '_zscore'])]
    
    # Combine all features
    feature_columns = available_features + derived_features + ['hour', 'day_of_week', 'is_weekend']
    feature_columns = [col for col in feature_columns if col in sensor_data.columns]
    
    X = sensor_data[feature_columns].fillna(0)
    y = sensor_data['is_anomaly']
    
    logger.info(f"Prepared {X.shape[1]} features for {len(X)} samples")
    logger.info(f"Anomaly rate: {y.mean():.3f}")
    
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series, config: DictConfig) -> Dict[str, Dict]:
    """Train all anomaly detection models.
    
    Args:
        X: Feature matrix.
        y: Target labels.
        config: Configuration object.
        
    Returns:
        Dictionary with trained models and results.
    """
    logger.info("Training anomaly detection models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Initialize models
    anomaly_detector = SafetyAnomalyDetector(config)
    threshold_detector = ThresholdBasedDetector(config)
    evaluator = SafetyEvaluationMetrics(config)
    
    # Train anomaly detector
    logger.info("Training ensemble anomaly detector...")
    anomaly_detector.fit(X_train, y_train)
    
    # Get predictions
    anomaly_pred, anomaly_prob = anomaly_detector.ensemble_predict(X_test)
    threshold_pred = threshold_detector.predict(X_test)
    threshold_prob = threshold_detector.predict_proba(X_test)
    
    # Evaluate models
    results = {}
    
    # Ensemble anomaly detector
    ensemble_metrics = evaluator.calculate_all_metrics(
        y_test, anomaly_pred, anomaly_prob, "Ensemble Anomaly Detector"
    )
    results['ensemble_anomaly_detector'] = ensemble_metrics
    
    # Threshold-based detector
    threshold_metrics = evaluator.calculate_all_metrics(
        y_test, threshold_pred, threshold_prob, "Threshold-Based Detector"
    )
    results['threshold_detector'] = threshold_metrics
    
    # Individual model evaluation
    individual_predictions = anomaly_detector.predict(X_test)
    individual_probabilities = anomaly_detector.predict_proba(X_test)
    
    for model_name in individual_predictions.keys():
        if model_name in individual_probabilities:
            model_metrics = evaluator.calculate_all_metrics(
                y_test, 
                individual_predictions[model_name], 
                individual_probabilities[model_name],
                model_name
            )
            results[model_name] = model_metrics
    
    logger.info(f"Trained and evaluated {len(results)} models")
    
    return {
        'models': {
            'anomaly_detector': anomaly_detector,
            'threshold_detector': threshold_detector
        },
        'evaluator': evaluator,
        'results': results,
        'test_data': (X_test, y_test)
    }


def save_results(results: Dict, config: DictConfig) -> None:
    """Save training results and models.
    
    Args:
        results: Dictionary with training results.
        config: Configuration object.
    """
    logger.info("Saving results and models...")
    
    # Create output directories
    os.makedirs("assets/models", exist_ok=True)
    os.makedirs("assets/reports", exist_ok=True)
    os.makedirs("assets/plots", exist_ok=True)
    
    # Save models
    anomaly_detector = results['models']['anomaly_detector']
    anomaly_detector.save_model("assets/models/anomaly_detector.pkl")
    
    # Create evaluation report
    evaluator = results['evaluator']
    evaluation_report = evaluator.create_evaluation_report(
        results['results'],
        save_path="assets/reports/evaluation_report.csv"
    )
    
    # Print top models
    print("\n" + "="*60)
    print("MODEL PERFORMANCE LEADERBOARD")
    print("="*60)
    print(evaluation_report[['rank', 'f1_score', 'auc_roc', 'detection_rate', 'cost_normalized_accuracy']].head(10))
    
    # Generate plots for best model
    X_test, y_test = results['test_data']
    best_model_name = evaluation_report.index[0]
    
    if best_model_name == 'ensemble_anomaly_detector':
        anomaly_detector = results['models']['anomaly_detector']
        pred, prob = anomaly_detector.ensemble_predict(X_test)
        
        evaluator.plot_evaluation_curves(
            y_test, prob, best_model_name,
            save_path="assets/plots/evaluation_curves.png"
        )
        
        evaluator.plot_calibration_curve(
            y_test, prob, best_model_name,
            save_path="assets/plots/calibration_curve.png"
        )
        
        evaluator.plot_confusion_matrix(
            y_test, pred, best_model_name,
            save_path="assets/plots/confusion_matrix.png"
        )
    
    logger.info("Results saved successfully")


def main():
    """Main training pipeline."""
    # Setup logging
    setup_logging()
    logger.info("Starting workplace safety monitoring training pipeline")
    
    try:
        # Load configuration
        config = load_config()
        
        # Set random seeds for reproducibility
        np.random.seed(config.data.synthetic.random_seed)
        
        # Prepare data
        sensor_data, incident_data = prepare_data(config)
        
        # Save raw data
        os.makedirs("data/raw", exist_ok=True)
        sensor_data.to_csv("data/raw/sensor_data.csv", index=False)
        incident_data.to_csv("data/raw/incident_data.csv", index=False)
        
        # Prepare features
        X, y = prepare_features(sensor_data)
        
        # Train models
        training_results = train_models(X, y, config)
        
        # Save results
        save_results(training_results, config)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
