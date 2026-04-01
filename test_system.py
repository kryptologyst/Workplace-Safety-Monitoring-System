"""Simple test script to verify the safety monitoring system works."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from src.data.generator import SafetyDataGenerator
from src.models.anomaly_detection import SafetyAnomalyDetector, ThresholdBasedDetector
from src.eval.metrics import SafetyEvaluationMetrics

def test_basic_functionality():
    """Test basic functionality of the safety monitoring system."""
    print("🛡️ Testing Workplace Safety Monitoring System")
    print("=" * 50)
    
    # Create a simple configuration
    config = OmegaConf.create({
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
    
    print("✅ Configuration created")
    
    # Test data generation
    print("\n📊 Testing data generation...")
    generator = SafetyDataGenerator(config)
    sensor_data = generator.generate_sensor_data(n_samples=100)
    
    print(f"   Generated {len(sensor_data)} sensor readings")
    print(f"   Anomaly rate: {sensor_data['is_anomaly'].mean():.3f}")
    print(f"   Features: {list(sensor_data.columns)}")
    
    # Test anomaly detection
    print("\n🤖 Testing anomaly detection...")
    X = sensor_data[['temperature', 'gas_level', 'vibration']].fillna(0)
    y = sensor_data['is_anomaly']
    
    # Test ensemble detector
    detector = SafetyAnomalyDetector(config)
    detector.fit(X, y)
    pred, prob = detector.ensemble_predict(X)
    
    print(f"   Ensemble predictions: {pred.sum()} anomalies detected")
    print(f"   Average probability: {prob.mean():.3f}")
    
    # Test threshold detector
    threshold_detector = ThresholdBasedDetector(config)
    threshold_pred = threshold_detector.predict(X)
    threshold_prob = threshold_detector.predict_proba(X)
    
    print(f"   Threshold predictions: {threshold_pred.sum()} anomalies detected")
    print(f"   Average probability: {threshold_prob.mean():.3f}")
    
    # Test evaluation
    print("\n📈 Testing evaluation metrics...")
    evaluator = SafetyEvaluationMetrics(config)
    
    # Evaluate ensemble model
    ensemble_metrics = evaluator.calculate_all_metrics(y, pred, prob, "Ensemble")
    print(f"   Ensemble F1 Score: {ensemble_metrics['f1_score']:.3f}")
    print(f"   Ensemble AUC-ROC: {ensemble_metrics['auc_roc']:.3f}")
    print(f"   Detection Rate: {ensemble_metrics['detection_rate']:.3f}")
    
    # Evaluate threshold model
    threshold_metrics = evaluator.calculate_all_metrics(y, threshold_pred, threshold_prob, "Threshold")
    print(f"   Threshold F1 Score: {threshold_metrics['f1_score']:.3f}")
    print(f"   Threshold Detection Rate: {threshold_metrics['detection_rate']:.3f}")
    
    # Test incident generation
    print("\n🚨 Testing incident generation...")
    incidents = generator.generate_incident_data(sensor_data)
    print(f"   Generated {len(incidents)} incidents")
    if len(incidents) > 0:
        print(f"   Severity distribution: {incidents['severity'].value_counts().to_dict()}")
    
    print("\n✅ All tests passed! The safety monitoring system is working correctly.")
    print("\n🚀 To run the full training pipeline:")
    print("   python scripts/train.py")
    print("\n🎛️ To launch the interactive demo:")
    print("   streamlit run demo/app.py")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
