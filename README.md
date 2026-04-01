# Workplace Safety Monitoring System

A comprehensive, production-ready workplace safety monitoring system that uses advanced anomaly detection techniques to identify unsafe conditions in real-time. This system combines rule-based thresholds with machine learning models to provide robust safety monitoring capabilities.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational demonstration only.**

This system is NOT intended for automated decision-making without human review. All safety decisions should be made by qualified safety professionals. This tool is for research, education, and demonstration purposes only.

## Project Overview

This project implements a sophisticated workplace safety monitoring system that:

- **Detects anomalies** in sensor data using ensemble machine learning models
- **Provides real-time alerts** for unsafe conditions
- **Offers comprehensive analytics** for safety trend analysis
- **Includes interactive dashboards** for monitoring and analysis
- **Ensures compliance** with safety standards and regulations

### Key Features

- **Multi-sensor monitoring**: Temperature, gas levels, vibration, noise, humidity
- **Advanced anomaly detection**: Isolation Forest, One-Class SVM, LOF, COPOD, ECOD
- **Ensemble methods**: Voting, averaging, and weighted ensemble approaches
- **Real-time alerting**: Configurable thresholds and severity levels
- **Interactive dashboard**: Streamlit-based web interface
- **Comprehensive evaluation**: Business and ML metrics
- **Explainable AI**: Feature importance and SHAP analysis
- **Compliance features**: Audit logging, bias detection, data privacy

## Architecture

```
src/
├── data/           # Data generation and processing
├── models/         # Anomaly detection models
├── eval/           # Evaluation metrics and reporting
├── viz/            # Visualization utilities
└── utils/          # Common utilities and logging

configs/            # Configuration files
scripts/            # Training and evaluation scripts
demo/               # Streamlit demo application
tests/              # Unit tests
assets/             # Model artifacts and reports
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Workplace-Safety-Monitoring-System.git
   cd Workplace-Safety-Monitoring-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training pipeline**:
   ```bash
   python scripts/train.py
   ```

4. **Launch the interactive demo**:
   ```bash
   streamlit run demo/app.py
   ```

### Quick Demo

```python
from omegaconf import OmegaConf
from src.data.generator import SafetyDataGenerator
from src.models.anomaly_detection import SafetyAnomalyDetector

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Generate sample data
generator = SafetyDataGenerator(config)
sensor_data = generator.generate_sensor_data(n_samples=1000)

# Train anomaly detector
detector = SafetyAnomalyDetector(config)
X = sensor_data[['temperature', 'gas_level', 'vibration']].fillna(0)
detector.fit(X, sensor_data['is_anomaly'])

# Make predictions
predictions, probabilities = detector.ensemble_predict(X)
print(f"Detected {predictions.sum()} anomalies")
```

## Dataset Schema

### Sensor Data (`sensor_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Reading timestamp |
| `sensor_id` | string | Unique sensor identifier |
| `location` | string | Monitoring location |
| `shift` | string | Work shift (Day/Night/Evening) |
| `temperature` | float | Temperature in °C |
| `gas_level` | float | Gas concentration in ppm |
| `vibration` | float | Vibration in G-force |
| `noise_level` | float | Noise level in dB |
| `humidity` | float | Humidity percentage |
| `is_anomaly` | int | True anomaly label (0/1) |

### Incident Data (`incident_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `incident_id` | string | Unique incident identifier |
| `timestamp` | datetime | Incident timestamp |
| `location` | string | Incident location |
| `sensor_id` | string | Associated sensor |
| `severity` | string | Severity level (Low/Medium/High/Critical) |
| `description` | string | Incident description |
| `resolved` | bool | Resolution status |
| `resolution_time` | datetime | Resolution timestamp |
| `cost_estimate` | float | Estimated cost |

## Models

### Anomaly Detection Models

1. **Isolation Forest**: Tree-based anomaly detection
2. **One-Class SVM**: Support vector machine for novelty detection
3. **Local Outlier Factor**: Density-based anomaly detection
4. **COPOD**: Copula-based outlier detection
5. **ECOD**: Empirical cumulative distribution-based detection

### Ensemble Methods

- **Voting**: Hard voting on binary predictions
- **Averaging**: Soft voting on probabilities
- **Weighted**: Performance-weighted ensemble

### Threshold-Based Detection

- **Rule-based**: Configurable thresholds per sensor type
- **Risk scoring**: Composite safety score calculation

## Evaluation Metrics

### Machine Learning Metrics

- **Classification**: Precision, Recall, F1-Score, AUC-ROC, AUC-PR
- **Calibration**: Expected Calibration Error, Brier Score
- **Precision@K**: Precision at different K values

### Business Metrics

- **Detection Rate**: Percentage of true anomalies caught
- **False Alarm Rate**: Percentage of false alerts
- **Alert Workload**: Percentage of readings flagged
- **Cost Metrics**: Cost-sensitive evaluation
- **Early Detection**: Capability to detect anomalies early

### Safety-Specific Metrics

- **Service Level**: Percentage of time system is operational
- **Response Time**: Time to detect and respond to incidents
- **Coverage**: Percentage of monitored areas
- **Compliance**: Adherence to safety standards

## Configuration

The system is configured via YAML files in the `configs/` directory:

### Key Configuration Sections

```yaml
# Data generation
data:
  synthetic:
    n_samples: 1000
    contamination: 0.1
    random_seed: 42

# Sensor thresholds
sensors:
  temperature:
    normal_range: [15, 35]
    critical_threshold: 60
    unit: "°C"

# Model parameters
models:
  isolation_forest:
    contamination: 0.1
    n_estimators: 100

# Alert configuration
alerts:
  severity_levels:
    low: 0.3
    medium: 0.6
    high: 0.8
    critical: 0.95
```

## Usage

### Training Models

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom_config.yaml
```

### Running the Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py

# Launch with custom port
streamlit run demo/app.py --server.port 8502
```

### Programmatic Usage

```python
from src.models.anomaly_detection import SafetyAnomalyDetector
from src.data.generator import SafetyDataGenerator

# Initialize components
generator = SafetyDataGenerator(config)
detector = SafetyAnomalyDetector(config)

# Generate and process data
sensor_data = generator.generate_sensor_data(n_samples=1000)
X = sensor_data[['temperature', 'gas_level', 'vibration']].fillna(0)

# Train and predict
detector.fit(X, sensor_data['is_anomaly'])
predictions, probabilities = detector.ensemble_predict(X)
```

## Demo Features

The Streamlit demo provides:

1. **Real-time Dashboard**: Live sensor readings and alerts
2. **Anomaly Detection**: Interactive analysis of detected anomalies
3. **Safety Analytics**: Location, time, and correlation analysis
4. **Model Performance**: Evaluation metrics and comparisons

### Dashboard Tabs

- **Dashboard**: Real-time monitoring and alerts
- **Anomaly Detection**: Detailed detection analysis
- **Analytics**: Safety trends and patterns
- **Model Performance**: Performance metrics and comparisons

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_anomaly_detection.py
```

## Development

### Code Quality

The project uses:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pytest**: Testing framework

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Adding New Models

1. Create model class in `src/models/`
2. Implement required methods (`fit`, `predict`, `predict_proba`)
3. Add configuration parameters
4. Update ensemble methods
5. Add tests

### Adding New Metrics

1. Add metric calculation in `src/eval/metrics.py`
2. Update evaluation pipeline
3. Add visualization if needed
4. Update documentation

## Compliance & Privacy

### Data Privacy

- **Anonymization**: Personal data is anonymized
- **PII Detection**: Automatic detection of personally identifiable information
- **Data Retention**: Configurable retention policies
- **Access Control**: Role-based access to sensitive data

### Audit & Logging

- **Audit Trails**: Complete decision logging
- **Model Versioning**: Track model changes
- **Performance Monitoring**: Continuous model evaluation
- **Compliance Reporting**: Automated compliance reports

### Bias & Fairness

- **Protected Attributes**: Monitoring for bias
- **Fairness Metrics**: Statistical parity, equalized odds
- **Bias Detection**: Automated bias testing
- **Mitigation Strategies**: Bias reduction techniques

## Deployment

### Production Considerations

1. **Model Serving**: Use FastAPI for API endpoints
2. **Monitoring**: Implement model drift detection
3. **Scaling**: Use containerization (Docker)
4. **Security**: Implement authentication and authorization
5. **Backup**: Regular model and data backups

### API Endpoints

```python
# Example FastAPI endpoints
@app.post("/predict")
async def predict_anomaly(data: SensorData):
    return detector.predict(data)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## References

- [Anomaly Detection Survey](https://arxiv.org/abs/1907.03816)
- [Workplace Safety Standards](https://www.osha.gov/)
- [Machine Learning Safety](https://arxiv.org/abs/1906.12320)
- [Explainable AI for Safety](https://arxiv.org/abs/2006.10674)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:

- **Issues**: Use GitHub Issues
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the docs/ directory

## Changelog

### Version 1.0.0
- Initial release
- Multi-model anomaly detection
- Interactive Streamlit demo
- Comprehensive evaluation metrics
- Production-ready architecture

---

**Remember**: This system is for research and educational purposes only. Always consult with qualified safety professionals for real-world safety decisions.
# Workplace-Safety-Monitoring-System
