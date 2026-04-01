"""Interactive Streamlit demo for workplace safety monitoring."""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from omegaconf import DictConfig, OmegaConf

from src.data.generator import SafetyDataGenerator
from src.models.anomaly_detection import SafetyAnomalyDetector, ThresholdBasedDetector
from src.eval.metrics import SafetyEvaluationMetrics
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Workplace Safety Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This system is NOT intended for automated decision-making without human review. 
    All safety decisions should be made by qualified safety professionals. 
    This tool is for research, education, and demonstration purposes only.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_config() -> DictConfig:
    """Load configuration with caching."""
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        return OmegaConf.load(config_path)
    else:
        # Return default config if file doesn't exist
        return OmegaConf.create({
            'data': {
                'synthetic': {'n_samples': 1000, 'contamination': 0.1, 'random_seed': 42},
                'sensors': {
                    'temperature': {'normal_range': [15, 35], 'critical_threshold': 60, 'unit': '°C'},
                    'gas_level': {'normal_range': [0, 10], 'critical_threshold': 25, 'unit': 'ppm'},
                    'vibration': {'normal_range': [0, 0.5], 'critical_threshold': 1.0, 'unit': 'G'},
                    'noise_level': {'normal_range': [40, 80], 'critical_threshold': 100, 'unit': 'dB'},
                    'humidity': {'normal_range': [30, 70], 'critical_threshold': 90, 'unit': '%'}
                }
            },
            'models': {
                'isolation_forest': {'contamination': 0.1, 'random_state': 42, 'n_estimators': 100},
                'one_class_svm': {'nu': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
                'local_outlier_factor': {'n_neighbors': 20, 'contamination': 0.1}
            },
            'evaluation': {'k_values': [5, 10, 20]},
            'alerts': {'severity_levels': {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 0.95}}
        })

@st.cache_data
def generate_sample_data(config: DictConfig, n_samples: int = 200) -> pd.DataFrame:
    """Generate sample sensor data."""
    generator = SafetyDataGenerator(config)
    return generator.generate_sensor_data(n_samples=n_samples, include_anomalies=True)

@st.cache_resource
def load_trained_model():
    """Load pre-trained model if available."""
    model_path = "assets/models/anomaly_detector.pkl"
    if os.path.exists(model_path):
        try:
            detector = SafetyAnomalyDetector(load_config())
            detector.load_model(model_path)
            return detector
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Workplace Safety Monitoring System</h1>', unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("📊 Data Parameters")
    n_samples = st.sidebar.slider("Number of samples", 50, 500, 200)
    contamination = st.sidebar.slider("Anomaly rate", 0.05, 0.3, 0.1)
    
    # Model parameters
    st.sidebar.subheader("🤖 Model Parameters")
    ensemble_method = st.sidebar.selectbox(
        "Ensemble method",
        ["voting", "averaging", "weighted"],
        index=1
    )
    
    # Alert thresholds
    st.sidebar.subheader("🚨 Alert Thresholds")
    alert_threshold = st.sidebar.slider("Alert probability threshold", 0.1, 0.9, 0.5)
    
    # Generate data
    if st.sidebar.button("🔄 Generate New Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Generate sample data
    sensor_data = generate_sample_data(config, n_samples)
    
    # Load or create model
    model = load_trained_model()
    if model is None:
        st.warning("⚠️ No pre-trained model found. Using threshold-based detection.")
        model = ThresholdBasedDetector(config)
        model_type = "threshold"
    else:
        model_type = "ensemble"
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Anomaly Detection", "📈 Analytics", "⚙️ Model Performance"])
    
    with tab1:
        st.header("📊 Real-time Safety Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_readings = len(sensor_data)
            st.metric("Total Readings", total_readings)
        
        with col2:
            anomalies = sensor_data['is_anomaly'].sum()
            st.metric("Anomalies Detected", anomalies)
        
        with col3:
            anomaly_rate = sensor_data['is_anomaly'].mean() * 100
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        
        with col4:
            locations = sensor_data['location'].nunique()
            st.metric("Monitoring Locations", locations)
        
        # Current sensor readings
        st.subheader("📡 Current Sensor Readings")
        
        # Get latest readings
        latest_data = sensor_data.tail(10)
        
        # Create sensor reading charts
        sensor_cols = ['temperature', 'gas_level', 'vibration', 'noise_level', 'humidity']
        available_sensors = [col for col in sensor_cols if col in sensor_data.columns]
        
        if available_sensors:
            fig = make_subplots(
                rows=len(available_sensors), cols=1,
                subplot_titles=[col.replace('_', ' ').title() for col in available_sensors],
                vertical_spacing=0.1
            )
            
            for i, sensor in enumerate(available_sensors, 1):
                fig.add_trace(
                    go.Scatter(
                        x=latest_data.index,
                        y=latest_data[sensor],
                        mode='lines+markers',
                        name=sensor,
                        line=dict(color='blue'),
                        showlegend=False
                    ),
                    row=i, col=1
                )
                
                # Add threshold line
                threshold = config.data.sensors[sensor].critical_threshold
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {threshold}",
                    row=i, col=1
                )
            
            fig.update_layout(
                height=200 * len(available_sensors),
                title="Sensor Readings Over Time",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Alert summary
        st.subheader("🚨 Safety Alerts")
        
        # Detect anomalies
        if model_type == "ensemble":
            X = sensor_data[available_sensors].fillna(0)
            pred, prob = model.ensemble_predict(X, method=ensemble_method)
            sensor_data['alert_probability'] = prob
            sensor_data['predicted_anomaly'] = pred
        else:
            X = sensor_data[available_sensors].fillna(0)
            pred = model.predict(X)
            prob = model.predict_proba(X)
            sensor_data['alert_probability'] = prob
            sensor_data['predicted_anomaly'] = pred
        
        # Filter alerts above threshold
        alerts = sensor_data[sensor_data['alert_probability'] > alert_threshold].copy()
        
        if len(alerts) > 0:
            st.warning(f"🚨 {len(alerts)} safety alerts detected!")
            
            # Display alerts
            for idx, alert in alerts.iterrows():
                prob = alert['alert_probability']
                severity = "High" if prob > 0.8 else "Medium" if prob > 0.5 else "Low"
                
                alert_class = f"alert-{severity.lower()}"
                
                st.markdown(f"""
                <div class="metric-card {alert_class}">
                    <h4>Alert #{idx}</h4>
                    <p><strong>Location:</strong> {alert['location']}</p>
                    <p><strong>Sensor:</strong> {alert['sensor_id']}</p>
                    <p><strong>Severity:</strong> {severity}</p>
                    <p><strong>Probability:</strong> {prob:.3f}</p>
                    <p><strong>Time:</strong> {alert['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No safety alerts detected")
    
    with tab2:
        st.header("🔍 Anomaly Detection Analysis")
        
        # Detection results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Summary")
            
            # Confusion matrix
            y_true = sensor_data['is_anomaly']
            y_pred = sensor_data['predicted_anomaly']
            
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            st.metric("True Positives", tp)
            st.metric("False Positives", fp)
            st.metric("True Negatives", tn)
            st.metric("False Negatives", fn)
            
            # Performance metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1 Score", f"{f1:.3f}")
        
        with col2:
            st.subheader("Anomaly Distribution")
            
            # Anomaly probability distribution
            fig = px.histogram(
                sensor_data,
                x='alert_probability',
                nbins=20,
                title="Anomaly Probability Distribution",
                labels={'alert_probability': 'Anomaly Probability', 'count': 'Count'}
            )
            fig.add_vline(x=alert_threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"Threshold: {alert_threshold}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.subheader("📊 Detailed Analysis")
        
        # Anomaly probability over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sensor_data.index,
            y=sensor_data['alert_probability'],
            mode='lines',
            name='Anomaly Probability',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=sensor_data.index,
            y=sensor_data['is_anomaly'],
            mode='markers',
            name='True Anomalies',
            marker=dict(color='red', size=8),
            yaxis='y2'
        ))
        
        fig.add_hline(y=alert_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Alert Threshold: {alert_threshold}")
        
        fig.update_layout(
            title="Anomaly Probability vs True Anomalies",
            xaxis_title="Time Index",
            yaxis_title="Anomaly Probability",
            yaxis2=dict(title="True Anomaly", overlaying="y", side="right"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📈 Safety Analytics")
        
        # Location analysis
        st.subheader("📍 Location Analysis")
        
        location_stats = sensor_data.groupby('location').agg({
            'is_anomaly': ['count', 'sum', 'mean'],
            'alert_probability': 'mean'
        }).round(3)
        
        location_stats.columns = ['Total Readings', 'Anomalies', 'Anomaly Rate', 'Avg Alert Prob']
        st.dataframe(location_stats)
        
        # Sensor correlation
        st.subheader("🔗 Sensor Correlations")
        
        sensor_corr = sensor_data[available_sensors].corr()
        fig = px.imshow(
            sensor_corr,
            text_auto=True,
            aspect="auto",
            title="Sensor Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time-based analysis
        st.subheader("⏰ Time-based Analysis")
        
        sensor_data['hour'] = sensor_data['timestamp'].dt.hour
        hourly_stats = sensor_data.groupby('hour').agg({
            'is_anomaly': 'mean',
            'alert_probability': 'mean'
        }).round(3)
        
        fig = px.line(
            hourly_stats,
            title="Anomaly Rate by Hour of Day",
            labels={'hour': 'Hour of Day', 'is_anomaly': 'Anomaly Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("⚙️ Model Performance")
        
        if model_type == "ensemble":
            st.success("✅ Using trained ensemble model")
            
            # Model comparison
            st.subheader("📊 Model Comparison")
            
            # Load evaluation results if available
            eval_path = "assets/reports/evaluation_report.csv"
            if os.path.exists(eval_path):
                eval_results = pd.read_csv(eval_path, index_col=0)
                
                # Display top models
                st.dataframe(eval_results[['rank', 'f1_score', 'auc_roc', 'detection_rate', 'cost_normalized_accuracy']].head(10))
                
                # Performance comparison chart
                metrics_to_plot = ['f1_score', 'auc_roc', 'detection_rate', 'cost_normalized_accuracy']
                available_metrics = [m for m in metrics_to_plot if m in eval_results.columns]
                
                if available_metrics:
                    fig = px.bar(
                        eval_results[available_metrics].head(5),
                        title="Model Performance Comparison",
                        labels={'index': 'Model', 'value': 'Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No evaluation results found. Run the training script to generate performance metrics.")
        else:
            st.warning("⚠️ Using threshold-based detection (no trained model available)")
            
            # Threshold analysis
            st.subheader("🎯 Threshold Analysis")
            
            thresholds = np.arange(0.1, 1.0, 0.1)
            precision_scores = []
            recall_scores = []
            
            for thresh in thresholds:
                pred_thresh = (sensor_data['alert_probability'] > thresh).astype(int)
                precision = precision_score(sensor_data['is_anomaly'], pred_thresh, zero_division=0)
                recall = recall_score(sensor_data['is_anomaly'], pred_thresh, zero_division=0)
                precision_scores.append(precision)
                recall_scores.append(recall)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thresholds, y=precision_scores, name='Precision', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=thresholds, y=recall_scores, name='Recall', line=dict(color='red')))
            
            fig.update_layout(
                title="Precision vs Recall by Threshold",
                xaxis_title="Threshold",
                yaxis_title="Score"
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
