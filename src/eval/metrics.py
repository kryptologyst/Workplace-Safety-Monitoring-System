"""Comprehensive evaluation metrics for workplace safety monitoring."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SafetyEvaluationMetrics:
    """Comprehensive evaluation metrics for safety monitoring systems."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize the evaluation metrics.
        
        Args:
            config: Configuration object containing evaluation parameters.
        """
        self.config = config
        self.k_values = config.evaluation.k_values
        
    def calculate_all_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True anomaly labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model for logging.
            
        Returns:
            Dictionary with all calculated metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_classification_metrics(y_true, y_pred))
        
        # Probability-based metrics
        metrics.update(self._calculate_probability_metrics(y_true, y_prob))
        
        # Precision at K metrics
        metrics.update(self._calculate_precision_at_k(y_true, y_prob))
        
        # Calibration metrics
        metrics.update(self._calculate_calibration_metrics(y_true, y_prob))
        
        # Business-specific metrics
        metrics.update(self._calculate_business_metrics(y_true, y_pred, y_prob))
        
        logger.info(f"Calculated {len(metrics)} metrics for {model_name}")
        
        return metrics
    
    def _calculate_classification_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dictionary with classification metrics.
        """
        metrics = {}
        
        try:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
            
            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Balanced accuracy
            metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            metrics.update({
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'specificity': 0.0, 'sensitivity': 0.0, 'balanced_accuracy': 0.0
            })
        
        return metrics
    
    def _calculate_probability_metrics(
        self, 
        y_true: pd.Series, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate probability-based metrics.
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary with probability metrics.
        """
        metrics = {}
        
        try:
            # AUC-ROC
            if len(np.unique(y_true)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            else:
                metrics['auc_roc'] = 0.0
            
            # AUC-PR (Average Precision)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            
            # Log loss
            epsilon = 1e-15
            y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
            log_loss = -np.mean(y_true * np.log(y_prob_clipped) + 
                               (1 - y_true) * np.log(1 - y_prob_clipped))
            metrics['log_loss'] = log_loss
            
        except Exception as e:
            logger.error(f"Error calculating probability metrics: {e}")
            metrics.update({'auc_roc': 0.0, 'auc_pr': 0.0, 'log_loss': float('inf')})
        
        return metrics
    
    def _calculate_precision_at_k(
        self, 
        y_true: pd.Series, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate precision at different K values.
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary with precision at K metrics.
        """
        metrics = {}
        
        try:
            # Sort by probability (descending)
            sorted_indices = np.argsort(y_prob)[::-1]
            sorted_y_true = y_true.iloc[sorted_indices]
            
            for k in self.k_values:
                if k <= len(sorted_y_true):
                    top_k_true = sorted_y_true[:k]
                    precision_k = top_k_true.sum() / k
                    metrics[f'precision_at_{k}'] = precision_k
                else:
                    metrics[f'precision_at_{k}'] = 0.0
                    
        except Exception as e:
            logger.error(f"Error calculating precision at K: {e}")
            for k in self.k_values:
                metrics[f'precision_at_{k}'] = 0.0
        
        return metrics
    
    def _calculate_calibration_metrics(
        self, 
        y_true: pd.Series, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary with calibration metrics.
        """
        metrics = {}
        
        try:
            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['expected_calibration_error'] = ece
            
            # Brier Score
            brier_score = np.mean((y_prob - y_true) ** 2)
            metrics['brier_score'] = brier_score
            
        except Exception as e:
            logger.error(f"Error calculating calibration metrics: {e}")
            metrics.update({'expected_calibration_error': float('inf'), 'brier_score': float('inf')})
        
        return metrics
    
    def _calculate_business_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate business-specific metrics for safety monitoring.
        
        Args:
            y_true: True anomaly labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities.
            
        Returns:
            Dictionary with business metrics.
        """
        metrics = {}
        
        try:
            # Alert workload (percentage of alerts generated)
            metrics['alert_rate'] = y_pred.mean()
            
            # Detection rate (percentage of true anomalies caught)
            if y_true.sum() > 0:
                metrics['detection_rate'] = ((y_true == 1) & (y_pred == 1)).sum() / y_true.sum()
            else:
                metrics['detection_rate'] = 0.0
            
            # False alarm rate
            if (y_true == 0).sum() > 0:
                metrics['false_alarm_rate'] = ((y_true == 0) & (y_pred == 1)).sum() / (y_true == 0).sum()
            else:
                metrics['false_alarm_rate'] = 0.0
            
            # Cost-sensitive metrics (assuming false negatives are more costly)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Cost ratio: assume missing an anomaly costs 10x more than false alarm
            cost_ratio = 10
            total_cost = fp + cost_ratio * fn
            metrics['total_cost'] = total_cost
            
            # Cost-normalized accuracy
            if total_cost > 0:
                metrics['cost_normalized_accuracy'] = (tp + tn) / (tp + tn + total_cost)
            else:
                metrics['cost_normalized_accuracy'] = (tp + tn) / len(y_true)
            
            # Early detection capability (how early we catch anomalies)
            anomaly_indices = y_true[y_true == 1].index
            if len(anomaly_indices) > 0:
                early_detection_scores = []
                for idx in anomaly_indices:
                    # Look at probability trend before the anomaly
                    window_size = min(5, idx)
                    if window_size > 0:
                        prev_probs = y_prob[max(0, idx - window_size):idx]
                        if len(prev_probs) > 0:
                            # Score based on how much probability increased
                            prob_increase = y_prob[idx] - prev_probs.mean()
                            early_detection_scores.append(prob_increase)
                
                if early_detection_scores:
                    metrics['early_detection_score'] = np.mean(early_detection_scores)
                else:
                    metrics['early_detection_score'] = 0.0
            else:
                metrics['early_detection_score'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating business metrics: {e}")
            metrics.update({
                'alert_rate': 0.0, 'detection_rate': 0.0, 'false_alarm_rate': 0.0,
                'total_cost': float('inf'), 'cost_normalized_accuracy': 0.0,
                'early_detection_score': 0.0
            })
        
        return metrics
    
    def create_evaluation_report(
        self, 
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Create a comprehensive evaluation report.
        
        Args:
            results: Dictionary with model results.
            save_path: Optional path to save the report.
            
        Returns:
            DataFrame with evaluation results.
        """
        # Convert results to DataFrame
        df_results = pd.DataFrame(results).T
        
        # Add model ranking based on key metrics
        key_metrics = ['f1_score', 'auc_roc', 'detection_rate', 'cost_normalized_accuracy']
        available_metrics = [m for m in key_metrics if m in df_results.columns]
        
        if available_metrics:
            # Calculate composite score
            df_results['composite_score'] = df_results[available_metrics].mean(axis=1)
            df_results = df_results.sort_values('composite_score', ascending=False)
        
        # Add rankings
        df_results['rank'] = range(1, len(df_results) + 1)
        
        if save_path:
            df_results.to_csv(save_path)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return df_results
    
    def plot_evaluation_curves(
        self, 
        y_true: pd.Series, 
        y_prob: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC and Precision-Recall curves.
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_roc = roc_auc_score(y_true, y_prob)
            
            ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--', label='Random')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend()
            ax1.grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        
        ax2.plot(recall, precision, label=f'{model_name} (AUC = {auc_pr:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation curves saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(
        self, 
        y_true: pd.Series, 
        y_prob: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f'{model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
