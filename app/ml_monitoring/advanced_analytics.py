"""
Advanced Analytics Service
Bias detection, automated retraining triggers, and advanced ML insights
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc, text
import json
import statistics
from collections import defaultdict, Counter
import warnings
from dataclasses import dataclass
from enum import Enum

from app.ml_monitoring.models import (
    GroundTruthValidation, MLPerformanceMetrics, DataQualityMetrics,
    ValidationStatus, RecognitionOutcome
)
from app.employees.models import Employee, TrainingPhoto
from app.attendance.models import AttendanceLog
from app.ml_monitoring.ml_metrics_service import MLMetricsService
from app.ml_monitoring.data_quality_analyzer import DataQualityAnalyzer
from app.ml_monitoring.model_evaluator import ModelEvaluator
from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BiasType(Enum):
    DEMOGRAPHIC = "demographic"
    TEMPORAL = "temporal"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    ENVIRONMENTAL = "environmental"


@dataclass
class PerformanceAlert:
    """Data class for performance alerts"""
    alert_id: str
    severity: AlertSeverity
    alert_type: str
    title: str
    description: str
    detected_at: datetime
    metrics: Dict[str, Any]
    recommendations: List[str]
    auto_action_taken: Optional[str] = None


@dataclass
class BiasDetectionResult:
    """Data class for bias detection results"""
    bias_type: BiasType
    severity: AlertSeverity
    confidence: float
    affected_groups: List[str]
    metrics: Dict[str, Any]
    description: str
    mitigation_strategies: List[str]


@dataclass
class RetrainingTrigger:
    """Data class for retraining triggers"""
    trigger_id: str
    trigger_type: str
    triggered_at: datetime
    trigger_conditions: Dict[str, Any]
    severity: AlertSeverity
    estimated_impact: str
    recommended_actions: List[str]
    auto_retrain_eligible: bool


class AdvancedAnalytics:
    """Advanced analytics service for ML monitoring and optimization"""
    
    def __init__(self, db: Session):
        self.db = db
        self.ml_metrics_service = MLMetricsService(db)
        self.data_quality_analyzer = DataQualityAnalyzer(db)
        self.model_evaluator = ModelEvaluator(db)
        
        # Performance thresholds for alerts
        self.performance_thresholds = {
            'accuracy_critical': 0.7,
            'accuracy_warning': 0.8,
            'precision_critical': 0.7,
            'precision_warning': 0.8,
            'recall_critical': 0.7,
            'recall_warning': 0.8,
            'f1_critical': 0.7,
            'f1_warning': 0.8,
            'false_positive_rate_critical': 0.2,
            'false_positive_rate_warning': 0.1,
            'false_negative_rate_critical': 0.2,
            'false_negative_rate_warning': 0.1
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'avg_quality_critical': 0.5,
            'avg_quality_warning': 0.7,
            'high_quality_ratio_critical': 0.3,
            'high_quality_ratio_warning': 0.6
        }
        
        # Retraining triggers
        self.retraining_thresholds = {
            'performance_degradation_days': 7,
            'min_accuracy_drop': 0.05,
            'min_new_validations': 50,
            'max_days_since_last_training': 90
        }
    
    def perform_comprehensive_analysis(self, analysis_period_days: int = 30) -> Dict:
        """Perform comprehensive advanced analytics"""
        
        logger.info(f"Starting comprehensive advanced analytics for {analysis_period_days} days")
        
        analysis_results = {
            'analysis_metadata': {
                'performed_at': datetime.utcnow().isoformat(),
                'analysis_period_days': analysis_period_days,
                'analysis_version': '1.0'
            }
        }
        
        try:
            # Performance monitoring and alerting
            performance_analysis = self.analyze_performance_trends(analysis_period_days)
            analysis_results['performance_analysis'] = performance_analysis
            
            # Bias detection
            bias_analysis = self.detect_bias_patterns(analysis_period_days)
            analysis_results['bias_analysis'] = bias_analysis
            
            # Data drift detection
            drift_analysis = self.detect_data_drift(analysis_period_days)
            analysis_results['drift_analysis'] = drift_analysis
            
            # Retraining trigger analysis
            retraining_analysis = self.analyze_retraining_triggers()
            analysis_results['retraining_analysis'] = retraining_analysis
            
            # System health assessment
            health_assessment = self.assess_system_health()
            analysis_results['health_assessment'] = health_assessment
            
            # Generate consolidated recommendations
            recommendations = self.generate_consolidated_recommendations(analysis_results)
            analysis_results['recommendations'] = recommendations
            
            # Generate executive summary
            executive_summary = self.generate_executive_summary(analysis_results)
            analysis_results['executive_summary'] = executive_summary
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def analyze_performance_trends(self, days: int) -> Dict:
        """Analyze performance trends and generate alerts"""
        
        logger.info(f"Analyzing performance trends for {days} days")
        
        # Get recent performance metrics
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        daily_metrics = self.db.query(MLPerformanceMetrics).filter(
            and_(
                MLPerformanceMetrics.date >= cutoff_date,
                MLPerformanceMetrics.period_type == 'daily'
            )
        ).order_by(MLPerformanceMetrics.date).all()
        
        if len(daily_metrics) < 3:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least 3 days of data, found {len(daily_metrics)}'
            }
        
        # Calculate trends
        trends = self._calculate_performance_trends(daily_metrics)
        
        # Detect performance degradation
        alerts = self._detect_performance_alerts(daily_metrics, trends)
        
        # Analyze variance and stability
        stability_analysis = self._analyze_performance_stability(daily_metrics)
        
        # Predict future performance
        predictions = self._predict_performance_trajectory(daily_metrics)
        
        return {
            'status': 'success',
            'analysis_period': f'{days} days',
            'data_points': len(daily_metrics),
            'trends': trends,
            'alerts': [alert.__dict__ for alert in alerts],
            'stability_analysis': stability_analysis,
            'predictions': predictions,
            'current_performance': self._get_current_performance_snapshot(daily_metrics[-1] if daily_metrics else None)
        }
    
    def _calculate_performance_trends(self, metrics: List[MLPerformanceMetrics]) -> Dict:
        """Calculate performance trends over time"""
        
        if len(metrics) < 2:
            return {'error': 'Insufficient data for trend calculation'}
        
        # Extract time series data
        dates = [m.date for m in metrics]
        accuracy_values = [m.accuracy or 0 for m in metrics]
        precision_values = [m.precision or 0 for m in metrics]
        recall_values = [m.recall or 0 for m in metrics]
        f1_values = [m.f1_score or 0 for m in metrics]
        
        # Calculate trends (simple linear regression slope)
        trends = {}
        for metric_name, values in [
            ('accuracy', accuracy_values),
            ('precision', precision_values),
            ('recall', recall_values),
            ('f1_score', f1_values)
        ]:
            if any(v > 0 for v in values):
                slope = self._calculate_trend_slope(values)
                trends[metric_name] = {
                    'slope': slope,
                    'direction': 'improving' if slope > 0.01 else 'degrading' if slope < -0.01 else 'stable',
                    'current_value': values[-1],
                    'change_magnitude': abs(slope),
                    'significance': 'high' if abs(slope) > 0.05 else 'medium' if abs(slope) > 0.02 else 'low'
                }
        
        return trends
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate linear trend slope"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x_val ** 2 for x_val in x)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    def _detect_performance_alerts(self, metrics: List[MLPerformanceMetrics], trends: Dict) -> List[PerformanceAlert]:
        """Detect performance-related alerts"""
        
        alerts = []
        
        if not metrics:
            return alerts
        
        latest_metric = metrics[-1]
        
        # Check current performance against thresholds
        current_accuracy = latest_metric.accuracy or 0
        current_precision = latest_metric.precision or 0
        current_recall = latest_metric.recall or 0
        current_f1 = latest_metric.f1_score or 0
        
        # Accuracy alerts
        if current_accuracy < self.performance_thresholds['accuracy_critical']:
            alerts.append(PerformanceAlert(
                alert_id=f"accuracy_critical_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                severity=AlertSeverity.CRITICAL,
                alert_type='performance_degradation',
                title='Critical Accuracy Drop',
                description=f'Accuracy dropped to {current_accuracy:.3f} (threshold: {self.performance_thresholds["accuracy_critical"]})',
                detected_at=datetime.utcnow(),
                metrics={'current_accuracy': current_accuracy, 'threshold': self.performance_thresholds['accuracy_critical']},
                recommendations=[
                    'Immediate investigation required',
                    'Review recent changes to the system',
                    'Check data quality',
                    'Consider emergency model rollback'
                ]
            ))
        elif current_accuracy < self.performance_thresholds['accuracy_warning']:
            alerts.append(PerformanceAlert(
                alert_id=f"accuracy_warning_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                severity=AlertSeverity.HIGH,
                alert_type='performance_degradation',
                title='Accuracy Below Target',
                description=f'Accuracy is {current_accuracy:.3f} (target: {self.performance_thresholds["accuracy_warning"]})',
                detected_at=datetime.utcnow(),
                metrics={'current_accuracy': current_accuracy, 'target': self.performance_thresholds['accuracy_warning']},
                recommendations=[
                    'Monitor closely for further degradation',
                    'Review recent validation results',
                    'Consider threshold optimization'
                ]
            ))
        
        # Trend-based alerts
        for metric_name, trend_data in trends.items():
            if trend_data['direction'] == 'degrading' and trend_data['significance'] in ['high', 'medium']:
                alerts.append(PerformanceAlert(
                    alert_id=f"{metric_name}_trend_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                    severity=AlertSeverity.MEDIUM if trend_data['significance'] == 'medium' else AlertSeverity.HIGH,
                    alert_type='performance_trend',
                    title=f'{metric_name.title()} Degrading Trend',
                    description=f'{metric_name.title()} showing {trend_data["significance"]} degradation trend',
                    detected_at=datetime.utcnow(),
                    metrics=trend_data,
                    recommendations=[
                        f'Investigate cause of {metric_name} degradation',
                        'Review recent system changes',
                        'Consider retraining if trend continues'
                    ]
                ))
        
        # Variance alerts (performance instability)
        recent_accuracy = [m.accuracy for m in metrics[-7:] if m.accuracy is not None]
        if len(recent_accuracy) >= 3:
            accuracy_variance = statistics.variance(recent_accuracy)
            if accuracy_variance > 0.01:  # High variance threshold
                alerts.append(PerformanceAlert(
                    alert_id=f"accuracy_instability_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                    severity=AlertSeverity.MEDIUM,
                    alert_type='performance_instability',
                    title='Performance Instability Detected',
                    description=f'High accuracy variance ({accuracy_variance:.4f}) in recent days',
                    detected_at=datetime.utcnow(),
                    metrics={'variance': accuracy_variance, 'recent_values': recent_accuracy},
                    recommendations=[
                        'Investigate causes of performance fluctuation',
                        'Check for data quality issues',
                        'Review environmental factors'
                    ]
                ))
        
        return alerts
    
    def _analyze_performance_stability(self, metrics: List[MLPerformanceMetrics]) -> Dict:
        """Analyze performance stability and consistency"""
        
        if len(metrics) < 3:
            return {'error': 'Insufficient data for stability analysis'}
        
        # Calculate stability metrics
        accuracy_values = [m.accuracy for m in metrics if m.accuracy is not None]
        precision_values = [m.precision for m in metrics if m.precision is not None]
        recall_values = [m.recall for m in metrics if m.recall is not None]
        
        stability_metrics = {}
        
        for metric_name, values in [
            ('accuracy', accuracy_values),
            ('precision', precision_values),
            ('recall', recall_values)
        ]:
            if len(values) >= 3:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
                
                stability_metrics[metric_name] = {
                    'mean': mean_val,
                    'std_dev': std_val,
                    'coefficient_of_variation': cv,
                    'stability_score': max(0, 1 - cv),  # Higher is more stable
                    'stability_level': (
                        'high' if cv < 0.05 else
                        'medium' if cv < 0.1 else 'low'
                    )
                }
        
        # Overall stability score
        stability_scores = [metrics[metric]['stability_score'] for metric in stability_metrics]
        overall_stability = statistics.mean(stability_scores) if stability_scores else 0
        
        return {
            'individual_metrics': stability_metrics,
            'overall_stability_score': overall_stability,
            'overall_stability_level': (
                'high' if overall_stability > 0.9 else
                'medium' if overall_stability > 0.8 else 'low'
            ),
            'recommendations': self._generate_stability_recommendations(overall_stability, stability_metrics)
        }
    
    def _predict_performance_trajectory(self, metrics: List[MLPerformanceMetrics]) -> Dict:
        """Predict future performance trajectory"""
        
        if len(metrics) < 5:
            return {'error': 'Insufficient data for prediction'}
        
        # Simple linear extrapolation for next 7 days
        accuracy_values = [m.accuracy for m in metrics[-14:] if m.accuracy is not None]
        
        if len(accuracy_values) < 3:
            return {'error': 'Insufficient accuracy data for prediction'}
        
        # Calculate trend
        slope = self._calculate_trend_slope(accuracy_values)
        current_accuracy = accuracy_values[-1]
        
        # Predict next 7 days
        predictions = []
        for days_ahead in range(1, 8):
            predicted_accuracy = current_accuracy + (slope * days_ahead)
            predicted_accuracy = max(0, min(1, predicted_accuracy))  # Clamp between 0 and 1
            
            predictions.append({
                'days_ahead': days_ahead,
                'predicted_accuracy': predicted_accuracy,
                'confidence': max(0, 1 - abs(slope) * days_ahead)  # Confidence decreases with time and slope magnitude
            })
        
        # Determine prediction outlook
        week_ahead_accuracy = predictions[-1]['predicted_accuracy']
        outlook = (
            'declining' if week_ahead_accuracy < current_accuracy - 0.02 else
            'improving' if week_ahead_accuracy > current_accuracy + 0.02 else
            'stable'
        )
        
        return {
            'predictions': predictions,
            'outlook': outlook,
            'trend_slope': slope,
            'confidence_level': 'high' if abs(slope) < 0.01 else 'medium' if abs(slope) < 0.03 else 'low'
        }
    
    def detect_bias_patterns(self, days: int) -> Dict:
        """Detect various types of bias in the system"""
        
        logger.info(f"Detecting bias patterns for {days} days")
        
        bias_results = {
            'analysis_period': f'{days} days',
            'detected_biases': [],
            'bias_summary': {
                'total_biases_detected': 0,
                'high_severity_biases': 0,
                'bias_types_found': []
            }
        }
        
        try:
            # Temporal bias detection
            temporal_bias = self._detect_temporal_bias(days)
            if temporal_bias:
                bias_results['detected_biases'].extend(temporal_bias)
            
            # Performance bias across employees
            performance_bias = self._detect_performance_bias(days)
            if performance_bias:
                bias_results['detected_biases'].extend(performance_bias)
            
            # Data quality bias
            quality_bias = self._detect_data_quality_bias(days)
            if quality_bias:
                bias_results['detected_biases'].extend(quality_bias)
            
            # Environmental bias (lighting, camera conditions)
            environmental_bias = self._detect_environmental_bias(days)
            if environmental_bias:
                bias_results['detected_biases'].extend(environmental_bias)
            
            # Update summary
            bias_results['bias_summary']['total_biases_detected'] = len(bias_results['detected_biases'])
            bias_results['bias_summary']['high_severity_biases'] = len([
                b for b in bias_results['detected_biases'] 
                if b.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            ])
            bias_results['bias_summary']['bias_types_found'] = list(set([
                b.bias_type.value for b in bias_results['detected_biases']
            ]))
            
        except Exception as e:
            logger.error(f"Error in bias detection: {str(e)}")
            bias_results['error'] = str(e)
        
        return bias_results
    
    def _detect_temporal_bias(self, days: int) -> List[BiasDetectionResult]:
        """Detect temporal bias patterns"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get attendance logs grouped by hour of day
        hourly_performance = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        # Get recent validations with time information
        validations = self.db.query(GroundTruthValidation).filter(
            and_(
                GroundTruthValidation.created_at >= cutoff_date,
                GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED
            )
        ).all()
        
        for validation in validations:
            hour = validation.created_at.hour
            hourly_performance[hour]['total'] += 1
            if validation.recognition_outcome == RecognitionOutcome.TRUE_POSITIVE:
                hourly_performance[hour]['successful'] += 1
        
        # Calculate success rates by hour
        hourly_success_rates = {}
        for hour, data in hourly_performance.items():
            if data['total'] > 0:
                hourly_success_rates[hour] = data['successful'] / data['total']
        
        biases = []
        
        if len(hourly_success_rates) >= 6:  # Need enough hours for analysis
            success_rates = list(hourly_success_rates.values())
            mean_rate = statistics.mean(success_rates)
            std_rate = statistics.stdev(success_rates) if len(success_rates) > 1 else 0
            
            # Detect hours with significantly lower performance
            problematic_hours = [
                hour for hour, rate in hourly_success_rates.items()
                if rate < mean_rate - std_rate
            ]
            
            if len(problematic_hours) > 0 and std_rate > 0.1:  # Significant variance
                biases.append(BiasDetectionResult(
                    bias_type=BiasType.TEMPORAL,
                    severity=AlertSeverity.MEDIUM if std_rate < 0.2 else AlertSeverity.HIGH,
                    confidence=min(1.0, std_rate * 5),  # Confidence based on variance
                    affected_groups=[f"Hour {hour}" for hour in problematic_hours],
                    metrics={
                        'hourly_success_rates': hourly_success_rates,
                        'mean_success_rate': mean_rate,
                        'std_success_rate': std_rate,
                        'problematic_hours': problematic_hours
                    },
                    description=f"Performance varies significantly by time of day (std: {std_rate:.3f})",
                    mitigation_strategies=[
                        'Investigate lighting conditions during problematic hours',
                        'Adjust camera settings for different times',
                        'Provide user guidance for optimal timing',
                        'Consider time-specific thresholds'
                    ]
                ))
        
        return biases
    
    def _detect_performance_bias(self, days: int) -> List[BiasDetectionResult]:
        """Detect performance bias across different employees"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get performance by employee
        employee_performance = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        validations = self.db.query(GroundTruthValidation).filter(
            and_(
                GroundTruthValidation.created_at >= cutoff_date,
                GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED,
                GroundTruthValidation.predicted_employee_id.isnot(None)
            )
        ).all()
        
        for validation in validations:
            emp_id = validation.predicted_employee_id
            employee_performance[emp_id]['total'] += 1
            if validation.recognition_outcome == RecognitionOutcome.TRUE_POSITIVE:
                employee_performance[emp_id]['successful'] += 1
        
        # Calculate success rates
        employee_success_rates = {}
        for emp_id, data in employee_performance.items():
            if data['total'] >= 5:  # Minimum samples for reliable analysis
                employee_success_rates[emp_id] = data['successful'] / data['total']
        
        biases = []
        
        if len(employee_success_rates) >= 5:
            success_rates = list(employee_success_rates.values())
            mean_rate = statistics.mean(success_rates)
            std_rate = statistics.stdev(success_rates) if len(success_rates) > 1 else 0
            
            # Calculate Gini coefficient for performance inequality
            gini = self._calculate_gini_coefficient(success_rates)
            
            if gini > 0.2:  # Significant inequality
                low_performers = [
                    emp_id for emp_id, rate in employee_success_rates.items()
                    if rate < mean_rate - std_rate
                ]
                
                biases.append(BiasDetectionResult(
                    bias_type=BiasType.PERFORMANCE,
                    severity=AlertSeverity.MEDIUM if gini < 0.4 else AlertSeverity.HIGH,
                    confidence=min(1.0, gini * 2),
                    affected_groups=[f"Employee {emp_id}" for emp_id in low_performers[:5]],
                    metrics={
                        'gini_coefficient': gini,
                        'mean_success_rate': mean_rate,
                        'std_success_rate': std_rate,
                        'low_performers_count': len(low_performers),
                        'employee_success_rates': {k: v for k, v in list(employee_success_rates.items())[:10]}
                    },
                    description=f"Significant performance inequality across employees (Gini: {gini:.3f})",
                    mitigation_strategies=[
                        'Review training data quality for low-performing employees',
                        'Collect additional training samples for affected employees',
                        'Investigate systematic issues with specific employee photos',
                        'Consider employee-specific threshold adjustments'
                    ]
                ))
        
        return biases
    
    def _detect_data_quality_bias(self, days: int) -> List[BiasDetectionResult]:
        """Detect bias related to data quality"""
        
        # Get recent quality metrics
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        quality_metrics = self.db.query(DataQualityMetrics).filter(
            and_(
                DataQualityMetrics.date >= cutoff_date,
                DataQualityMetrics.period_type == 'daily'
            )
        ).all()
        
        biases = []
        
        if len(quality_metrics) >= 3:
            # Check for quality degradation trend
            quality_scores = [m.avg_image_quality_score for m in quality_metrics if m.avg_image_quality_score]
            
            if len(quality_scores) >= 3:
                trend_slope = self._calculate_trend_slope(quality_scores)
                current_quality = quality_scores[-1]
                
                if trend_slope < -0.02 and current_quality < 0.7:  # Degrading quality
                    biases.append(BiasDetectionResult(
                        bias_type=BiasType.DATA_QUALITY,
                        severity=AlertSeverity.HIGH if current_quality < 0.5 else AlertSeverity.MEDIUM,
                        confidence=min(1.0, abs(trend_slope) * 10),
                        affected_groups=['All users'],
                        metrics={
                            'trend_slope': trend_slope,
                            'current_quality': current_quality,
                            'quality_scores': quality_scores[-7:]  # Last 7 days
                        },
                        description=f"Data quality degrading trend detected (current: {current_quality:.3f})",
                        mitigation_strategies=[
                            'Investigate camera hardware issues',
                            'Review environmental conditions',
                            'Implement stricter quality controls',
                            'Provide user guidance for better image quality'
                        ]
                    ))
        
        return biases
    
    def _detect_environmental_bias(self, days: int) -> List[BiasDetectionResult]:
        """Detect environmental bias (lighting, camera conditions)"""
        
        # This would require more detailed environmental data
        # For now, we'll use quality metrics as a proxy
        
        biases = []
        
        # Get recent quality data
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        quality_metrics = self.db.query(DataQualityMetrics).filter(
            and_(
                DataQualityMetrics.date >= cutoff_date,
                DataQualityMetrics.period_type == 'daily'
            )
        ).all()
        
        if len(quality_metrics) >= 5:
            # Check for lighting-related issues
            dark_image_counts = [m.too_dark_images for m in quality_metrics if m.too_dark_images is not None]
            bright_image_counts = [m.too_bright_images for m in quality_metrics if m.too_bright_images is not None]
            
            if dark_image_counts and bright_image_counts:
                avg_dark = statistics.mean(dark_image_counts)
                avg_bright = statistics.mean(bright_image_counts)
                total_issues = avg_dark + avg_bright
                
                if total_issues > 10:  # Significant lighting issues
                    primary_issue = 'dark' if avg_dark > avg_bright else 'bright'
                    
                    biases.append(BiasDetectionResult(
                        bias_type=BiasType.ENVIRONMENTAL,
                        severity=AlertSeverity.MEDIUM if total_issues < 20 else AlertSeverity.HIGH,
                        confidence=min(1.0, total_issues / 50),
                        affected_groups=[f'{primary_issue}_lighting_conditions'],
                        metrics={
                            'avg_dark_images': avg_dark,
                            'avg_bright_images': avg_bright,
                            'primary_issue': primary_issue,
                            'total_lighting_issues': total_issues
                        },
                        description=f"Environmental lighting bias detected (avg {total_issues:.1f} lighting issues/day)",
                        mitigation_strategies=[
                            'Optimize lighting conditions in capture area',
                            'Install adjustable lighting systems',
                            'Provide user guidance for optimal positioning',
                            'Consider automatic camera exposure adjustment'
                        ]
                    ))
        
        return biases
    
    def detect_data_drift(self, days: int) -> Dict:
        """Detect data drift in the system"""
        
        logger.info(f"Detecting data drift for {days} days")
        
        # Compare recent data distribution with historical baseline
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        baseline_date = datetime.utcnow() - timedelta(days=days*2)  # Previous period as baseline
        
        # Get recent and baseline quality metrics
        recent_metrics = self.db.query(DataQualityMetrics).filter(
            and_(
                DataQualityMetrics.date >= cutoff_date,
                DataQualityMetrics.period_type == 'daily'
            )
        ).all()
        
        baseline_metrics = self.db.query(DataQualityMetrics).filter(
            and_(
                DataQualityMetrics.date >= baseline_date,
                DataQualityMetrics.date < cutoff_date,
                DataQualityMetrics.period_type == 'daily'
            )
        ).all()
        
        if len(recent_metrics) < 3 or len(baseline_metrics) < 3:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least 3 days of data in each period. Recent: {len(recent_metrics)}, Baseline: {len(baseline_metrics)}'
            }
        
        # Calculate drift metrics
        drift_analysis = {
            'quality_drift': self._calculate_quality_drift(recent_metrics, baseline_metrics),
            'distribution_drift': self._calculate_distribution_drift(recent_metrics, baseline_metrics),
            'temporal_drift': self._calculate_temporal_drift(recent_metrics, baseline_metrics)
        }
        
        # Determine overall drift severity
        drift_scores = [analysis.get('drift_score', 0) for analysis in drift_analysis.values()]
        overall_drift_score = max(drift_scores) if drift_scores else 0
        
        drift_level = (
            'high' if overall_drift_score > 0.7 else
            'medium' if overall_drift_score > 0.4 else
            'low'
        )
        
        return {
            'status': 'success',
            'analysis_period': f'{days} days',
            'baseline_period': f'{days} days (previous period)',
            'overall_drift_score': overall_drift_score,
            'drift_level': drift_level,
            'drift_analysis': drift_analysis,
            'recommendations': self._generate_drift_recommendations(drift_level, drift_analysis)
        }
    
    def _calculate_quality_drift(self, recent_metrics: List, baseline_metrics: List) -> Dict:
        """Calculate drift in data quality metrics"""
        
        # Extract quality scores
        recent_quality = [m.avg_image_quality_score for m in recent_metrics if m.avg_image_quality_score]
        baseline_quality = [m.avg_image_quality_score for m in baseline_metrics if m.avg_image_quality_score]
        
        if not recent_quality or not baseline_quality:
            return {'error': 'No quality data available'}
        
        # Calculate statistical differences
        recent_mean = statistics.mean(recent_quality)
        baseline_mean = statistics.mean(baseline_quality)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(recent_quality) + np.var(baseline_quality)) / 2)
        cohens_d = abs(recent_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        drift_score = min(1.0, cohens_d / 2)  # Normalize to 0-1
        
        return {
            'recent_mean_quality': recent_mean,
            'baseline_mean_quality': baseline_mean,
            'quality_change': recent_mean - baseline_mean,
            'effect_size': cohens_d,
            'drift_score': drift_score,
            'drift_direction': 'degradation' if recent_mean < baseline_mean else 'improvement'
        }
    
    def _calculate_distribution_drift(self, recent_metrics: List, baseline_metrics: List) -> Dict:
        """Calculate drift in data distribution"""
        
        # Compare distribution of quality levels
        recent_dist = {'high': 0, 'medium': 0, 'low': 0}
        baseline_dist = {'high': 0, 'medium': 0, 'low': 0}
        
        for metric in recent_metrics:
            total = metric.total_images_processed or 1
            recent_dist['high'] += (metric.high_quality_images or 0) / total
            recent_dist['medium'] += (metric.medium_quality_images or 0) / total
            recent_dist['low'] += (metric.low_quality_images or 0) / total
        
        for metric in baseline_metrics:
            total = metric.total_images_processed or 1
            baseline_dist['high'] += (metric.high_quality_images or 0) / total
            baseline_dist['medium'] += (metric.medium_quality_images or 0) / total
            baseline_dist['low'] += (metric.low_quality_images or 0) / total
        
        # Normalize
        recent_total = sum(recent_dist.values())
        baseline_total = sum(baseline_dist.values())
        
        if recent_total > 0:
            recent_dist = {k: v/recent_total for k, v in recent_dist.items()}
        if baseline_total > 0:
            baseline_dist = {k: v/baseline_total for k, v in baseline_dist.items()}
        
        # Calculate KL divergence (simplified)
        kl_divergence = 0
        for quality_level in ['high', 'medium', 'low']:
            r = recent_dist[quality_level]
            b = baseline_dist[quality_level]
            if r > 0 and b > 0:
                kl_divergence += r * np.log(r / b)
        
        drift_score = min(1.0, kl_divergence)
        
        return {
            'recent_distribution': recent_dist,
            'baseline_distribution': baseline_dist,
            'kl_divergence': kl_divergence,
            'drift_score': drift_score
        }
    
    def _calculate_temporal_drift(self, recent_metrics: List, baseline_metrics: List) -> Dict:
        """Calculate temporal drift patterns"""
        
        # Simple temporal drift based on variance changes
        recent_quality = [m.avg_image_quality_score for m in recent_metrics if m.avg_image_quality_score]
        baseline_quality = [m.avg_image_quality_score for m in baseline_metrics if m.avg_image_quality_score]
        
        if len(recent_quality) < 2 or len(baseline_quality) < 2:
            return {'error': 'Insufficient data for temporal drift calculation'}
        
        recent_var = statistics.variance(recent_quality)
        baseline_var = statistics.variance(baseline_quality)
        
        variance_ratio = recent_var / baseline_var if baseline_var > 0 else 1
        drift_score = min(1.0, abs(np.log(variance_ratio)) / 2) if variance_ratio > 0 else 0
        
        return {
            'recent_variance': recent_var,
            'baseline_variance': baseline_var,
            'variance_ratio': variance_ratio,
            'drift_score': drift_score,
            'stability_change': 'less_stable' if variance_ratio > 1.5 else 'more_stable' if variance_ratio < 0.67 else 'similar'
        }
    
    def analyze_retraining_triggers(self) -> Dict:
        """Analyze conditions that might trigger model retraining"""
        
        logger.info("Analyzing retraining triggers")
        
        triggers = []
        
        # Check performance degradation
        performance_trigger = self._check_performance_degradation_trigger()
        if performance_trigger:
            triggers.append(performance_trigger)
        
        # Check data accumulation
        data_trigger = self._check_new_data_trigger()
        if data_trigger:
            triggers.append(data_trigger)
        
        # Check time-based trigger
        time_trigger = self._check_time_based_trigger()
        if time_trigger:
            triggers.append(time_trigger)
        
        # Check bias accumulation
        bias_trigger = self._check_bias_accumulation_trigger()
        if bias_trigger:
            triggers.append(bias_trigger)
        
        # Determine overall retraining recommendation
        if triggers:
            highest_severity = max(trigger.severity for trigger in triggers)
            auto_retrain_eligible = any(trigger.auto_retrain_eligible for trigger in triggers)
            
            recommendation = (
                'immediate_retraining' if highest_severity == AlertSeverity.CRITICAL else
                'schedule_retraining' if highest_severity == AlertSeverity.HIGH else
                'consider_retraining' if highest_severity == AlertSeverity.MEDIUM else
                'monitor_closely'
            )
        else:
            recommendation = 'no_retraining_needed'
            auto_retrain_eligible = False
        
        return {
            'retraining_triggers': [trigger.__dict__ for trigger in triggers],
            'trigger_count': len(triggers),
            'highest_severity': triggers[0].severity.value if triggers else 'none',
            'recommendation': recommendation,
            'auto_retrain_eligible': auto_retrain_eligible,
            'next_evaluation_date': (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
    
    def _check_performance_degradation_trigger(self) -> Optional[RetrainingTrigger]:
        """Check if performance degradation warrants retraining"""
        
        # Get recent performance metrics
        days = self.retraining_thresholds['performance_degradation_days']
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_metrics = self.db.query(MLPerformanceMetrics).filter(
            and_(
                MLPerformanceMetrics.date >= cutoff_date,
                MLPerformanceMetrics.period_type == 'daily'
            )
        ).order_by(MLPerformanceMetrics.date).all()
        
        if len(recent_metrics) < 3:
            return None
        
        # Check for accuracy drop
        accuracy_values = [m.accuracy for m in recent_metrics if m.accuracy is not None]
        if len(accuracy_values) < 3:
            return None
        
        initial_accuracy = accuracy_values[0]
        current_accuracy = accuracy_values[-1]
        accuracy_drop = initial_accuracy - current_accuracy
        
        if accuracy_drop >= self.retraining_thresholds['min_accuracy_drop']:
            severity = (
                AlertSeverity.CRITICAL if accuracy_drop > 0.1 else
                AlertSeverity.HIGH if accuracy_drop > 0.07 else
                AlertSeverity.MEDIUM
            )
            
            return RetrainingTrigger(
                trigger_id=f"performance_degradation_{datetime.utcnow().strftime('%Y%m%d')}",
                trigger_type='performance_degradation',
                triggered_at=datetime.utcnow(),
                trigger_conditions={
                    'accuracy_drop': accuracy_drop,
                    'initial_accuracy': initial_accuracy,
                    'current_accuracy': current_accuracy,
                    'days_analyzed': days
                },
                severity=severity,
                estimated_impact='high' if accuracy_drop > 0.07 else 'medium',
                recommended_actions=[
                    'Collect recent validation data for retraining',
                    'Analyze causes of performance degradation',
                    'Prepare retraining dataset with recent examples',
                    'Schedule model retraining and evaluation'
                ],
                auto_retrain_eligible=severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            )
        
        return None
    
    def _check_new_data_trigger(self) -> Optional[RetrainingTrigger]:
        """Check if sufficient new validation data warrants retraining"""
        
        # Count recent validations
        days_since_last_retrain = 30  # Assume 30 days for now
        cutoff_date = datetime.utcnow() - timedelta(days=days_since_last_retrain)
        
        new_validations = self.db.query(GroundTruthValidation).filter(
            and_(
                GroundTruthValidation.created_at >= cutoff_date,
                GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED
            )
        ).count()
        
        if new_validations >= self.retraining_thresholds['min_new_validations']:
            return RetrainingTrigger(
                trigger_id=f"new_data_{datetime.utcnow().strftime('%Y%m%d')}",
                trigger_type='new_data_accumulation',
                triggered_at=datetime.utcnow(),
                trigger_conditions={
                    'new_validations': new_validations,
                    'threshold': self.retraining_thresholds['min_new_validations'],
                    'days_analyzed': days_since_last_retrain
                },
                severity=AlertSeverity.MEDIUM,
                estimated_impact='medium',
                recommended_actions=[
                    'Incorporate new validation data into training set',
                    'Perform incremental model update',
                    'Validate improved performance on held-out set'
                ],
                auto_retrain_eligible=True
            )
        
        return None
    
    def _check_time_based_trigger(self) -> Optional[RetrainingTrigger]:
        """Check if enough time has passed since last retraining"""
        
        # For now, assume we need to check against a fixed schedule
        # In practice, you'd track the last retraining date
        
        days_since_last = 60  # Placeholder - would come from database
        max_days = self.retraining_thresholds['max_days_since_last_training']
        
        if days_since_last >= max_days:
            return RetrainingTrigger(
                trigger_id=f"time_based_{datetime.utcnow().strftime('%Y%m%d')}",
                trigger_type='time_based',
                triggered_at=datetime.utcnow(),
                trigger_conditions={
                    'days_since_last_training': days_since_last,
                    'max_days_threshold': max_days
                },
                severity=AlertSeverity.MEDIUM,
                estimated_impact='low',
                recommended_actions=[
                    'Schedule routine model retraining',
                    'Update with recent data',
                    'Validate continued performance'
                ],
                auto_retrain_eligible=True
            )
        
        return None
    
    def _check_bias_accumulation_trigger(self) -> Optional[RetrainingTrigger]:
        """Check if bias accumulation warrants retraining"""
        
        # This would check for accumulated bias patterns
        # For now, return None as it requires more sophisticated analysis
        
        return None
    
    def assess_system_health(self) -> Dict:
        """Comprehensive system health assessment"""
        
        logger.info("Assessing overall system health")
        
        health_scores = {}
        
        # Performance health
        performance_health = self._assess_performance_health()
        health_scores['performance'] = performance_health
        
        # Data quality health
        quality_health = self._assess_data_quality_health()
        health_scores['data_quality'] = quality_health
        
        # System stability health
        stability_health = self._assess_stability_health()
        health_scores['stability'] = stability_health
        
        # Bias health
        bias_health = self._assess_bias_health()
        health_scores['bias'] = bias_health
        
        # Calculate overall health score
        weights = {
            'performance': 0.4,
            'data_quality': 0.3,
            'stability': 0.2,
            'bias': 0.1
        }
        
        overall_score = sum(
            health_scores[component]['score'] * weights[component]
            for component in weights
            if 'score' in health_scores[component]
        )
        
        # Determine health status
        health_status = (
            'excellent' if overall_score >= 0.9 else
            'good' if overall_score >= 0.8 else
            'fair' if overall_score >= 0.7 else
            'poor' if overall_score >= 0.5 else
            'critical'
        )
        
        return {
            'overall_health': {
                'score': overall_score,
                'status': health_status,
                'color': self._get_health_color(health_status)
            },
            'component_health': health_scores,
            'health_trends': self._calculate_health_trends(),
            'immediate_actions': self._get_immediate_health_actions(health_scores),
            'assessment_timestamp': datetime.utcnow().isoformat()
        }
    
    def _assess_performance_health(self) -> Dict:
        """Assess performance health component"""
        
        # Get recent performance metrics
        recent_metrics = self.db.query(MLPerformanceMetrics).filter(
            MLPerformanceMetrics.date >= datetime.utcnow() - timedelta(days=7)
        ).order_by(desc(MLPerformanceMetrics.date)).first()
        
        if not recent_metrics:
            return {'score': 0.5, 'status': 'unknown', 'reason': 'No recent performance data'}
        
        # Calculate performance score
        accuracy = recent_metrics.accuracy or 0
        precision = recent_metrics.precision or 0
        recall = recent_metrics.recall or 0
        f1 = recent_metrics.f1_score or 0
        
        performance_score = (accuracy + precision + recall + f1) / 4
        
        status = (
            'excellent' if performance_score >= 0.9 else
            'good' if performance_score >= 0.8 else
            'fair' if performance_score >= 0.7 else
            'poor'
        )
        
        return {
            'score': performance_score,
            'status': status,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }
    
    def _assess_data_quality_health(self) -> Dict:
        """Assess data quality health component"""
        
        # Get recent quality metrics
        recent_quality = self.db.query(DataQualityMetrics).filter(
            DataQualityMetrics.date >= datetime.utcnow() - timedelta(days=7)
        ).order_by(desc(DataQualityMetrics.date)).first()
        
        if not recent_quality:
            return {'score': 0.5, 'status': 'unknown', 'reason': 'No recent quality data'}
        
        # Calculate quality score
        avg_quality = recent_quality.avg_image_quality_score or 0
        total_images = recent_quality.total_images_processed or 1
        high_quality_ratio = (recent_quality.high_quality_images or 0) / total_images
        
        quality_score = (avg_quality + high_quality_ratio) / 2
        
        status = (
            'excellent' if quality_score >= 0.8 else
            'good' if quality_score >= 0.7 else
            'fair' if quality_score >= 0.6 else
            'poor'
        )
        
        return {
            'score': quality_score,
            'status': status,
            'metrics': {
                'avg_quality': avg_quality,
                'high_quality_ratio': high_quality_ratio,
                'total_images': total_images
            }
        }
    
    def _assess_stability_health(self) -> Dict:
        """Assess system stability health component"""
        
        # Get performance variance over last 7 days
        recent_metrics = self.db.query(MLPerformanceMetrics).filter(
            MLPerformanceMetrics.date >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if len(recent_metrics) < 3:
            return {'score': 0.5, 'status': 'unknown', 'reason': 'Insufficient data for stability assessment'}
        
        # Calculate stability based on variance
        accuracy_values = [m.accuracy for m in recent_metrics if m.accuracy is not None]
        
        if len(accuracy_values) < 3:
            return {'score': 0.5, 'status': 'unknown', 'reason': 'Insufficient accuracy data'}
        
        cv = statistics.stdev(accuracy_values) / statistics.mean(accuracy_values)
        stability_score = max(0, 1 - cv * 5)  # Convert CV to stability score
        
        status = (
            'excellent' if stability_score >= 0.9 else
            'good' if stability_score >= 0.8 else
            'fair' if stability_score >= 0.7 else
            'poor'
        )
        
        return {
            'score': stability_score,
            'status': status,
            'metrics': {
                'coefficient_of_variation': cv,
                'accuracy_mean': statistics.mean(accuracy_values),
                'accuracy_std': statistics.stdev(accuracy_values)
            }
        }
    
    def _assess_bias_health(self) -> Dict:
        """Assess bias health component"""
        
        # Simple bias health assessment
        # In practice, this would use results from bias detection
        
        return {
            'score': 0.8,  # Placeholder
            'status': 'good',
            'note': 'Bias assessment requires more detailed analysis'
        }
    
    def _get_health_color(self, status: str) -> str:
        """Get color code for health status"""
        colors = {
            'excellent': '#28a745',
            'good': '#6f42c1',
            'fair': '#ffc107',
            'poor': '#fd7e14',
            'critical': '#dc3545'
        }
        return colors.get(status, '#6c757d')
    
    def _calculate_health_trends(self) -> Dict:
        """Calculate health trends over time"""
        
        # Placeholder for health trend calculation
        return {
            'trend_direction': 'stable',
            'trend_confidence': 0.7,
            'note': 'Health trends require historical data accumulation'
        }
    
    def _get_immediate_health_actions(self, health_scores: Dict) -> List[str]:
        """Get immediate actions based on health assessment"""
        
        actions = []
        
        for component, health_data in health_scores.items():
            if health_data.get('status') == 'poor':
                actions.append(f"Address {component} issues immediately")
            elif health_data.get('status') == 'fair':
                actions.append(f"Monitor {component} closely")
        
        if not actions:
            actions.append("System health is good - continue monitoring")
        
        return actions
    
    def generate_consolidated_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate consolidated recommendations from all analyses"""
        
        all_recommendations = []
        
        # Extract recommendations from different analyses
        if 'performance_analysis' in analysis_results:
            perf_analysis = analysis_results['performance_analysis']
            if 'alerts' in perf_analysis:
                for alert in perf_analysis['alerts']:
                    if isinstance(alert, dict) and 'recommendations' in alert:
                        for rec in alert['recommendations']:
                            all_recommendations.append({
                                'source': 'performance_analysis',
                                'priority': alert.get('severity', 'medium'),
                                'recommendation': rec,
                                'category': 'performance'
                            })
        
        # Add bias-related recommendations
        if 'bias_analysis' in analysis_results:
            bias_analysis = analysis_results['bias_analysis']
            if 'detected_biases' in bias_analysis:
                for bias in bias_analysis['detected_biases']:
                    if isinstance(bias, dict) and 'mitigation_strategies' in bias:
                        for strategy in bias['mitigation_strategies']:
                            all_recommendations.append({
                                'source': 'bias_analysis',
                                'priority': bias.get('severity', 'medium'),
                                'recommendation': strategy,
                                'category': 'bias_mitigation'
                            })
        
        # Add retraining recommendations
        if 'retraining_analysis' in analysis_results:
            retrain_analysis = analysis_results['retraining_analysis']
            if retrain_analysis.get('recommendation') != 'no_retraining_needed':
                all_recommendations.append({
                    'source': 'retraining_analysis',
                    'priority': 'high',
                    'recommendation': f"Consider {retrain_analysis.get('recommendation', 'retraining')}",
                    'category': 'model_maintenance'
                })
        
        # Prioritize and deduplicate
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            rec_text = rec['recommendation']
            if rec_text not in seen:
                seen.add(rec_text)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def generate_executive_summary(self, analysis_results: Dict) -> Dict:
        """Generate executive summary of advanced analytics"""
        
        # Extract key metrics
        performance_status = 'unknown'
        current_accuracy = 0.0
        
        if 'performance_analysis' in analysis_results:
            perf_analysis = analysis_results['performance_analysis']
            if 'current_performance' in perf_analysis:
                current_accuracy = perf_analysis['current_performance'].get('accuracy', 0.0)
                performance_status = 'good' if current_accuracy > 0.8 else 'needs_attention'
        
        # Count issues
        total_alerts = 0
        critical_issues = 0
        
        if 'performance_analysis' in analysis_results:
            alerts = analysis_results['performance_analysis'].get('alerts', [])
            total_alerts += len(alerts)
            critical_issues += len([a for a in alerts if isinstance(a, dict) and a.get('severity') == 'critical'])
        
        bias_count = 0
        if 'bias_analysis' in analysis_results:
            bias_count = analysis_results['bias_analysis'].get('bias_summary', {}).get('total_biases_detected', 0)
        
        # Retraining recommendation
        retraining_needed = False
        if 'retraining_analysis' in analysis_results:
            retraining_needed = analysis_results['retraining_analysis'].get('recommendation') != 'no_retraining_needed'
        
        # Overall system status
        if critical_issues > 0:
            system_status = 'critical'
        elif total_alerts > 2 or bias_count > 2:
            system_status = 'needs_attention'
        elif current_accuracy > 0.85:
            system_status = 'good'
        else:
            system_status = 'fair'
        
        return {
            'system_status': system_status,
            'key_metrics': {
                'current_accuracy': f"{current_accuracy:.3f}",
                'performance_status': performance_status,
                'total_alerts': total_alerts,
                'critical_issues': critical_issues,
                'biases_detected': bias_count,
                'retraining_recommended': retraining_needed
            },
            'top_priorities': [
                rec['recommendation'] for rec in analysis_results.get('recommendations', [])[:3]
            ],
            'health_indicator': {
                'color': self._get_health_color(system_status),
                'message': self._get_status_message(system_status)
            }
        }
    
    def _get_status_message(self, status: str) -> str:
        """Get status message for system health"""
        messages = {
            'excellent': 'System performing excellently',
            'good': 'System performing well',
            'fair': 'System performance is acceptable',
            'needs_attention': 'System requires attention',
            'critical': 'Critical issues detected - immediate action required'
        }
        return messages.get(status, 'Status unknown')
    
    # Helper methods
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        numerator = sum((2 * i - n - 1) * value for i, value in enumerate(sorted_values, 1))
        denominator = n * sum(sorted_values)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _generate_stability_recommendations(self, overall_stability: float, metrics: Dict) -> List[str]:
        """Generate stability improvement recommendations"""
        
        recommendations = []
        
        if overall_stability < 0.7:
            recommendations.extend([
                'Investigate causes of performance instability',
                'Review data collection processes for consistency',
                'Consider implementing performance monitoring alerts'
            ])
        
        # Component-specific recommendations
        for metric_name, metric_data in metrics.items():
            if metric_data.get('stability_level') == 'low':
                recommendations.append(f'Focus on improving {metric_name} consistency')
        
        return recommendations
    
    def _generate_drift_recommendations(self, drift_level: str, drift_analysis: Dict) -> List[str]:
        """Generate recommendations for addressing data drift"""
        
        recommendations = []
        
        if drift_level == 'high':
            recommendations.extend([
                'Immediate investigation of data drift causes required',
                'Consider emergency model retraining',
                'Review recent changes in data collection process'
            ])
        elif drift_level == 'medium':
            recommendations.extend([
                'Monitor drift closely for further changes',
                'Plan model retraining with recent data',
                'Investigate potential causes of drift'
            ])
        
        # Specific recommendations based on drift type
        for analysis_type, analysis_data in drift_analysis.items():
            if analysis_data.get('drift_score', 0) > 0.5:
                if analysis_type == 'quality_drift':
                    recommendations.append('Address data quality degradation')
                elif analysis_type == 'distribution_drift':
                    recommendations.append('Rebalance data collection to match original distribution')
                elif analysis_type == 'temporal_drift':
                    recommendations.append('Investigate temporal patterns in data collection')
        
        return list(set(recommendations))  # Remove duplicates



