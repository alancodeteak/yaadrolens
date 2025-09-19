"""
ML Metrics Service
Comprehensive service for tracking ML performance, false positives/negatives, and ground truth validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
import json
import numpy as np

from app.ml_monitoring.models import (
    GroundTruthValidation, MLPerformanceMetrics, ThresholdExperiment, 
    ModelEvaluationResult, DataQualityMetrics, ValidationStatus, RecognitionOutcome
)
from app.attendance.models import AttendanceLog
from app.employees.models import Employee
from app.auth.models import User
from app.core.config import settings

logger = logging.getLogger(__name__)


class MLMetricsService:
    """Service for comprehensive ML performance tracking and validation"""
    
    def __init__(self, db: Session):
        self.db = db
        
    # ========== Ground Truth Validation ==========
    
    def create_ground_truth_validation(
        self, 
        attendance_log_id: int,
        predicted_employee_id: Optional[str],
        confidence_score: float,
        image_quality_metrics: Optional[Dict] = None
    ) -> GroundTruthValidation:
        """Create a ground truth validation record for manual review"""
        
        validation = GroundTruthValidation(
            attendance_log_id=attendance_log_id,
            predicted_employee_id=predicted_employee_id,
            confidence_score=confidence_score,
            validation_status=ValidationStatus.PENDING,
            image_quality_score=image_quality_metrics.get('quality_score') if image_quality_metrics else None,
            face_area_ratio=image_quality_metrics.get('face_area_ratio') if image_quality_metrics else None,
            brightness_score=image_quality_metrics.get('brightness') if image_quality_metrics else None,
            blur_score=image_quality_metrics.get('blur_score') if image_quality_metrics else None,
            environmental_factors=image_quality_metrics.get('environmental_factors') if image_quality_metrics else None
        )
        
        self.db.add(validation)
        self.db.commit()
        self.db.refresh(validation)
        
        logger.info(f"Created ground truth validation for attendance log {attendance_log_id}")
        return validation
    
    def validate_recognition_result(
        self,
        validation_id: int,
        actual_employee_id: Optional[str],
        validator_user_id: int,
        validation_status: ValidationStatus,
        notes: Optional[str] = None
    ) -> GroundTruthValidation:
        """Validate a recognition result and determine if it was correct"""
        
        validation = self.db.query(GroundTruthValidation).filter(
            GroundTruthValidation.id == validation_id
        ).first()
        
        if not validation:
            raise ValueError(f"Validation record {validation_id} not found")
        
        # Determine recognition outcome
        recognition_outcome = self._determine_recognition_outcome(
            validation.predicted_employee_id,
            actual_employee_id
        )
        
        # Update validation record
        validation.actual_employee_id = actual_employee_id
        validation.validation_status = validation_status
        validation.recognition_outcome = recognition_outcome
        validation.validated_by = validator_user_id
        validation.validated_at = datetime.utcnow()
        validation.validation_notes = notes
        
        self.db.commit()
        self.db.refresh(validation)
        
        # Update performance metrics
        self._update_performance_metrics_from_validation(validation)
        
        logger.info(f"Validated recognition result: {recognition_outcome} for validation {validation_id}")
        return validation
    
    def _determine_recognition_outcome(
        self, 
        predicted_employee_id: Optional[str], 
        actual_employee_id: Optional[str]
    ) -> RecognitionOutcome:
        """Determine if recognition was TP, TN, FP, or FN"""
        
        if predicted_employee_id is None and actual_employee_id is None:
            return RecognitionOutcome.TRUE_NEGATIVE  # Correctly rejected unknown person
        elif predicted_employee_id is None and actual_employee_id is not None:
            return RecognitionOutcome.FALSE_NEGATIVE  # Failed to identify known person
        elif predicted_employee_id is not None and actual_employee_id is None:
            return RecognitionOutcome.FALSE_POSITIVE  # Incorrectly identified unknown person
        elif predicted_employee_id == actual_employee_id:
            return RecognitionOutcome.TRUE_POSITIVE   # Correctly identified person
        else:
            return RecognitionOutcome.FALSE_POSITIVE  # Incorrectly identified wrong person
    
    def get_pending_validations(self, limit: int = 50) -> List[GroundTruthValidation]:
        """Get pending validation records for manual review"""
        
        return self.db.query(GroundTruthValidation).filter(
            GroundTruthValidation.validation_status == ValidationStatus.PENDING
        ).order_by(desc(GroundTruthValidation.created_at)).limit(limit).all()
    
    def get_validation_stats(self, days: int = 30) -> Dict:
        """Get validation statistics for the past N days"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        stats = self.db.query(
            GroundTruthValidation.recognition_outcome,
            func.count(GroundTruthValidation.id).label('count')
        ).filter(
            GroundTruthValidation.created_at >= cutoff_date,
            GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED
        ).group_by(GroundTruthValidation.recognition_outcome).all()
        
        result = {outcome.value: 0 for outcome in RecognitionOutcome}
        for stat in stats:
            if stat.recognition_outcome:
                result[stat.recognition_outcome] = stat.count
        
        # Calculate derived metrics
        tp = result[RecognitionOutcome.TRUE_POSITIVE.value]
        tn = result[RecognitionOutcome.TRUE_NEGATIVE.value]
        fp = result[RecognitionOutcome.FALSE_POSITIVE.value]
        fn = result[RecognitionOutcome.FALSE_NEGATIVE.value]
        
        total = tp + tn + fp + fn
        
        if total > 0:
            result.update({
                'total_validations': total,
                'accuracy': (tp + tn) / total,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (tp + fn) if (tp + fn) > 0 else 0
            })
            
            # Calculate F1 score
            precision = result['precision']
            recall = result['recall']
            result['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return result
    
    # ========== Performance Metrics Tracking ==========
    
    def record_recognition_attempt(
        self,
        success: bool,
        confidence_score: Optional[float] = None,
        processing_time: Optional[float] = None,
        image_quality_score: Optional[float] = None,
        was_cache_hit: bool = False
    ):
        """Record a recognition attempt for performance tracking"""
        
        # This will be called from the recognition service
        # For now, we'll store in memory and batch update to DB periodically
        attempt_data = {
            'timestamp': datetime.utcnow(),
            'success': success,
            'confidence_score': confidence_score,
            'processing_time': processing_time,
            'image_quality_score': image_quality_score,
            'was_cache_hit': was_cache_hit
        }
        
        # In a production system, you'd want to use a queue/buffer for this
        logger.info(f"Recognition attempt recorded: success={success}, confidence={confidence_score}")
    
    def _update_performance_metrics_from_validation(self, validation: GroundTruthValidation):
        """Update aggregated performance metrics when a validation is completed"""
        
        today = datetime.utcnow().date()
        
        # Get or create daily metrics record
        daily_metrics = self.db.query(MLPerformanceMetrics).filter(
            and_(
                func.date(MLPerformanceMetrics.date) == today,
                MLPerformanceMetrics.period_type == 'daily'
            )
        ).first()
        
        if not daily_metrics:
            daily_metrics = MLPerformanceMetrics(
                date=datetime.combine(today, datetime.min.time()),
                period_type='daily'
            )
            self.db.add(daily_metrics)
        
        # Update counts based on recognition outcome
        if validation.recognition_outcome == RecognitionOutcome.TRUE_POSITIVE:
            daily_metrics.true_positives += 1
        elif validation.recognition_outcome == RecognitionOutcome.TRUE_NEGATIVE:
            daily_metrics.true_negatives += 1
        elif validation.recognition_outcome == RecognitionOutcome.FALSE_POSITIVE:
            daily_metrics.false_positives += 1
        elif validation.recognition_outcome == RecognitionOutcome.FALSE_NEGATIVE:
            daily_metrics.false_negatives += 1
        
        # Recalculate derived metrics
        tp = daily_metrics.true_positives
        tn = daily_metrics.true_negatives
        fp = daily_metrics.false_positives
        fn = daily_metrics.false_negatives
        
        total = tp + tn + fp + fn
        
        if total > 0:
            daily_metrics.accuracy = (tp + tn) / total
            daily_metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            daily_metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision = daily_metrics.precision
            recall = daily_metrics.recall
            daily_metrics.f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.db.commit()
    
    def get_performance_trends(self, days: int = 30) -> List[Dict]:
        """Get performance metrics trends over time"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.db.query(MLPerformanceMetrics).filter(
            and_(
                MLPerformanceMetrics.date >= cutoff_date,
                MLPerformanceMetrics.period_type == 'daily'
            )
        ).order_by(MLPerformanceMetrics.date).all()
        
        return [{
            'date': metric.date.isoformat(),
            'accuracy': metric.accuracy,
            'precision': metric.precision,
            'recall': metric.recall,
            'f1_score': metric.f1_score,
            'true_positives': metric.true_positives,
            'false_positives': metric.false_positives,
            'false_negatives': metric.false_negatives,
            'total_attempts': metric.total_attempts
        } for metric in metrics]
    
    # ========== Data Quality Assessment ==========
    
    def record_image_quality_metrics(
        self,
        quality_score: float,
        face_area_ratio: float,
        brightness: float,
        blur_score: float,
        issues: List[str] = None
    ):
        """Record image quality metrics for trending analysis"""
        
        today = datetime.utcnow().date()
        
        # Get or create daily quality metrics
        quality_metrics = self.db.query(DataQualityMetrics).filter(
            and_(
                func.date(DataQualityMetrics.date) == today,
                DataQualityMetrics.period_type == 'daily'
            )
        ).first()
        
        if not quality_metrics:
            quality_metrics = DataQualityMetrics(
                date=datetime.combine(today, datetime.min.time()),
                period_type='daily'
            )
            self.db.add(quality_metrics)
        
        # Update running averages (simplified - in production you'd want proper aggregation)
        current_count = quality_metrics.total_images_processed
        new_count = current_count + 1
        
        # Update averages
        if current_count == 0:
            quality_metrics.avg_image_quality_score = quality_score
            quality_metrics.avg_face_area_ratio = face_area_ratio
            quality_metrics.avg_brightness = brightness
            quality_metrics.avg_blur_score = blur_score
        else:
            quality_metrics.avg_image_quality_score = (
                quality_metrics.avg_image_quality_score * current_count + quality_score
            ) / new_count
            quality_metrics.avg_face_area_ratio = (
                quality_metrics.avg_face_area_ratio * current_count + face_area_ratio
            ) / new_count
            quality_metrics.avg_brightness = (
                quality_metrics.avg_brightness * current_count + brightness
            ) / new_count
            quality_metrics.avg_blur_score = (
                quality_metrics.avg_blur_score * current_count + blur_score
            ) / new_count
        
        quality_metrics.total_images_processed = new_count
        
        # Update quality distribution
        if quality_score >= 0.8:
            quality_metrics.high_quality_images += 1
        elif quality_score >= 0.5:
            quality_metrics.medium_quality_images += 1
        else:
            quality_metrics.low_quality_images += 1
        
        # Update issue counts
        if issues:
            for issue in issues:
                if 'too dark' in issue.lower():
                    quality_metrics.too_dark_images += 1
                elif 'too bright' in issue.lower():
                    quality_metrics.too_bright_images += 1
                elif 'blurry' in issue.lower():
                    quality_metrics.blurry_images += 1
                elif 'face too small' in issue.lower():
                    quality_metrics.face_too_small += 1
                elif 'no face' in issue.lower():
                    quality_metrics.no_face_detected += 1
        
        self.db.commit()
    
    def get_quality_trends(self, days: int = 30) -> List[Dict]:
        """Get data quality trends over time"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.db.query(DataQualityMetrics).filter(
            and_(
                DataQualityMetrics.date >= cutoff_date,
                DataQualityMetrics.period_type == 'daily'
            )
        ).order_by(DataQualityMetrics.date).all()
        
        return [{
            'date': metric.date.isoformat(),
            'avg_quality_score': metric.avg_image_quality_score,
            'total_images': metric.total_images_processed,
            'high_quality_ratio': metric.high_quality_images / max(metric.total_images_processed, 1),
            'quality_issues': {
                'too_dark': metric.too_dark_images,
                'too_bright': metric.too_bright_images,
                'blurry': metric.blurry_images,
                'face_too_small': metric.face_too_small,
                'no_face': metric.no_face_detected
            }
        } for metric in metrics]
    
    # ========== Threshold Optimization ==========
    
    def create_threshold_experiment(
        self,
        experiment_name: str,
        recognition_threshold: float,
        high_confidence_threshold: float,
        low_confidence_threshold: float,
        creator_user_id: int,
        description: Optional[str] = None,
        traffic_percentage: float = 10.0
    ) -> ThresholdExperiment:
        """Create a new threshold experiment for A/B testing"""
        
        experiment = ThresholdExperiment(
            experiment_name=experiment_name,
            description=description,
            recognition_threshold=recognition_threshold,
            high_confidence_threshold=high_confidence_threshold,
            low_confidence_threshold=low_confidence_threshold,
            start_date=datetime.utcnow(),
            traffic_percentage=traffic_percentage,
            created_by=creator_user_id
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        
        logger.info(f"Created threshold experiment: {experiment_name}")
        return experiment
    
    def get_optimal_thresholds(self, min_samples: int = 100) -> Dict:
        """Analyze validation data to suggest optimal thresholds"""
        
        # Get recent validations with confidence scores
        validations = self.db.query(GroundTruthValidation).filter(
            and_(
                GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED,
                GroundTruthValidation.confidence_score.isnot(None)
            )
        ).limit(1000).all()
        
        if len(validations) < min_samples:
            return {
                'error': f'Insufficient data for threshold optimization. Need at least {min_samples} samples, have {len(validations)}',
                'current_thresholds': {
                    'recognition_threshold': settings.recognition_threshold,
                    'high_confidence_threshold': settings.high_confidence_threshold,
                    'low_confidence_threshold': settings.low_confidence_threshold
                }
            }
        
        # Separate true positives and false positives
        tp_scores = [v.confidence_score for v in validations if v.recognition_outcome == RecognitionOutcome.TRUE_POSITIVE]
        fp_scores = [v.confidence_score for v in validations if v.recognition_outcome == RecognitionOutcome.FALSE_POSITIVE]
        
        # Calculate ROC curve points
        thresholds = np.linspace(0.1, 0.9, 50)
        roc_points = []
        
        for threshold in thresholds:
            tp = sum(1 for score in tp_scores if score >= threshold)
            fp = sum(1 for score in fp_scores if score >= threshold)
            
            tpr = tp / len(tp_scores) if tp_scores else 0  # True Positive Rate (Recall)
            fpr = fp / len(fp_scores) if fp_scores else 0  # False Positive Rate
            
            roc_points.append({
                'threshold': threshold,
                'tpr': tpr,
                'fpr': fpr,
                'f1': 2 * tpr / (2 * tpr + fpr + (len(tp_scores) - tp) / len(tp_scores)) if tp_scores else 0
            })
        
        # Find optimal threshold (maximize F1 score)
        optimal_point = max(roc_points, key=lambda x: x['f1'])
        
        return {
            'current_thresholds': {
                'recognition_threshold': settings.recognition_threshold,
                'high_confidence_threshold': settings.high_confidence_threshold,
                'low_confidence_threshold': settings.low_confidence_threshold
            },
            'recommended_thresholds': {
                'recognition_threshold': optimal_point['threshold'],
                'high_confidence_threshold': min(optimal_point['threshold'] + 0.2, 0.9),
                'low_confidence_threshold': max(optimal_point['threshold'] - 0.2, 0.1)
            },
            'performance_at_optimal': {
                'threshold': optimal_point['threshold'],
                'true_positive_rate': optimal_point['tpr'],
                'false_positive_rate': optimal_point['fpr'],
                'f1_score': optimal_point['f1']
            },
            'sample_size': len(validations),
            'roc_curve_data': roc_points
        }
    
    # ========== Dashboard and Reporting ==========
    
    def get_ml_dashboard_data(self) -> Dict:
        """Get comprehensive ML metrics for dashboard display"""
        
        # Get recent validation stats
        validation_stats = self.get_validation_stats(days=7)
        
        # Get performance trends
        performance_trends = self.get_performance_trends(days=30)
        
        # Get quality trends
        quality_trends = self.get_quality_trends(days=30)
        
        # Get pending validations count
        pending_count = self.db.query(GroundTruthValidation).filter(
            GroundTruthValidation.validation_status == ValidationStatus.PENDING
        ).count()
        
        # Get threshold optimization recommendations
        threshold_analysis = self.get_optimal_thresholds()
        
        return {
            'validation_stats': validation_stats,
            'performance_trends': performance_trends[-7:],  # Last 7 days
            'quality_trends': quality_trends[-7:],  # Last 7 days
            'pending_validations': pending_count,
            'threshold_analysis': threshold_analysis,
            'system_health': {
                'total_employees': self.db.query(Employee).filter(Employee.is_active == True).count(),
                'recent_attendance_logs': self.db.query(AttendanceLog).filter(
                    AttendanceLog.timestamp >= datetime.utcnow() - timedelta(days=1)
                ).count()
            }
        }
