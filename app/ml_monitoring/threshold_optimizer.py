import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
import json

from app.ml_monitoring.models import (
    GroundTruthValidation, MLPerformanceMetrics, ThresholdExperiment,
    ValidationStatus, RecognitionOutcome
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Service for dynamic threshold optimization and calibration"""
    
    def __init__(self, db: Session):
        self.db = db
        self.current_thresholds = {
            'recognition_threshold': settings.recognition_threshold,
            'high_confidence_threshold': settings.high_confidence_threshold,
            'low_confidence_threshold': settings.low_confidence_threshold
        }
    
    def analyze_threshold_performance(self, days: int = 30, min_samples: int = 100) -> Dict:
        """Analyze current threshold performance and suggest optimizations"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get validated recognition attempts
        validations = self.db.query(GroundTruthValidation).filter(
            and_(
                GroundTruthValidation.created_at >= cutoff_date,
                GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED,
                GroundTruthValidation.confidence_score.isnot(None)
            )
        ).all()
        
        if len(validations) < min_samples:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {min_samples} validated samples, have {len(validations)}',
                'sample_count': len(validations),
                'current_thresholds': self.current_thresholds
            }
        
        # Separate outcomes by confidence scores
        tp_scores = [v.confidence_score for v in validations if v.recognition_outcome == RecognitionOutcome.TRUE_POSITIVE]
        tn_scores = [v.confidence_score for v in validations if v.recognition_outcome == RecognitionOutcome.TRUE_NEGATIVE]
        fp_scores = [v.confidence_score for v in validations if v.recognition_outcome == RecognitionOutcome.FALSE_POSITIVE]
        fn_scores = [v.confidence_score for v in validations if v.recognition_outcome == RecognitionOutcome.FALSE_NEGATIVE]
        
        # Calculate ROC curve and find optimal thresholds
        roc_analysis = self._calculate_roc_curve(tp_scores, fp_scores, tn_scores, fn_scores)
        optimal_thresholds = self._find_optimal_thresholds(roc_analysis)
        
        # Analyze current threshold performance
        current_performance = self._analyze_current_performance(
            tp_scores, tn_scores, fp_scores, fn_scores
        )
        
        # Generate recommendations
        recommendations = self._generate_threshold_recommendations(
            current_performance, optimal_thresholds, roc_analysis
        )
        
        return {
            'status': 'success',
            'analysis_period': f'{days} days',
            'sample_count': len(validations),
            'outcome_distribution': {
                'true_positives': len(tp_scores),
                'true_negatives': len(tn_scores),
                'false_positives': len(fp_scores),
                'false_negatives': len(fn_scores)
            },
            'current_thresholds': self.current_thresholds,
            'current_performance': current_performance,
            'optimal_thresholds': optimal_thresholds,
            'recommendations': recommendations,
            'roc_analysis': roc_analysis
        }
    
    def _calculate_roc_curve(
        self, 
        tp_scores: List[float], 
        fp_scores: List[float], 
        tn_scores: List[float], 
        fn_scores: List[float]
    ) -> Dict:
        """Calculate ROC curve points for different thresholds"""
        
        # Combine all positive and negative cases
        positive_scores = tp_scores + fn_scores  # All actual positives
        negative_scores = tn_scores + fp_scores  # All actual negatives
        
        if not positive_scores or not negative_scores:
            return {'error': 'Insufficient data for ROC calculation'}
        
        # Test thresholds from 0.1 to 0.9
        thresholds = np.linspace(0.1, 0.9, 50)
        roc_points = []
        
        for threshold in thresholds:
            # Calculate TP, FP, TN, FN at this threshold
            tp = sum(1 for score in positive_scores if score >= threshold)
            fp = sum(1 for score in negative_scores if score >= threshold)
            tn = sum(1 for score in negative_scores if score < threshold)
            fn = sum(1 for score in positive_scores if score < threshold)
            
            # Calculate rates
            tpr = tp / len(positive_scores) if positive_scores else 0  # Sensitivity/Recall
            fpr = fp / len(negative_scores) if negative_scores else 0  # 1 - Specificity
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
            
            # Calculate accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            roc_points.append({
                'threshold': float(threshold),
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision,
                'f1_score': f1,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
        
        # Calculate AUC (Area Under Curve)
        auc = self._calculate_auc(roc_points)
        
        return {
            'roc_points': roc_points,
            'auc': auc,
            'sample_sizes': {
                'positive_samples': len(positive_scores),
                'negative_samples': len(negative_scores)
            }
        }
    
    def _calculate_auc(self, roc_points: List[Dict]) -> float:
        """Calculate Area Under ROC Curve using trapezoidal rule"""
        
        # Sort by FPR
        sorted_points = sorted(roc_points, key=lambda x: x['fpr'])
        
        auc = 0.0
        for i in range(1, len(sorted_points)):
            # Trapezoidal rule
            width = sorted_points[i]['fpr'] - sorted_points[i-1]['fpr']
            height = (sorted_points[i]['tpr'] + sorted_points[i-1]['tpr']) / 2
            auc += width * height
        
        return auc
    
    def _find_optimal_thresholds(self, roc_analysis: Dict) -> Dict:
        """Find optimal thresholds based on different optimization criteria"""
        
        if 'error' in roc_analysis:
            return {'error': roc_analysis['error']}
        
        roc_points = roc_analysis['roc_points']
        
        # Find optimal points for different criteria
        optimal_f1 = max(roc_points, key=lambda x: x['f1_score'])
        optimal_accuracy = max(roc_points, key=lambda x: x['accuracy'])
        
        # Find balanced point (minimize |TPR - (1-FPR)|)
        balanced_point = min(roc_points, key=lambda x: abs(x['tpr'] - (1 - x['fpr'])))
        
        # Find high precision point (precision > 0.9, maximize recall)
        high_precision_points = [p for p in roc_points if p['precision'] >= 0.9]
        optimal_high_precision = max(high_precision_points, key=lambda x: x['tpr']) if high_precision_points else None
        
        # Find high recall point (recall > 0.9, maximize precision)
        high_recall_points = [p for p in roc_points if p['tpr'] >= 0.9]
        optimal_high_recall = max(high_recall_points, key=lambda x: x['precision']) if high_recall_points else None
        
        return {
            'max_f1': {
                'threshold': optimal_f1['threshold'],
                'f1_score': optimal_f1['f1_score'],
                'precision': optimal_f1['precision'],
                'recall': optimal_f1['tpr'],
                'accuracy': optimal_f1['accuracy']
            },
            'max_accuracy': {
                'threshold': optimal_accuracy['threshold'],
                'accuracy': optimal_accuracy['accuracy'],
                'f1_score': optimal_accuracy['f1_score'],
                'precision': optimal_accuracy['precision'],
                'recall': optimal_accuracy['tpr']
            },
            'balanced': {
                'threshold': balanced_point['threshold'],
                'balance_score': 1 - abs(balanced_point['tpr'] - (1 - balanced_point['fpr'])),
                'precision': balanced_point['precision'],
                'recall': balanced_point['tpr'],
                'accuracy': balanced_point['accuracy']
            },
            'high_precision': optimal_high_precision,
            'high_recall': optimal_high_recall
        }
    
    def _analyze_current_performance(
        self, 
        tp_scores: List[float], 
        tn_scores: List[float], 
        fp_scores: List[float], 
        fn_scores: List[float]
    ) -> Dict:
        """Analyze performance at current thresholds"""
        
        current_threshold = self.current_thresholds['recognition_threshold']
        
        # Calculate confusion matrix at current threshold
        tp = sum(1 for score in tp_scores if score >= current_threshold)
        fp = sum(1 for score in fp_scores if score >= current_threshold)
        tn = sum(1 for score in tn_scores if score < current_threshold)
        fn = sum(1 for score in fn_scores if score < current_threshold)
        
        total = tp + fp + tn + fn
        
        if total == 0:
            return {'error': 'No data for current threshold analysis'}
        
        # Calculate metrics
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'threshold': current_threshold,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'total_samples': total
        }
    
    def _generate_threshold_recommendations(
        self, 
        current_performance: Dict, 
        optimal_thresholds: Dict, 
        roc_analysis: Dict
    ) -> List[Dict]:
        """Generate actionable threshold optimization recommendations"""
        
        recommendations = []
        
        if 'error' in current_performance or 'error' in optimal_thresholds:
            return [{'type': 'error', 'message': 'Insufficient data for recommendations'}]
        
        current_f1 = current_performance['f1_score']
        current_accuracy = current_performance['accuracy']
        current_threshold = current_performance['threshold']
        
        # Recommendation 1: F1 Score Optimization
        optimal_f1 = optimal_thresholds['max_f1']
        if optimal_f1['f1_score'] > current_f1 + 0.05:  # Significant improvement
            recommendations.append({
                'type': 'f1_optimization',
                'priority': 'high',
                'title': 'Optimize for F1 Score',
                'current_value': current_f1,
                'recommended_value': optimal_f1['f1_score'],
                'improvement': optimal_f1['f1_score'] - current_f1,
                'recommended_threshold': optimal_f1['threshold'],
                'impact': f"Could improve F1 score by {(optimal_f1['f1_score'] - current_f1):.3f}",
                'tradeoffs': f"Precision: {optimal_f1['precision']:.3f}, Recall: {optimal_f1['recall']:.3f}"
            })
        
        # Recommendation 2: Accuracy Optimization
        optimal_accuracy = optimal_thresholds['max_accuracy']
        if optimal_accuracy['accuracy'] > current_accuracy + 0.03:
            recommendations.append({
                'type': 'accuracy_optimization',
                'priority': 'medium',
                'title': 'Optimize for Accuracy',
                'current_value': current_accuracy,
                'recommended_value': optimal_accuracy['accuracy'],
                'improvement': optimal_accuracy['accuracy'] - current_accuracy,
                'recommended_threshold': optimal_accuracy['threshold'],
                'impact': f"Could improve accuracy by {(optimal_accuracy['accuracy'] - current_accuracy):.3f}",
                'tradeoffs': f"F1: {optimal_accuracy['f1_score']:.3f}"
            })
        
        # Recommendation 3: False Positive Reduction
        if current_performance['false_positive_rate'] > 0.1:  # High FP rate
            high_precision = optimal_thresholds.get('high_precision')
            if high_precision:
                recommendations.append({
                    'type': 'false_positive_reduction',
                    'priority': 'high',
                    'title': 'Reduce False Positives',
                    'current_value': current_performance['false_positive_rate'],
                    'recommended_threshold': high_precision['threshold'],
                    'impact': f"Achieve precision ≥ 90% (currently {current_performance['precision']:.3f})",
                    'tradeoffs': f"May reduce recall to {high_precision['recall']:.3f}"
                })
        
        # Recommendation 4: False Negative Reduction
        if current_performance['false_negative_rate'] > 0.15:  # High FN rate
            high_recall = optimal_thresholds.get('high_recall')
            if high_recall:
                recommendations.append({
                    'type': 'false_negative_reduction',
                    'priority': 'medium',
                    'title': 'Reduce False Negatives',
                    'current_value': current_performance['false_negative_rate'],
                    'recommended_threshold': high_recall['threshold'],
                    'impact': f"Achieve recall ≥ 90% (currently {current_performance['recall']:.3f})",
                    'tradeoffs': f"Precision may decrease to {high_recall['precision']:.3f}"
                })
        
        # Recommendation 5: Balanced Performance
        balanced = optimal_thresholds['balanced']
        if abs(balanced['threshold'] - current_threshold) > 0.1:
            recommendations.append({
                'type': 'balanced_optimization',
                'priority': 'low',
                'title': 'Balance Precision and Recall',
                'recommended_threshold': balanced['threshold'],
                'impact': f"Balance score: {balanced['balance_score']:.3f}",
                'tradeoffs': f"Precision: {balanced['precision']:.3f}, Recall: {balanced['recall']:.3f}"
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
        
        return recommendations
    
    def create_threshold_experiment(
        self, 
        recommendation: Dict, 
        creator_user_id: int,
        traffic_percentage: float = 10.0
    ) -> ThresholdExperiment:
        """Create a threshold experiment based on a recommendation"""
        
        new_threshold = recommendation['recommended_threshold']
        
        # Calculate high and low confidence thresholds relative to new threshold
        high_confidence = min(new_threshold + 0.2, 0.9)
        low_confidence = max(new_threshold - 0.2, 0.1)
        
        experiment = ThresholdExperiment(
            experiment_name=f"Auto_{recommendation['type']}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            description=f"Automated experiment based on {recommendation['title']} recommendation",
            recognition_threshold=new_threshold,
            high_confidence_threshold=high_confidence,
            low_confidence_threshold=low_confidence,
            start_date=datetime.utcnow(),
            traffic_percentage=traffic_percentage,
            created_by=creator_user_id
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        
        logger.info(f"Created threshold experiment: {experiment.experiment_name}")
        return experiment
    
    def evaluate_experiment_performance(self, experiment_id: int, days: int = 7) -> Dict:
        """Evaluate the performance of a threshold experiment"""
        
        experiment = self.db.query(ThresholdExperiment).filter(
            ThresholdExperiment.id == experiment_id
        ).first()
        
        if not experiment:
            return {'error': f'Experiment {experiment_id} not found'}
        
        # Get validation data during experiment period
        start_date = experiment.start_date
        end_date = experiment.end_date or datetime.utcnow()
        
        validations = self.db.query(GroundTruthValidation).filter(
            and_(
                GroundTruthValidation.created_at >= start_date,
                GroundTruthValidation.created_at <= end_date,
                GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED
            )
        ).all()
        
        if len(validations) < 20:  # Minimum samples for evaluation
            return {
                'status': 'insufficient_data',
                'sample_count': len(validations),
                'message': 'Need at least 20 validated samples for experiment evaluation'
            }
        
        # Calculate performance metrics using experiment thresholds
        tp = sum(1 for v in validations 
                if v.recognition_outcome == RecognitionOutcome.TRUE_POSITIVE 
                and v.confidence_score >= experiment.recognition_threshold)
        
        fp = sum(1 for v in validations 
                if v.recognition_outcome == RecognitionOutcome.FALSE_POSITIVE 
                and v.confidence_score >= experiment.recognition_threshold)
        
        tn = sum(1 for v in validations 
                if v.recognition_outcome == RecognitionOutcome.TRUE_NEGATIVE 
                and v.confidence_score < experiment.recognition_threshold)
        
        fn = sum(1 for v in validations 
                if v.recognition_outcome == RecognitionOutcome.FALSE_NEGATIVE 
                and v.confidence_score < experiment.recognition_threshold)
        
        total = tp + fp + tn + fn
        
        if total == 0:
            return {'error': 'No applicable data for experiment evaluation'}
        
        # Calculate metrics
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update experiment results
        experiment.total_samples = total
        experiment.accuracy = accuracy
        experiment.precision = precision
        experiment.recall = recall
        experiment.f1_score = f1_score
        experiment.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        experiment.false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        self.db.commit()
        
        return {
            'status': 'success',
            'experiment': {
                'id': experiment.id,
                'name': experiment.experiment_name,
                'thresholds': {
                    'recognition': experiment.recognition_threshold,
                    'high_confidence': experiment.high_confidence_threshold,
                    'low_confidence': experiment.low_confidence_threshold
                }
            },
            'performance': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'false_positive_rate': experiment.false_positive_rate,
                'false_negative_rate': experiment.false_negative_rate
            },
            'sample_count': total,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
