"""
Comprehensive Model Evaluation Service
Advanced model performance evaluation with cross-validation, bias detection, and benchmarking
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
import json
import os
import random
from collections import defaultdict, Counter
import statistics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import io
import base64

from app.ml_monitoring.models import (
    ModelEvaluationResult, GroundTruthValidation, ValidationStatus, RecognitionOutcome
)
from app.employees.models import Employee, TrainingPhoto, EmployeeEmbedding
from app.face_recognition.deepface_service import deepface_service
from app.face_recognition.smart_recognition_service import smart_recognition_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive service for model evaluation and performance analysis"""
    
    def __init__(self, db: Session):
        self.db = db
        self.model_name = settings.embedding_model
        self.current_thresholds = {
            'recognition': settings.recognition_threshold,
            'high_confidence': settings.high_confidence_threshold,
            'low_confidence': settings.low_confidence_threshold
        }
    
    def perform_comprehensive_evaluation(
        self, 
        evaluation_type: str = "holdout",
        test_ratio: float = 0.2,
        cross_validation_folds: int = 5,
        evaluator_user_id: int = 1
    ) -> Dict:
        """Perform comprehensive model evaluation with multiple methodologies"""
        
        logger.info(f"Starting comprehensive model evaluation: {evaluation_type}")
        
        # Get all employees with sufficient training data
        employees = self._get_employees_for_evaluation()
        
        if len(employees) < 10:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least 10 employees for evaluation, found {len(employees)}',
                'employees_found': len(employees)
            }
        
        # Prepare evaluation dataset
        evaluation_data = self._prepare_evaluation_dataset(employees)
        
        if not evaluation_data:
            return {
                'status': 'no_evaluation_data',
                'message': 'No valid evaluation data could be prepared'
            }
        
        # Perform evaluation based on type
        if evaluation_type == "holdout":
            results = self._perform_holdout_evaluation(evaluation_data, test_ratio)
        elif evaluation_type == "cross_validation":
            results = self._perform_cross_validation(evaluation_data, cross_validation_folds)
        elif evaluation_type == "temporal_split":
            results = self._perform_temporal_split_evaluation(evaluation_data)
        else:
            return {
                'status': 'invalid_evaluation_type',
                'message': f'Unknown evaluation type: {evaluation_type}'
            }
        
        # Perform bias analysis
        bias_analysis = self._perform_bias_analysis(evaluation_data, results)
        
        # Perform threshold sensitivity analysis
        threshold_analysis = self._perform_threshold_sensitivity_analysis(evaluation_data)
        
        # Generate visualizations
        visualizations = self._generate_evaluation_visualizations(results, threshold_analysis)
        
        # Save results to database
        evaluation_record = self._save_evaluation_results(
            evaluation_type, results, bias_analysis, evaluator_user_id
        )
        
        return {
            'status': 'success',
            'evaluation_id': evaluation_record.id,
            'evaluation_type': evaluation_type,
            'dataset_info': {
                'total_employees': len(employees),
                'total_images': len(evaluation_data),
                'evaluation_date': datetime.utcnow().isoformat()
            },
            'performance_metrics': results,
            'bias_analysis': bias_analysis,
            'threshold_analysis': threshold_analysis,
            'visualizations': visualizations,
            'recommendations': self._generate_evaluation_recommendations(results, bias_analysis)
        }
    
    def _get_employees_for_evaluation(self) -> List[Employee]:
        """Get employees suitable for evaluation (with sufficient training data)"""
        
        # Get employees with at least minimum training photos
        min_photos = max(5, settings.training_photos_required // 2)
        
        employees = self.db.query(Employee).filter(
            and_(
                Employee.is_active == True,
                Employee.id.in_(
                    self.db.query(TrainingPhoto.employee_id).group_by(
                        TrainingPhoto.employee_id
                    ).having(func.count(TrainingPhoto.id) >= min_photos)
                )
            )
        ).all()
        
        return employees
    
    def _prepare_evaluation_dataset(self, employees: List[Employee]) -> List[Dict]:
        """Prepare evaluation dataset with embeddings and labels"""
        
        evaluation_data = []
        
        for employee in employees:
            # Get training photos for this employee
            training_photos = self.db.query(TrainingPhoto).filter(
                TrainingPhoto.employee_id == employee.id
            ).all()
            
            for photo in training_photos:
                if not os.path.exists(photo.file_path):
                    continue
                
                try:
                    # Extract embedding
                    embedding = deepface_service.extract_embedding(photo.file_path)
                    if embedding is not None:
                        evaluation_data.append({
                            'employee_id': employee.id,
                            'employee_name': employee.name,
                            'embedding': embedding.tolist(),
                            'image_path': photo.file_path,
                            'photo_id': photo.id,
                            'created_at': photo.created_at
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for photo {photo.id}: {str(e)}")
                    continue
        
        logger.info(f"Prepared evaluation dataset with {len(evaluation_data)} samples from {len(employees)} employees")
        return evaluation_data
    
    def _perform_holdout_evaluation(self, evaluation_data: List[Dict], test_ratio: float) -> Dict:
        """Perform holdout evaluation (train/test split)"""
        
        # Group by employee
        employee_data = defaultdict(list)
        for item in evaluation_data:
            employee_data[item['employee_id']].append(item)
        
        # Split each employee's data
        train_data = []
        test_data = []
        
        for employee_id, items in employee_data.items():
            random.shuffle(items)
            split_idx = int(len(items) * (1 - test_ratio))
            train_data.extend(items[:split_idx])
            test_data.extend(items[split_idx:])
        
        if len(test_data) == 0:
            return {'error': 'No test data after split'}
        
        # Build embeddings database from training data
        train_embeddings = self._build_embeddings_db(train_data)
        
        # Evaluate on test data
        results = self._evaluate_embeddings(test_data, train_embeddings)
        
        results.update({
            'evaluation_method': 'holdout',
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'test_ratio': test_ratio
        })
        
        return results
    
    def _perform_cross_validation(self, evaluation_data: List[Dict], n_folds: int) -> Dict:
        """Perform k-fold cross validation"""
        
        # Group by employee for stratified splitting
        employee_data = defaultdict(list)
        for item in evaluation_data:
            employee_data[item['employee_id']].append(item)
        
        employees = list(employee_data.keys())
        if len(employees) < n_folds:
            return {'error': f'Not enough employees ({len(employees)}) for {n_folds}-fold CV'}
        
        # Create stratified folds
        random.shuffle(employees)
        fold_size = len(employees) // n_folds
        
        fold_results = []
        
        for fold in range(n_folds):
            # Define test employees for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else len(employees)
            test_employees = employees[start_idx:end_idx]
            train_employees = [emp for emp in employees if emp not in test_employees]
            
            # Prepare fold data
            train_data = []
            test_data = []
            
            for emp in train_employees:
                train_data.extend(employee_data[emp])
            
            for emp in test_employees:
                test_data.extend(employee_data[emp])
            
            # Build embeddings and evaluate
            train_embeddings = self._build_embeddings_db(train_data)
            fold_result = self._evaluate_embeddings(test_data, train_embeddings)
            fold_result['fold'] = fold + 1
            fold_results.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(fold_results)
        aggregated_results.update({
            'evaluation_method': 'cross_validation',
            'n_folds': n_folds,
            'fold_results': fold_results
        })
        
        return aggregated_results
    
    def _perform_temporal_split_evaluation(self, evaluation_data: List[Dict]) -> Dict:
        """Perform temporal split evaluation (older data for training, newer for testing)"""
        
        # Sort by creation date
        evaluation_data.sort(key=lambda x: x['created_at'])
        
        # Split at 80% point
        split_idx = int(len(evaluation_data) * 0.8)
        train_data = evaluation_data[:split_idx]
        test_data = evaluation_data[split_idx:]
        
        if len(test_data) == 0:
            return {'error': 'No test data after temporal split'}
        
        # Build embeddings and evaluate
        train_embeddings = self._build_embeddings_db(train_data)
        results = self._evaluate_embeddings(test_data, train_embeddings)
        
        results.update({
            'evaluation_method': 'temporal_split',
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'temporal_split_date': train_data[-1]['created_at'].isoformat() if train_data else None
        })
        
        return results
    
    def _build_embeddings_db(self, train_data: List[Dict]) -> List[Dict]:
        """Build embeddings database from training data"""
        
        embeddings_db = []
        for item in train_data:
            embeddings_db.append({
                'employee_id': item['employee_id'],
                'embedding': item['embedding']
            })
        
        return embeddings_db
    
    def _evaluate_embeddings(self, test_data: List[Dict], train_embeddings: List[Dict]) -> Dict:
        """Evaluate embeddings against test data"""
        
        y_true = []
        y_pred = []
        y_scores = []
        predictions = []
        
        for test_item in test_data:
            true_employee = test_item['employee_id']
            test_embedding = np.array(test_item['embedding'])
            
            # Find best match
            best_similarity = 0.0
            best_match = None
            
            for train_item in train_embeddings:
                train_embedding = np.array(train_item['embedding'])
                similarity = self._cosine_similarity(test_embedding, train_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = train_item['employee_id']
            
            # Record results
            y_true.append(true_employee)
            y_pred.append(best_match if best_similarity >= self.current_thresholds['recognition'] else None)
            y_scores.append(best_similarity)
            
            predictions.append({
                'true_employee': true_employee,
                'predicted_employee': best_match,
                'confidence': best_similarity,
                'correct': best_match == true_employee and best_similarity >= self.current_thresholds['recognition']
            })
        
        # Calculate metrics
        metrics = self._calculate_detailed_metrics(y_true, y_pred, y_scores, predictions)
        
        return metrics
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def _calculate_detailed_metrics(
        self, 
        y_true: List, 
        y_pred: List, 
        y_scores: List, 
        predictions: List[Dict]
    ) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic counts
        total_samples = len(y_true)
        correct_predictions = sum(1 for p in predictions if p['correct'])
        
        # Accuracy
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # Precision, Recall, F1 for each employee
        employee_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
        
        for pred in predictions:
            true_emp = pred['true_employee']
            pred_emp = pred['predicted_employee']
            
            if pred_emp == true_emp and pred_emp is not None:
                employee_metrics[true_emp]['tp'] += 1
            elif pred_emp is not None and pred_emp != true_emp:
                employee_metrics[pred_emp]['fp'] += 1
                employee_metrics[true_emp]['fn'] += 1
            else:  # pred_emp is None
                employee_metrics[true_emp]['fn'] += 1
        
        # Calculate macro-averaged metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for emp_id, metrics in employee_metrics.items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        macro_precision = statistics.mean(precision_scores) if precision_scores else 0
        macro_recall = statistics.mean(recall_scores) if recall_scores else 0
        macro_f1 = statistics.mean(f1_scores) if f1_scores else 0
        
        # Confidence analysis
        correct_confidences = [p['confidence'] for p in predictions if p['correct']]
        incorrect_confidences = [p['confidence'] for p in predictions if not p['correct']]
        
        # ROC/AUC calculation (binary classification per employee)
        roc_auc_scores = []
        unique_employees = list(set(y_true))
        
        for employee in unique_employees:
            # Binary labels for this employee
            binary_true = [1 if emp == employee else 0 for emp in y_true]
            binary_scores = []
            
            for i, pred in enumerate(predictions):
                if pred['predicted_employee'] == employee:
                    binary_scores.append(pred['confidence'])
                else:
                    binary_scores.append(0.0)
            
            if sum(binary_true) > 0 and sum(binary_true) < len(binary_true):
                try:
                    fpr, tpr, _ = roc_curve(binary_true, binary_scores)
                    roc_auc = auc(fpr, tpr)
                    roc_auc_scores.append(roc_auc)
                except Exception:
                    pass
        
        avg_auc = statistics.mean(roc_auc_scores) if roc_auc_scores else 0
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'avg_auc': avg_auc,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'confidence_analysis': {
                'avg_correct_confidence': statistics.mean(correct_confidences) if correct_confidences else 0,
                'avg_incorrect_confidence': statistics.mean(incorrect_confidences) if incorrect_confidences else 0,
                'confidence_separation': (
                    statistics.mean(correct_confidences) - statistics.mean(incorrect_confidences)
                    if correct_confidences and incorrect_confidences else 0
                )
            },
            'per_employee_metrics': dict(employee_metrics),
            'unique_employees_tested': len(unique_employees),
            'prediction_details': predictions[:100]  # Limit for storage
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results"""
        
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'avg_auc']
        aggregated = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'{metric}_mean'] = statistics.mean(values)
                aggregated[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0
                aggregated[f'{metric}_min'] = min(values)
                aggregated[f'{metric}_max'] = max(values)
        
        # Aggregate confidence analysis
        correct_confidences = []
        incorrect_confidences = []
        
        for fold in fold_results:
            if 'confidence_analysis' in fold:
                if fold['confidence_analysis']['avg_correct_confidence'] > 0:
                    correct_confidences.append(fold['confidence_analysis']['avg_correct_confidence'])
                if fold['confidence_analysis']['avg_incorrect_confidence'] > 0:
                    incorrect_confidences.append(fold['confidence_analysis']['avg_incorrect_confidence'])
        
        aggregated['confidence_analysis'] = {
            'avg_correct_confidence_mean': statistics.mean(correct_confidences) if correct_confidences else 0,
            'avg_incorrect_confidence_mean': statistics.mean(incorrect_confidences) if incorrect_confidences else 0,
            'confidence_separation_mean': (
                statistics.mean(correct_confidences) - statistics.mean(incorrect_confidences)
                if correct_confidences and incorrect_confidences else 0
            )
        }
        
        return aggregated
    
    def _perform_bias_analysis(self, evaluation_data: List[Dict], results: Dict) -> Dict:
        """Perform bias analysis across different demographic groups"""
        
        logger.info("Performing bias analysis")
        
        # For now, we'll analyze bias based on data availability and quality
        # In a production system, you'd have demographic data
        
        bias_analysis = {
            'data_distribution_bias': self._analyze_data_distribution_bias(evaluation_data),
            'performance_consistency': self._analyze_performance_consistency(evaluation_data, results),
            'temporal_bias': self._analyze_temporal_bias(evaluation_data),
            'quality_bias': self._analyze_quality_bias(evaluation_data)
        }
        
        return bias_analysis
    
    def _analyze_data_distribution_bias(self, evaluation_data: List[Dict]) -> Dict:
        """Analyze bias in data distribution"""
        
        # Count samples per employee
        employee_counts = Counter(item['employee_id'] for item in evaluation_data)
        
        counts = list(employee_counts.values())
        mean_count = statistics.mean(counts)
        std_count = statistics.stdev(counts) if len(counts) > 1 else 0
        
        # Calculate Gini coefficient for data distribution inequality
        gini_coefficient = self._calculate_gini_coefficient(counts)
        
        # Identify under-represented employees
        under_represented = [
            emp_id for emp_id, count in employee_counts.items() 
            if count < mean_count - std_count
        ]
        
        return {
            'total_employees': len(employee_counts),
            'avg_samples_per_employee': mean_count,
            'std_samples_per_employee': std_count,
            'gini_coefficient': gini_coefficient,
            'data_inequality_level': (
                'high' if gini_coefficient > 0.4 else 
                'medium' if gini_coefficient > 0.25 else 'low'
            ),
            'under_represented_employees': len(under_represented),
            'distribution_fairness_score': max(0, 1 - gini_coefficient)
        }
    
    def _analyze_performance_consistency(self, evaluation_data: List[Dict], results: Dict) -> Dict:
        """Analyze performance consistency across employees"""
        
        if 'per_employee_metrics' not in results:
            return {'error': 'No per-employee metrics available'}
        
        employee_f1_scores = []
        employee_recall_scores = []
        
        for emp_id, metrics in results['per_employee_metrics'].items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            employee_f1_scores.append(f1)
            employee_recall_scores.append(recall)
        
        f1_std = statistics.stdev(employee_f1_scores) if len(employee_f1_scores) > 1 else 0
        recall_std = statistics.stdev(employee_recall_scores) if len(employee_recall_scores) > 1 else 0
        
        return {
            'f1_score_consistency': max(0, 1 - f1_std),
            'recall_consistency': max(0, 1 - recall_std),
            'performance_variance': {
                'f1_std': f1_std,
                'recall_std': recall_std
            },
            'consistency_level': (
                'high' if f1_std < 0.1 and recall_std < 0.1 else
                'medium' if f1_std < 0.2 and recall_std < 0.2 else 'low'
            )
        }
    
    def _analyze_temporal_bias(self, evaluation_data: List[Dict]) -> Dict:
        """Analyze temporal bias in the data"""
        
        # Group data by time periods
        data_by_month = defaultdict(list)
        
        for item in evaluation_data:
            month_key = item['created_at'].strftime('%Y-%m')
            data_by_month[month_key].append(item)
        
        monthly_counts = {month: len(items) for month, items in data_by_month.items()}
        
        if len(monthly_counts) <= 1:
            return {
                'temporal_distribution': 'insufficient_data',
                'months_covered': len(monthly_counts)
            }
        
        counts = list(monthly_counts.values())
        temporal_variance = statistics.stdev(counts) / statistics.mean(counts) if counts else 0
        
        return {
            'months_covered': len(monthly_counts),
            'temporal_variance': temporal_variance,
            'temporal_consistency': max(0, 1 - temporal_variance),
            'monthly_distribution': monthly_counts,
            'temporal_bias_level': (
                'high' if temporal_variance > 0.5 else
                'medium' if temporal_variance > 0.3 else 'low'
            )
        }
    
    def _analyze_quality_bias(self, evaluation_data: List[Dict]) -> Dict:
        """Analyze bias related to image quality"""
        
        # This would require quality scores for each image
        # For now, we'll provide a framework
        
        return {
            'quality_analysis': 'not_implemented',
            'note': 'Quality bias analysis requires quality scores for training images'
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        numerator = sum((2 * i - n - 1) * value for i, value in enumerate(sorted_values, 1))
        denominator = n * sum(sorted_values)
        
        return numerator / denominator if denominator > 0 else 0
    
    def _perform_threshold_sensitivity_analysis(self, evaluation_data: List[Dict]) -> Dict:
        """Analyze sensitivity to different threshold values"""
        
        logger.info("Performing threshold sensitivity analysis")
        
        # Test different threshold values
        thresholds = np.linspace(0.1, 0.9, 17)  # 0.1 to 0.9 in steps of 0.05
        
        # Split data for analysis
        random.shuffle(evaluation_data)
        split_idx = int(len(evaluation_data) * 0.8)
        train_data = evaluation_data[:split_idx]
        test_data = evaluation_data[split_idx:]
        
        if len(test_data) == 0:
            return {'error': 'Insufficient data for threshold analysis'}
        
        train_embeddings = self._build_embeddings_db(train_data)
        
        threshold_results = []
        
        for threshold in thresholds:
            # Temporarily set threshold
            original_threshold = self.current_thresholds['recognition']
            self.current_thresholds['recognition'] = threshold
            
            # Evaluate with this threshold
            results = self._evaluate_embeddings(test_data, train_embeddings)
            
            threshold_results.append({
                'threshold': float(threshold),
                'accuracy': results['accuracy'],
                'precision': results['macro_precision'],
                'recall': results['macro_recall'],
                'f1_score': results['macro_f1']
            })
            
            # Restore original threshold
            self.current_thresholds['recognition'] = original_threshold
        
        # Find optimal threshold
        optimal_f1 = max(threshold_results, key=lambda x: x['f1_score'])
        optimal_accuracy = max(threshold_results, key=lambda x: x['accuracy'])
        
        return {
            'threshold_sensitivity_curve': threshold_results,
            'optimal_thresholds': {
                'best_f1': optimal_f1,
                'best_accuracy': optimal_accuracy
            },
            'current_threshold_performance': next(
                (r for r in threshold_results if abs(r['threshold'] - original_threshold) < 0.01),
                None
            )
        }
    
    def _generate_evaluation_visualizations(self, results: Dict, threshold_analysis: Dict) -> Dict:
        """Generate evaluation visualizations (base64 encoded images)"""
        
        visualizations = {}
        
        try:
            # ROC Curve visualization
            if 'threshold_sensitivity_curve' in threshold_analysis:
                roc_plot = self._create_threshold_sensitivity_plot(threshold_analysis)
                visualizations['threshold_sensitivity'] = roc_plot
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _create_threshold_sensitivity_plot(self, threshold_analysis: Dict) -> str:
        """Create threshold sensitivity plot"""
        
        data = threshold_analysis['threshold_sensitivity_curve']
        thresholds = [d['threshold'] for d in data]
        accuracies = [d['accuracy'] for d in data]
        f1_scores = [d['f1_score'] for d in data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
        plt.plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Sensitivity Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
    
    def _save_evaluation_results(
        self, 
        evaluation_type: str, 
        results: Dict, 
        bias_analysis: Dict, 
        evaluator_user_id: int
    ) -> ModelEvaluationResult:
        """Save evaluation results to database"""
        
        evaluation_record = ModelEvaluationResult(
            model_version="1.0",
            model_name=self.model_name,
            evaluation_type=evaluation_type,
            dataset_size=results.get('total_samples', 0),
            test_samples=results.get('test_samples', 0),
            accuracy=results.get('accuracy', 0),
            precision=results.get('macro_precision', 0),
            recall=results.get('macro_recall', 0),
            f1_score=results.get('macro_f1', 0),
            auc_score=results.get('avg_auc', 0),
            avg_true_positive_confidence=results.get('confidence_analysis', {}).get('avg_correct_confidence', 0),
            avg_false_positive_confidence=results.get('confidence_analysis', {}).get('avg_incorrect_confidence', 0),
            performance_by_group=bias_analysis,
            evaluated_by=evaluator_user_id,
            notes=f"Comprehensive evaluation using {evaluation_type} methodology"
        )
        
        self.db.add(evaluation_record)
        self.db.commit()
        self.db.refresh(evaluation_record)
        
        logger.info(f"Saved evaluation results with ID: {evaluation_record.id}")
        return evaluation_record
    
    def _generate_evaluation_recommendations(self, results: Dict, bias_analysis: Dict) -> List[Dict]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Performance recommendations
        if results.get('accuracy', 0) < 0.9:
            recommendations.append({
                'type': 'performance_improvement',
                'priority': 'high',
                'title': 'Improve Overall Accuracy',
                'description': f"Current accuracy is {results.get('accuracy', 0):.3f}. Target: >0.90",
                'actions': [
                    'Collect more high-quality training data',
                    'Improve image quality standards',
                    'Consider model fine-tuning',
                    'Optimize recognition thresholds'
                ]
            })
        
        # Data distribution recommendations
        data_bias = bias_analysis.get('data_distribution_bias', {})
        if data_bias.get('gini_coefficient', 0) > 0.4:
            recommendations.append({
                'type': 'data_balance',
                'priority': 'medium',
                'title': 'Balance Training Data Distribution',
                'description': f"High data inequality (Gini: {data_bias.get('gini_coefficient', 0):.3f})",
                'actions': [
                    'Collect more data for under-represented employees',
                    'Use data augmentation techniques',
                    'Implement balanced sampling strategies'
                ]
            })
        
        # Consistency recommendations
        perf_consistency = bias_analysis.get('performance_consistency', {})
        if perf_consistency.get('consistency_level') == 'low':
            recommendations.append({
                'type': 'consistency_improvement',
                'priority': 'medium',
                'title': 'Improve Performance Consistency',
                'description': "Large performance variations across employees detected",
                'actions': [
                    'Standardize data collection process',
                    'Implement quality controls',
                    'Review outlier cases',
                    'Consider employee-specific thresholds'
                ]
            })
        
        # Confidence separation recommendations
        conf_analysis = results.get('confidence_analysis', {})
        conf_separation = conf_analysis.get('confidence_separation', 0)
        if conf_separation < 0.2:
            recommendations.append({
                'type': 'confidence_improvement',
                'priority': 'high',
                'title': 'Improve Confidence Separation',
                'description': f"Low separation between correct/incorrect predictions: {conf_separation:.3f}",
                'actions': [
                    'Review and improve training data quality',
                    'Consider ensemble methods',
                    'Implement uncertainty quantification',
                    'Add quality-based confidence adjustment'
                ]
            })
        
        return recommendations
    
    def compare_models(self, model_versions: List[str]) -> Dict:
        """Compare performance across different model versions"""
        
        # Get evaluation results for specified models
        evaluations = self.db.query(ModelEvaluationResult).filter(
            ModelEvaluationResult.model_version.in_(model_versions)
        ).order_by(desc(ModelEvaluationResult.evaluation_date)).all()
        
        if not evaluations:
            return {
                'status': 'no_data',
                'message': 'No evaluation results found for specified model versions'
            }
        
        # Group by model version
        model_results = defaultdict(list)
        for eval_result in evaluations:
            model_results[eval_result.model_version].append(eval_result)
        
        # Compare metrics
        comparison = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        for model_version, results in model_results.items():
            if results:
                latest_result = results[0]  # Most recent
                comparison[model_version] = {
                    'evaluation_date': latest_result.evaluation_date.isoformat(),
                    'metrics': {
                        metric: getattr(latest_result, metric, 0) 
                        for metric in metrics
                    },
                    'dataset_size': latest_result.dataset_size
                }
        
        # Find best performing model
        best_model = None
        best_f1 = 0
        
        for model_version, data in comparison.items():
            f1_score = data['metrics']['f1_score']
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_version
        
        return {
            'status': 'success',
            'comparison': comparison,
            'best_model': {
                'version': best_model,
                'f1_score': best_f1
            },
            'recommendation': f"Model {best_model} shows the best performance" if best_model else "No clear winner"
        }



