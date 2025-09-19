"""
ML Monitoring API Routes
Endpoints for managing ML performance metrics, ground truth validation, and threshold optimization
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.dependencies import get_current_admin_user, get_current_user
from app.auth.models import User
from app.ml_monitoring.ml_metrics_service import MLMetricsService
from app.ml_monitoring.models import ValidationStatus, RecognitionOutcome
from app.ml_monitoring import schemas
from app.ml_monitoring.data_quality_analyzer import DataQualityAnalyzer
from app.ml_monitoring.model_evaluator import ModelEvaluator
from app.ml_monitoring.advanced_analytics import AdvancedAnalytics
from app.face_recognition.smart_recognition_service import smart_recognition_service

router = APIRouter(prefix="/ml-monitoring", tags=["ML Monitoring"])


# ========== Ground Truth Validation Endpoints ==========

@router.get("/validations/pending", response_model=List[schemas.GroundTruthValidationResponse])
async def get_pending_validations(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get pending validation records for manual review"""
    
    ml_service = MLMetricsService(db)
    validations = ml_service.get_pending_validations(limit=limit)
    
    return validations


@router.post("/validations/{validation_id}/validate")
async def validate_recognition_result(
    validation_id: int,
    validation_data: schemas.ValidationRequest,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Validate a recognition result and mark it as correct/incorrect"""
    
    ml_service = MLMetricsService(db)
    
    try:
        validation = ml_service.validate_recognition_result(
            validation_id=validation_id,
            actual_employee_id=validation_data.actual_employee_id,
            validator_user_id=current_user.id,
            validation_status=validation_data.validation_status,
            notes=validation_data.notes
        )
        
        return {
            "success": True,
            "message": f"Validation completed: {validation.recognition_outcome}",
            "validation": validation
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/validations/stats")
async def get_validation_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get validation statistics for the specified period"""
    
    ml_service = MLMetricsService(db)
    stats = ml_service.get_validation_stats(days=days)
    
    return {
        "period_days": days,
        "statistics": stats
    }


# ========== Performance Metrics Endpoints ==========

@router.get("/performance/trends")
async def get_performance_trends(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get performance metrics trends over time"""
    
    ml_service = MLMetricsService(db)
    trends = ml_service.get_performance_trends(days=days)
    
    return {
        "period_days": days,
        "trends": trends
    }


@router.get("/performance/current")
async def get_current_performance_stats(
    current_user: User = Depends(get_current_admin_user)
):
    """Get current performance statistics from the recognition service"""
    
    return {
        "current_session_stats": smart_recognition_service.get_enhanced_recognition_stats(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/performance/reset-stats")
async def reset_performance_stats(
    current_user: User = Depends(get_current_admin_user)
):
    """Reset current session performance statistics"""
    
    smart_recognition_service.reset_stats()
    
    return {
        "success": True,
        "message": "Performance statistics reset successfully",
        "timestamp": datetime.utcnow().isoformat()
    }


# ========== Data Quality Endpoints ==========

@router.get("/quality/trends")
async def get_quality_trends(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get data quality trends over time"""
    
    ml_service = MLMetricsService(db)
    trends = ml_service.get_quality_trends(days=days)
    
    return {
        "period_days": days,
        "quality_trends": trends
    }


# ========== Threshold Optimization Endpoints ==========

@router.get("/thresholds/current")
async def get_current_thresholds(
    current_user: User = Depends(get_current_admin_user)
):
    """Get current threshold configuration"""
    
    from app.core.config import settings
    
    return {
        "current_thresholds": {
            "recognition_threshold": settings.recognition_threshold,
            "high_confidence_threshold": settings.high_confidence_threshold,
            "low_confidence_threshold": settings.low_confidence_threshold
        },
        "model": settings.embedding_model
    }


@router.get("/thresholds/optimization")
async def get_threshold_optimization(
    min_samples: int = Query(100, ge=50, le=1000),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get threshold optimization recommendations based on validation data"""
    
    ml_service = MLMetricsService(db)
    optimization = ml_service.get_optimal_thresholds(min_samples=min_samples)
    
    return optimization


@router.post("/thresholds/experiment")
async def create_threshold_experiment(
    experiment_data: schemas.ThresholdExperimentRequest,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Create a new threshold experiment for A/B testing"""
    
    ml_service = MLMetricsService(db)
    
    experiment = ml_service.create_threshold_experiment(
        experiment_name=experiment_data.experiment_name,
        recognition_threshold=experiment_data.recognition_threshold,
        high_confidence_threshold=experiment_data.high_confidence_threshold,
        low_confidence_threshold=experiment_data.low_confidence_threshold,
        creator_user_id=current_user.id,
        description=experiment_data.description,
        traffic_percentage=experiment_data.traffic_percentage
    )
    
    return {
        "success": True,
        "message": f"Threshold experiment '{experiment_data.experiment_name}' created successfully",
        "experiment": experiment
    }


# ========== Dashboard and Reporting Endpoints ==========

@router.get("/dashboard")
async def get_ml_dashboard_data(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive ML metrics for dashboard display"""
    
    ml_service = MLMetricsService(db)
    dashboard_data = ml_service.get_ml_dashboard_data()
    
    # Add current session stats
    dashboard_data["current_session"] = smart_recognition_service.get_enhanced_recognition_stats()
    
    return dashboard_data


@router.get("/reports/false-positives")
async def get_false_positive_analysis(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed analysis of false positive cases"""
    
    ml_service = MLMetricsService(db)
    
    # Get recent false positive validations
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    false_positives = db.query(ml_service.db.query(
        ml_service.GroundTruthValidation
    ).filter(
        ml_service.GroundTruthValidation.recognition_outcome == RecognitionOutcome.FALSE_POSITIVE,
        ml_service.GroundTruthValidation.created_at >= cutoff_date,
        ml_service.GroundTruthValidation.validation_status == ValidationStatus.CONFIRMED
    ).all())
    
    # Analyze patterns
    confidence_distribution = [fp.confidence_score for fp in false_positives if fp.confidence_score]
    quality_distribution = [fp.image_quality_score for fp in false_positives if fp.image_quality_score]
    
    return {
        "period_days": days,
        "total_false_positives": len(false_positives),
        "analysis": {
            "avg_confidence": sum(confidence_distribution) / len(confidence_distribution) if confidence_distribution else 0,
            "avg_quality": sum(quality_distribution) / len(quality_distribution) if quality_distribution else 0,
            "confidence_range": {
                "min": min(confidence_distribution) if confidence_distribution else 0,
                "max": max(confidence_distribution) if confidence_distribution else 0
            },
            "common_issues": [
                "Low image quality",
                "Poor lighting conditions",
                "Similar facial features",
                "Outdated employee photos"
            ]
        },
        "recommendations": [
            "Consider raising recognition threshold",
            "Improve image quality validation",
            "Update employee training photos",
            "Implement stricter quality controls"
        ]
    }


@router.get("/reports/model-performance")
async def get_model_performance_report(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive model performance report"""
    
    ml_service = MLMetricsService(db)
    
    # Get validation stats
    validation_stats = ml_service.get_validation_stats(days=days)
    
    # Get performance trends
    performance_trends = ml_service.get_performance_trends(days=days)
    
    # Get quality trends
    quality_trends = ml_service.get_quality_trends(days=days)
    
    # Calculate summary metrics
    if performance_trends:
        recent_accuracy = [t['accuracy'] for t in performance_trends[-7:] if t['accuracy']]
        avg_recent_accuracy = sum(recent_accuracy) / len(recent_accuracy) if recent_accuracy else 0
    else:
        avg_recent_accuracy = 0
    
    return {
        "report_period": {
            "days": days,
            "start_date": (datetime.utcnow() - timedelta(days=days)).isoformat(),
            "end_date": datetime.utcnow().isoformat()
        },
        "summary": {
            "overall_accuracy": validation_stats.get('accuracy', 0),
            "precision": validation_stats.get('precision', 0),
            "recall": validation_stats.get('recall', 0),
            "f1_score": validation_stats.get('f1_score', 0),
            "recent_avg_accuracy": avg_recent_accuracy,
            "total_validations": validation_stats.get('total_validations', 0)
        },
        "detailed_metrics": {
            "validation_stats": validation_stats,
            "performance_trends": performance_trends,
            "quality_trends": quality_trends
        },
        "model_info": {
            "model_name": "ArcFace",
            "current_thresholds": {
                "recognition": 0.5,
                "high_confidence": 0.8,
                "low_confidence": 0.3
            }
        }
    }


# ========== Manual Correction Endpoints ==========

@router.post("/corrections/record")
async def record_manual_correction(
    correction_data: schemas.ManualCorrectionRequest,
    current_user: User = Depends(get_current_admin_user)
):
    """Record a manual correction for tracking false positives/negatives"""
    
    smart_recognition_service.record_manual_correction(
        was_false_positive=correction_data.was_false_positive,
        was_false_negative=correction_data.was_false_negative
    )
    
    return {
        "success": True,
        "message": "Manual correction recorded successfully",
        "correction_type": "false_positive" if correction_data.was_false_positive else "false_negative",
        "timestamp": datetime.utcnow().isoformat()
    }


# ========== Advanced Data Quality Analysis ==========

@router.get("/quality/training-analysis")
async def analyze_training_data_quality(
    employee_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Comprehensive analysis of training data quality and diversity"""
    
    analyzer = DataQualityAnalyzer(db)
    analysis = analyzer.analyze_training_data_quality(employee_id)
    
    return analysis


@router.get("/quality/realtime-analysis")
async def analyze_realtime_quality(
    days: int = Query(7, ge=1, le=30),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Analyze real-time image quality trends"""
    
    analyzer = DataQualityAnalyzer(db)
    analysis = analyzer.analyze_real_time_quality_trends(days)
    
    return analysis


@router.get("/quality/comprehensive-report")
async def generate_quality_report(
    days: int = Query(30, ge=7, le=90),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Generate comprehensive data quality report"""
    
    analyzer = DataQualityAnalyzer(db)
    report = analyzer.generate_quality_report(days)
    
    return report


# ========== Model Evaluation Endpoints ==========

@router.post("/model/evaluate")
async def perform_model_evaluation(
    evaluation_type: str = Body("holdout", regex="^(holdout|cross_validation|temporal_split)$"),
    test_ratio: float = Body(0.2, ge=0.1, le=0.5),
    cross_validation_folds: int = Body(5, ge=3, le=10),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Perform comprehensive model evaluation"""
    
    evaluator = ModelEvaluator(db)
    evaluation = evaluator.perform_comprehensive_evaluation(
        evaluation_type=evaluation_type,
        test_ratio=test_ratio,
        cross_validation_folds=cross_validation_folds,
        evaluator_user_id=current_user.id
    )
    
    return evaluation


@router.get("/model/evaluations")
async def get_model_evaluations(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get recent model evaluation results"""
    
    from app.ml_monitoring.models import ModelEvaluationResult
    
    evaluations = db.query(ModelEvaluationResult).order_by(
        ModelEvaluationResult.evaluation_date.desc()
    ).limit(limit).all()
    
    return [
        {
            'id': eval_result.id,
            'model_name': eval_result.model_name,
            'model_version': eval_result.model_version,
            'evaluation_type': eval_result.evaluation_type,
            'accuracy': eval_result.accuracy,
            'precision': eval_result.precision,
            'recall': eval_result.recall,
            'f1_score': eval_result.f1_score,
            'evaluation_date': eval_result.evaluation_date.isoformat(),
            'dataset_size': eval_result.dataset_size
        } for eval_result in evaluations
    ]


@router.post("/model/compare")
async def compare_model_versions(
    model_versions: List[str] = Body(..., min_items=2, max_items=5),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Compare performance across different model versions"""
    
    evaluator = ModelEvaluator(db)
    comparison = evaluator.compare_models(model_versions)
    
    return comparison


# ========== Advanced Analytics Endpoints ==========

@router.get("/analytics/comprehensive")
async def perform_comprehensive_analytics(
    analysis_period_days: int = Query(30, ge=7, le=90),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Perform comprehensive advanced analytics including bias detection and retraining analysis"""
    
    analytics = AdvancedAnalytics(db)
    analysis = analytics.perform_comprehensive_analysis(analysis_period_days)
    
    return analysis


@router.get("/analytics/performance-trends")
async def analyze_performance_trends(
    days: int = Query(30, ge=7, le=90),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Analyze performance trends with alerting"""
    
    analytics = AdvancedAnalytics(db)
    trends = analytics.analyze_performance_trends(days)
    
    return trends


@router.get("/analytics/bias-detection")
async def detect_bias_patterns(
    days: int = Query(30, ge=7, le=90),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Detect bias patterns in the system"""
    
    analytics = AdvancedAnalytics(db)
    bias_analysis = analytics.detect_bias_patterns(days)
    
    return bias_analysis


@router.get("/analytics/data-drift")
async def detect_data_drift(
    days: int = Query(30, ge=7, le=90),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Detect data drift in the system"""
    
    analytics = AdvancedAnalytics(db)
    drift_analysis = analytics.detect_data_drift(days)
    
    return drift_analysis


@router.get("/analytics/retraining-triggers")
async def analyze_retraining_triggers(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Analyze conditions that might trigger model retraining"""
    
    analytics = AdvancedAnalytics(db)
    triggers = analytics.analyze_retraining_triggers()
    
    return triggers


@router.get("/analytics/system-health")
async def assess_system_health(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Comprehensive system health assessment"""
    
    analytics = AdvancedAnalytics(db)
    health = analytics.assess_system_health()
    
    return health
