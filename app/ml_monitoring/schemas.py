"""
ML Monitoring Pydantic Schemas
Request and response models for ML monitoring API endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from app.ml_monitoring.models import ValidationStatus, RecognitionOutcome


# ========== Ground Truth Validation Schemas ==========

class ValidationRequest(BaseModel):
    """Request schema for validating a recognition result"""
    actual_employee_id: Optional[str] = Field(None, description="Actual employee ID (None if unknown person)")
    validation_status: ValidationStatus = Field(..., description="Validation status")
    notes: Optional[str] = Field(None, max_length=500, description="Optional validation notes")


class GroundTruthValidationResponse(BaseModel):
    """Response schema for ground truth validation records"""
    id: int
    attendance_log_id: int
    predicted_employee_id: Optional[str]
    actual_employee_id: Optional[str]
    confidence_score: float
    validation_status: str
    recognition_outcome: Optional[str]
    validated_by: Optional[int]
    validated_at: Optional[datetime]
    validation_notes: Optional[str]
    
    # Quality metrics
    image_quality_score: Optional[float]
    face_area_ratio: Optional[float]
    brightness_score: Optional[float]
    blur_score: Optional[float]
    
    # Metadata
    created_at: datetime
    updated_at: Optional[datetime]
    environmental_factors: Optional[Dict[str, Any]]
    
    class Config:
        from_attributes = True


# ========== Performance Metrics Schemas ==========

class PerformanceMetricsResponse(BaseModel):
    """Response schema for performance metrics"""
    date: datetime
    period_type: str
    
    # Basic counts
    total_attempts: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    
    # ML metrics
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Calculated metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    
    # Quality and performance
    avg_confidence_score: Optional[float] = None
    avg_image_quality: Optional[float] = None
    avg_processing_time: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    
    class Config:
        from_attributes = True


class CurrentStatsResponse(BaseModel):
    """Response schema for current session statistics"""
    total_attempts: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    retry_attempts: int = 0
    quality_rejections: int = 0
    spoofing_detections: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    manual_corrections: int = 0
    
    # Calculated rates
    success_rate: float = 0.0
    retry_rate: float = 0.0
    quality_rejection_rate: float = 0.0
    spoofing_detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    manual_correction_rate: float = 0.0
    system_accuracy: float = 0.0


# ========== Data Quality Schemas ==========

class DataQualityMetricsResponse(BaseModel):
    """Response schema for data quality metrics"""
    date: datetime
    period_type: str
    
    # Image processing stats
    total_images_processed: int = 0
    avg_image_quality_score: Optional[float] = None
    avg_face_area_ratio: Optional[float] = None
    avg_brightness: Optional[float] = None
    avg_blur_score: Optional[float] = None
    
    # Quality distribution
    high_quality_images: int = 0
    medium_quality_images: int = 0
    low_quality_images: int = 0
    
    # Issue tracking
    too_dark_images: int = 0
    too_bright_images: int = 0
    blurry_images: int = 0
    face_too_small: int = 0
    multiple_faces: int = 0
    no_face_detected: int = 0
    
    class Config:
        from_attributes = True


class QualityTrendResponse(BaseModel):
    """Response schema for quality trend data"""
    date: str
    avg_quality_score: Optional[float]
    total_images: int
    high_quality_ratio: float
    quality_issues: Dict[str, int]


# ========== Threshold Optimization Schemas ==========

class ThresholdExperimentRequest(BaseModel):
    """Request schema for creating threshold experiments"""
    experiment_name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    
    recognition_threshold: float = Field(..., ge=0.1, le=0.9)
    high_confidence_threshold: float = Field(..., ge=0.1, le=0.9)
    low_confidence_threshold: float = Field(..., ge=0.1, le=0.9)
    
    traffic_percentage: float = Field(10.0, ge=1.0, le=100.0)
    
    @validator('high_confidence_threshold')
    def high_threshold_must_be_higher(cls, v, values):
        if 'recognition_threshold' in values and v <= values['recognition_threshold']:
            raise ValueError('high_confidence_threshold must be higher than recognition_threshold')
        return v
    
    @validator('low_confidence_threshold')
    def low_threshold_must_be_lower(cls, v, values):
        if 'recognition_threshold' in values and v >= values['recognition_threshold']:
            raise ValueError('low_confidence_threshold must be lower than recognition_threshold')
        return v


class ThresholdExperimentResponse(BaseModel):
    """Response schema for threshold experiments"""
    id: int
    experiment_name: str
    description: Optional[str]
    
    # Threshold configuration
    recognition_threshold: float
    high_confidence_threshold: float
    low_confidence_threshold: float
    
    # Experiment details
    start_date: datetime
    end_date: Optional[datetime]
    is_active: bool
    traffic_percentage: float
    
    # Results
    total_samples: int = 0
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Metadata
    created_by: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class ThresholdOptimizationResponse(BaseModel):
    """Response schema for threshold optimization recommendations"""
    current_thresholds: Dict[str, float]
    recommended_thresholds: Dict[str, float]
    performance_at_optimal: Dict[str, float]
    sample_size: int
    roc_curve_data: List[Dict[str, float]]


# ========== Dashboard Schemas ==========

class MLDashboardResponse(BaseModel):
    """Response schema for ML dashboard data"""
    validation_stats: Dict[str, Any]
    performance_trends: List[Dict[str, Any]]
    quality_trends: List[Dict[str, Any]]
    pending_validations: int
    threshold_analysis: Dict[str, Any]
    system_health: Dict[str, Any]
    current_session: CurrentStatsResponse


# ========== Manual Correction Schemas ==========

class ManualCorrectionRequest(BaseModel):
    """Request schema for recording manual corrections"""
    was_false_positive: bool = Field(False, description="Whether this was a false positive")
    was_false_negative: bool = Field(False, description="Whether this was a false negative")
    notes: Optional[str] = Field(None, max_length=500, description="Optional correction notes")
    
    @validator('was_false_negative')
    def only_one_correction_type(cls, v, values):
        if v and values.get('was_false_positive'):
            raise ValueError('Cannot be both false positive and false negative')
        return v


# ========== Model Evaluation Schemas ==========

class ModelEvaluationRequest(BaseModel):
    """Request schema for model evaluation"""
    model_name: str = Field(..., min_length=2, max_length=50)
    model_version: str = Field(..., min_length=1, max_length=20)
    evaluation_type: str = Field(..., pattern="^(cross_validation|holdout|temporal_split)$")
    
    dataset_size: int = Field(..., ge=10)
    training_samples: Optional[int] = Field(None, ge=1)
    validation_samples: Optional[int] = Field(None, ge=1)
    test_samples: Optional[int] = Field(None, ge=1)
    
    notes: Optional[str] = Field(None, max_length=1000)


class ModelEvaluationResponse(BaseModel):
    """Response schema for model evaluation results"""
    id: int
    model_version: str
    model_name: str
    evaluation_type: str
    
    # Dataset info
    dataset_size: int
    training_samples: Optional[int]
    validation_samples: Optional[int]
    test_samples: Optional[int]
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Detailed metrics
    true_positive_rate: Optional[float]
    false_positive_rate: Optional[float]
    true_negative_rate: Optional[float]
    false_negative_rate: Optional[float]
    
    # ROC metrics
    auc_score: Optional[float]
    optimal_threshold: Optional[float]
    
    # Confidence analysis
    avg_true_positive_confidence: Optional[float]
    avg_false_positive_confidence: Optional[float]
    confidence_std_dev: Optional[float]
    
    # Evaluation metadata
    evaluated_by: int
    evaluation_date: datetime
    notes: Optional[str]
    
    class Config:
        from_attributes = True


# ========== Report Schemas ==========

class FalsePositiveAnalysisResponse(BaseModel):
    """Response schema for false positive analysis"""
    period_days: int
    total_false_positives: int
    analysis: Dict[str, Any]
    recommendations: List[str]


class ModelPerformanceReportResponse(BaseModel):
    """Response schema for comprehensive model performance report"""
    report_period: Dict[str, Any]
    summary: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    model_info: Dict[str, Any]


# ========== Validation Enums for API ==========

class ValidationStatusEnum(str, Enum):
    """Enum for validation status options"""
    pending = "pending"
    confirmed = "confirmed"
    rejected = "rejected"
    uncertain = "uncertain"


class RecognitionOutcomeEnum(str, Enum):
    """Enum for recognition outcome options"""
    true_positive = "true_positive"
    true_negative = "true_negative"
    false_positive = "false_positive"
    false_negative = "false_negative"
