"""
ML Monitoring Database Models
Models for tracking recognition performance, ground truth validation, and ML metrics
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base, generate_uuid
import enum


class ValidationStatus(str, enum.Enum):
    """Status of ground truth validation"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class RecognitionOutcome(str, enum.Enum):
    """Actual outcome of recognition attempt"""
    TRUE_POSITIVE = "true_positive"      # Correctly identified person
    TRUE_NEGATIVE = "true_negative"      # Correctly rejected unknown person
    FALSE_POSITIVE = "false_positive"    # Incorrectly identified wrong person
    FALSE_NEGATIVE = "false_negative"    # Failed to identify known person


class GroundTruthValidation(Base):
    """
    Ground truth validation for attendance recognition
    Tracks manual verification of recognition results
    """
    __tablename__ = "ground_truth_validations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference to original attendance log
    attendance_log_id = Column(UUID(as_uuid=True), ForeignKey("attendance.id"), nullable=False)
    
    # Recognition details
    predicted_employee_id = Column(String, nullable=True)  # Who system thought it was
    actual_employee_id = Column(String, nullable=True)     # Who it actually was
    confidence_score = Column(Float, nullable=False)
    
    # Validation details
    validation_status = Column(String, default=ValidationStatus.PENDING)
    recognition_outcome = Column(String, nullable=True)  # RecognitionOutcome enum
    
    # Validator information
    validated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    validated_at = Column(DateTime(timezone=True), nullable=True)
    validation_notes = Column(Text, nullable=True)
    
    # Quality metrics at time of recognition
    image_quality_score = Column(Float, nullable=True)
    face_area_ratio = Column(Float, nullable=True)
    brightness_score = Column(Float, nullable=True)
    blur_score = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional context
    environmental_factors = Column(JSON, nullable=True)  # lighting, angle, etc.
    
    # Relationships
    attendance_log = relationship("AttendanceLog", back_populates="ground_truth_validation")
    validator = relationship("User", foreign_keys=[validated_by])


class MLPerformanceMetrics(Base):
    """
    ML Performance tracking over time
    Aggregated metrics for model performance monitoring
    """
    __tablename__ = "ml_performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Time period for metrics
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    period_type = Column(String, nullable=False)  # 'hourly', 'daily', 'weekly'
    
    # Basic counts
    total_attempts = Column(Integer, default=0)
    successful_recognitions = Column(Integer, default=0)
    failed_recognitions = Column(Integer, default=0)
    
    # False positive/negative tracking
    true_positives = Column(Integer, default=0)
    true_negatives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    
    # Quality metrics
    avg_confidence_score = Column(Float, nullable=True)
    avg_image_quality = Column(Float, nullable=True)
    quality_rejections = Column(Integer, default=0)
    
    # Performance metrics (calculated)
    precision = Column(Float, nullable=True)  # TP / (TP + FP)
    recall = Column(Float, nullable=True)     # TP / (TP + FN)
    f1_score = Column(Float, nullable=True)   # 2 * (precision * recall) / (precision + recall)
    accuracy = Column(Float, nullable=True)   # (TP + TN) / (TP + TN + FP + FN)
    
    # Threshold performance
    current_threshold = Column(Float, nullable=True)
    high_confidence_threshold = Column(Float, nullable=True)
    low_confidence_threshold = Column(Float, nullable=True)
    
    # System performance
    avg_processing_time = Column(Float, nullable=True)  # milliseconds
    cache_hit_rate = Column(Float, nullable=True)
    
    # Anti-spoofing metrics
    spoofing_detections = Column(Integer, default=0)
    spoofing_false_alarms = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ThresholdExperiment(Base):
    """
    A/B testing for threshold optimization
    Tracks different threshold configurations and their performance
    """
    __tablename__ = "threshold_experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Experiment details
    experiment_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Threshold configuration
    recognition_threshold = Column(Float, nullable=False)
    high_confidence_threshold = Column(Float, nullable=False)
    low_confidence_threshold = Column(Float, nullable=False)
    
    # Experiment parameters
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    traffic_percentage = Column(Float, default=100.0)  # % of traffic using these thresholds
    
    # Results (populated during experiment)
    total_samples = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Performance metrics
    avg_confidence_score = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    false_negative_rate = Column(Float, nullable=True)
    
    # Business metrics
    user_satisfaction_score = Column(Float, nullable=True)
    manual_correction_rate = Column(Float, nullable=True)
    
    # Metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])


class ModelEvaluationResult(Base):
    """
    Comprehensive model evaluation results
    Stores detailed evaluation metrics for different model versions or configurations
    """
    __tablename__ = "model_evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model information
    model_version = Column(String, nullable=False)
    model_name = Column(String, nullable=False)  # e.g., "ArcFace", "FaceNet"
    evaluation_type = Column(String, nullable=False)  # e.g., "cross_validation", "holdout"
    
    # Dataset information
    dataset_size = Column(Integer, nullable=False)
    training_samples = Column(Integer, nullable=True)
    validation_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    
    # Performance metrics
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    
    # Detailed metrics
    true_positive_rate = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    true_negative_rate = Column(Float, nullable=True)
    false_negative_rate = Column(Float, nullable=True)
    
    # ROC metrics
    auc_score = Column(Float, nullable=True)
    optimal_threshold = Column(Float, nullable=True)
    
    # Confidence distribution
    avg_true_positive_confidence = Column(Float, nullable=True)
    avg_false_positive_confidence = Column(Float, nullable=True)
    confidence_std_dev = Column(Float, nullable=True)
    
    # Performance by demographic (if available)
    performance_by_group = Column(JSON, nullable=True)  # e.g., by age, gender, ethnicity
    
    # Evaluation conditions
    lighting_conditions = Column(JSON, nullable=True)
    image_quality_distribution = Column(JSON, nullable=True)
    pose_variations = Column(JSON, nullable=True)
    
    # Metadata
    evaluated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    evaluation_date = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)
    
    # Relationships
    evaluator = relationship("User", foreign_keys=[evaluated_by])


class DataQualityMetrics(Base):
    """
    Data quality assessment and trending
    Tracks quality of incoming images and training data
    """
    __tablename__ = "data_quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Time period
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    period_type = Column(String, nullable=False)  # 'hourly', 'daily', 'weekly'
    
    # Image quality metrics
    total_images_processed = Column(Integer, default=0)
    avg_image_quality_score = Column(Float, nullable=True)
    avg_face_area_ratio = Column(Float, nullable=True)
    avg_brightness = Column(Float, nullable=True)
    avg_blur_score = Column(Float, nullable=True)
    
    # Quality distribution
    high_quality_images = Column(Integer, default=0)     # > 0.8 quality
    medium_quality_images = Column(Integer, default=0)   # 0.5-0.8 quality
    low_quality_images = Column(Integer, default=0)      # < 0.5 quality
    
    # Issue tracking
    too_dark_images = Column(Integer, default=0)
    too_bright_images = Column(Integer, default=0)
    blurry_images = Column(Integer, default=0)
    face_too_small = Column(Integer, default=0)
    multiple_faces = Column(Integer, default=0)
    no_face_detected = Column(Integer, default=0)
    
    # Environmental factors
    lighting_distribution = Column(JSON, nullable=True)  # distribution of lighting conditions
    pose_distribution = Column(JSON, nullable=True)      # distribution of face poses
    
    # Training data quality (for enrollment)
    enrollment_quality_avg = Column(Float, nullable=True)
    enrollment_diversity_score = Column(Float, nullable=True)  # pose/lighting diversity
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
