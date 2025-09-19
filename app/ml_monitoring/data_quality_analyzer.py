"""
Advanced Data Quality Analyzer
Comprehensive analysis of image quality, training data diversity, and environmental factors
"""

import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
import json
import os
from collections import Counter, defaultdict
import statistics

from app.ml_monitoring.models import DataQualityMetrics, GroundTruthValidation
from app.employees.models import Employee, TrainingPhoto
from app.attendance.models import AttendanceLog
from app.face_recognition.face_quality_utils import FaceQualityValidator
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Advanced service for comprehensive data quality analysis and monitoring"""
    
    def __init__(self, db: Session):
        self.db = db
        self.quality_validator = FaceQualityValidator()
        self.quality_validator_strict = FaceQualityValidator(strict_mode=True)
    
    def analyze_training_data_quality(self, employee_id: Optional[str] = None) -> Dict:
        """Comprehensive analysis of training data quality and diversity"""
        
        logger.info(f"Starting training data quality analysis for employee: {employee_id or 'all'}")
        
        # Get training photos
        query = self.db.query(TrainingPhoto)
        if employee_id:
            query = query.filter(TrainingPhoto.employee_id == employee_id)
        
        training_photos = query.all()
        
        if not training_photos:
            return {
                'status': 'no_data',
                'message': 'No training photos found',
                'employee_id': employee_id
            }
        
        # Analyze each photo
        quality_scores = []
        diversity_metrics = {
            'pose_variations': [],
            'lighting_conditions': [],
            'expressions': [],
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'technical_issues': defaultdict(int)
        }
        
        employee_analysis = defaultdict(lambda: {
            'photo_count': 0,
            'avg_quality': 0,
            'diversity_score': 0,
            'issues': []
        })
        
        for photo in training_photos:
            try:
                analysis = self._analyze_single_training_photo(photo)
                quality_scores.append(analysis['quality_score'])
                
                # Update diversity metrics
                self._update_diversity_metrics(diversity_metrics, analysis)
                
                # Update per-employee analysis
                emp_data = employee_analysis[photo.employee_id]
                emp_data['photo_count'] += 1
                emp_data['avg_quality'] = (emp_data['avg_quality'] * (emp_data['photo_count'] - 1) + analysis['quality_score']) / emp_data['photo_count']
                emp_data['issues'].extend(analysis['issues'])
                
            except Exception as e:
                logger.error(f"Error analyzing training photo {photo.id}: {str(e)}")
                diversity_metrics['technical_issues']['analysis_failed'] += 1
        
        # Calculate overall metrics
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0
        quality_std = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
        
        # Calculate diversity scores
        diversity_score = self._calculate_diversity_score(diversity_metrics)
        
        # Generate recommendations
        recommendations = self._generate_training_data_recommendations(
            diversity_metrics, overall_quality, quality_std, employee_analysis
        )
        
        return {
            'status': 'success',
            'analysis_date': datetime.utcnow().isoformat(),
            'employee_id': employee_id,
            'total_photos': len(training_photos),
            'overall_metrics': {
                'average_quality': overall_quality,
                'quality_std_dev': quality_std,
                'diversity_score': diversity_score,
                'completeness_score': self._calculate_completeness_score(diversity_metrics)
            },
            'quality_distribution': diversity_metrics['quality_distribution'],
            'diversity_analysis': {
                'pose_variations': len(set(diversity_metrics['pose_variations'])),
                'lighting_conditions': len(set(diversity_metrics['lighting_conditions'])),
                'expression_variety': len(set(diversity_metrics['expressions'])),
                'technical_issues': dict(diversity_metrics['technical_issues'])
            },
            'employee_breakdown': dict(employee_analysis) if not employee_id else None,
            'recommendations': recommendations
        }
    
    def _analyze_single_training_photo(self, photo: TrainingPhoto) -> Dict:
        """Analyze a single training photo for quality and characteristics"""
        
        if not os.path.exists(photo.file_path):
            return {
                'quality_score': 0,
                'pose': 'unknown',
                'lighting': 'unknown',
                'expression': 'unknown',
                'issues': ['file_not_found']
            }
        
        # Load and analyze image
        image = cv2.imread(photo.file_path)
        if image is None:
            return {
                'quality_score': 0,
                'pose': 'unknown',
                'lighting': 'unknown',
                'expression': 'unknown',
                'issues': ['invalid_image']
            }
        
        # Quality analysis
        quality_result = self.quality_validator.validate_face_quality_from_array(image)
        
        # Pose estimation
        pose = self._estimate_pose(image)
        
        # Lighting analysis
        lighting = self._analyze_lighting_conditions(image)
        
        # Expression analysis (basic)
        expression = self._analyze_expression(image)
        
        # Technical issues
        issues = []
        if quality_result.confidence < 0.5:
            issues.append('low_quality')
        if not quality_result.face_detected:
            issues.append('no_face_detected')
        if quality_result.hands_detected:
            issues.append('hands_detected')
        if not quality_result.brightness_ok:
            issues.append('poor_lighting')
        if not quality_result.blur_ok:
            issues.append('blurry')
        
        return {
            'quality_score': quality_result.confidence,
            'pose': pose,
            'lighting': lighting,
            'expression': expression,
            'issues': issues,
            'face_area_ratio': quality_result.face_area_ratio,
            'brightness_ok': quality_result.brightness_ok,
            'blur_ok': quality_result.blur_ok
        }
    
    def _estimate_pose(self, image: np.ndarray) -> str:
        """Estimate face pose from image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                return 'no_face'
            
            # Get the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Simple pose estimation based on face symmetry and eye detection
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(face_roi)
            
            if len(eyes) >= 2:
                # Sort eyes by x coordinate
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye, right_eye = eyes[0], eyes[1]
                
                # Calculate eye positions relative to face center
                face_center_x = w // 2
                left_eye_center = left_eye[0] + left_eye[2] // 2
                right_eye_center = right_eye[0] + right_eye[2] // 2
                
                # Estimate pose based on eye positions
                eye_distance = abs(right_eye_center - left_eye_center)
                left_offset = abs(left_eye_center - face_center_x * 0.3)
                right_offset = abs(right_eye_center - face_center_x * 1.7)
                
                if eye_distance < w * 0.2:  # Eyes too close - profile view
                    return 'profile'
                elif left_offset > w * 0.15:  # Left eye too far right
                    return 'left_turn'
                elif right_offset > w * 0.15:  # Right eye too far left
                    return 'right_turn'
                else:
                    return 'frontal'
            else:
                # Fallback: analyze face width vs height ratio
                aspect_ratio = w / h
                if aspect_ratio < 0.7:
                    return 'profile'
                else:
                    return 'frontal'
                    
        except Exception as e:
            logger.warning(f"Pose estimation failed: {str(e)}")
            return 'unknown'
    
    def _analyze_lighting_conditions(self, image: np.ndarray) -> str:
        """Analyze lighting conditions in the image"""
        try:
            # Convert to LAB color space for better lighting analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Calculate lighting metrics
            mean_brightness = np.mean(l_channel)
            brightness_std = np.std(l_channel)
            
            # Analyze histogram
            hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
            
            # Find peaks in histogram
            peak_indices = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                    peak_indices.append(i)
            
            # Classify lighting conditions
            if mean_brightness < 50:
                return 'low_light'
            elif mean_brightness > 200:
                return 'overexposed'
            elif brightness_std < 20:
                return 'uniform'  # Very even lighting
            elif len(peak_indices) > 2:
                return 'mixed'    # Multiple light sources
            elif brightness_std > 50:
                return 'harsh'    # High contrast lighting
            else:
                return 'natural'  # Good natural lighting
                
        except Exception as e:
            logger.warning(f"Lighting analysis failed: {str(e)}")
            return 'unknown'
    
    def _analyze_expression(self, image: np.ndarray) -> str:
        """Basic expression analysis"""
        try:
            # This is a simplified approach - in production you'd use a proper expression classifier
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face and mouth
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                return 'unknown'
            
            # Get face region
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Look for smile using mouth detection
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.8, minNeighbors=20)
            
            if len(smiles) > 0:
                return 'smile'
            else:
                # Simple analysis based on mouth region intensity
                mouth_region = face_roi[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
                mouth_variance = np.var(mouth_region)
                
                if mouth_variance > 100:  # High variance might indicate open mouth/expression
                    return 'expressive'
                else:
                    return 'neutral'
                    
        except Exception as e:
            logger.warning(f"Expression analysis failed: {str(e)}")
            return 'unknown'
    
    def _update_diversity_metrics(self, diversity_metrics: Dict, analysis: Dict):
        """Update diversity metrics with analysis results"""
        diversity_metrics['pose_variations'].append(analysis['pose'])
        diversity_metrics['lighting_conditions'].append(analysis['lighting'])
        diversity_metrics['expressions'].append(analysis['expression'])
        
        # Update quality distribution
        quality = analysis['quality_score']
        if quality >= 0.8:
            diversity_metrics['quality_distribution']['high'] += 1
        elif quality >= 0.5:
            diversity_metrics['quality_distribution']['medium'] += 1
        else:
            diversity_metrics['quality_distribution']['low'] += 1
        
        # Update technical issues
        for issue in analysis['issues']:
            diversity_metrics['technical_issues'][issue] += 1
    
    def _calculate_diversity_score(self, diversity_metrics: Dict) -> float:
        """Calculate overall diversity score (0-1)"""
        try:
            # Pose diversity (0-0.4)
            unique_poses = len(set(diversity_metrics['pose_variations']))
            pose_score = min(unique_poses / 4, 1.0) * 0.4  # Max 4 poses: frontal, left, right, profile
            
            # Lighting diversity (0-0.3)
            unique_lighting = len(set(diversity_metrics['lighting_conditions']))
            lighting_score = min(unique_lighting / 5, 1.0) * 0.3  # Max 5 lighting conditions
            
            # Expression diversity (0-0.2)
            unique_expressions = len(set(diversity_metrics['expressions']))
            expression_score = min(unique_expressions / 3, 1.0) * 0.2  # Max 3 expressions
            
            # Quality consistency (0-0.1)
            total_photos = sum(diversity_metrics['quality_distribution'].values())
            high_quality_ratio = diversity_metrics['quality_distribution']['high'] / max(total_photos, 1)
            quality_score = high_quality_ratio * 0.1
            
            return pose_score + lighting_score + expression_score + quality_score
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {str(e)}")
            return 0.0
    
    def _calculate_completeness_score(self, diversity_metrics: Dict) -> float:
        """Calculate training data completeness score"""
        try:
            total_photos = sum(diversity_metrics['quality_distribution'].values())
            
            # Minimum requirements
            min_photos = settings.training_photos_required
            min_poses = 2  # At least frontal and one angle
            min_lighting = 2  # At least 2 lighting conditions
            
            # Calculate scores
            photo_score = min(total_photos / min_photos, 1.0) * 0.5
            pose_score = min(len(set(diversity_metrics['pose_variations'])) / min_poses, 1.0) * 0.3
            lighting_score = min(len(set(diversity_metrics['lighting_conditions'])) / min_lighting, 1.0) * 0.2
            
            return photo_score + pose_score + lighting_score
            
        except Exception as e:
            logger.error(f"Error calculating completeness score: {str(e)}")
            return 0.0
    
    def _generate_training_data_recommendations(
        self, 
        diversity_metrics: Dict, 
        overall_quality: float, 
        quality_std: float,
        employee_analysis: Dict
    ) -> List[Dict]:
        """Generate actionable recommendations for improving training data"""
        
        recommendations = []
        
        # Quality recommendations
        if overall_quality < 0.7:
            recommendations.append({
                'type': 'quality_improvement',
                'priority': 'high',
                'title': 'Improve Image Quality',
                'description': f'Average quality is {overall_quality:.2f}. Aim for >0.7',
                'actions': [
                    'Use better lighting conditions',
                    'Ensure faces are clearly visible',
                    'Reduce blur by using stable camera',
                    'Remove low-quality images and retake'
                ]
            })
        
        # Diversity recommendations
        unique_poses = len(set(diversity_metrics['pose_variations']))
        if unique_poses < 3:
            recommendations.append({
                'type': 'pose_diversity',
                'priority': 'medium',
                'title': 'Add More Pose Variations',
                'description': f'Only {unique_poses} different poses detected',
                'actions': [
                    'Capture frontal face images',
                    'Include slight left and right turns',
                    'Add some profile shots if needed',
                    'Ensure variety in head positions'
                ]
            })
        
        # Lighting recommendations
        unique_lighting = len(set(diversity_metrics['lighting_conditions']))
        if unique_lighting < 3:
            recommendations.append({
                'type': 'lighting_diversity',
                'priority': 'medium',
                'title': 'Improve Lighting Diversity',
                'description': f'Only {unique_lighting} lighting conditions detected',
                'actions': [
                    'Capture images in different lighting',
                    'Include natural and artificial light',
                    'Avoid harsh shadows',
                    'Test different times of day'
                ]
            })
        
        # Technical issue recommendations
        if diversity_metrics['technical_issues']:
            top_issues = sorted(
                diversity_metrics['technical_issues'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            recommendations.append({
                'type': 'technical_issues',
                'priority': 'high',
                'title': 'Fix Technical Issues',
                'description': f'Found {sum(diversity_metrics["technical_issues"].values())} technical issues',
                'actions': [f'Address {issue}: {count} occurrences' for issue, count in top_issues]
            })
        
        # Employee-specific recommendations
        if employee_analysis:
            low_quality_employees = [
                emp_id for emp_id, data in employee_analysis.items() 
                if data['avg_quality'] < 0.6
            ]
            
            if low_quality_employees:
                recommendations.append({
                    'type': 'employee_specific',
                    'priority': 'medium',
                    'title': 'Re-enroll Low Quality Employees',
                    'description': f'{len(low_quality_employees)} employees need better training data',
                    'actions': [f'Re-enroll employee {emp_id}' for emp_id in low_quality_employees[:5]]
                })
        
        return recommendations
    
    def analyze_real_time_quality_trends(self, days: int = 7) -> Dict:
        """Analyze real-time image quality trends from recent attendance"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent attendance logs with images
        recent_logs = self.db.query(AttendanceLog).filter(
            and_(
                AttendanceLog.timestamp >= cutoff_date,
                AttendanceLog.image_path.isnot(None)
            )
        ).order_by(desc(AttendanceLog.timestamp)).limit(1000).all()
        
        if not recent_logs:
            return {
                'status': 'no_data',
                'message': 'No recent attendance logs with images found'
            }
        
        # Analyze quality trends
        daily_quality = defaultdict(list)
        hourly_quality = defaultdict(list)
        quality_issues = defaultdict(int)
        
        for log in recent_logs:
            try:
                if not os.path.exists(log.image_path):
                    quality_issues['missing_files'] += 1
                    continue
                
                # Analyze image quality
                quality_result = self.quality_validator.validate_face_quality(log.image_path)
                
                # Group by date and hour
                log_date = log.timestamp.date()
                log_hour = log.timestamp.hour
                
                daily_quality[log_date.isoformat()].append(quality_result.confidence)
                hourly_quality[log_hour].append(quality_result.confidence)
                
                # Track issues
                if not quality_result.is_clear:
                    for issue in quality_result.issues:
                        quality_issues[issue] += 1
                
            except Exception as e:
                logger.error(f"Error analyzing attendance image {log.image_path}: {str(e)}")
                quality_issues['analysis_failed'] += 1
        
        # Calculate trends
        daily_averages = {
            date: statistics.mean(qualities) 
            for date, qualities in daily_quality.items()
        }
        
        hourly_averages = {
            hour: statistics.mean(qualities) 
            for hour, qualities in hourly_quality.items()
        }
        
        # Detect quality degradation
        quality_trend = self._detect_quality_trend(list(daily_averages.values()))
        
        # Generate alerts
        alerts = self._generate_quality_alerts(daily_averages, hourly_averages, quality_issues)
        
        return {
            'status': 'success',
            'analysis_period': f'{days} days',
            'total_images_analyzed': len(recent_logs),
            'daily_quality_trends': daily_averages,
            'hourly_quality_patterns': hourly_averages,
            'quality_trend': quality_trend,
            'common_issues': dict(quality_issues),
            'alerts': alerts,
            'recommendations': self._generate_realtime_quality_recommendations(
                hourly_averages, quality_issues
            )
        }
    
    def _detect_quality_trend(self, quality_values: List[float]) -> Dict:
        """Detect if quality is improving, degrading, or stable"""
        if len(quality_values) < 3:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        # Simple linear regression to detect trend
        x = list(range(len(quality_values)))
        n = len(quality_values)
        
        sum_x = sum(x)
        sum_y = sum(quality_values)
        sum_xy = sum(x[i] * quality_values[i] for i in range(n))
        sum_x2 = sum(x_val ** 2 for x_val in x)
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'degrading'
        
        # Calculate confidence based on R-squared
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in quality_values)
        ss_res = sum((quality_values[i] - (slope * x[i] + (sum_y - slope * sum_x) / n)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'trend': trend,
            'slope': slope,
            'confidence': r_squared,
            'significance': 'high' if r_squared > 0.7 else 'medium' if r_squared > 0.4 else 'low'
        }
    
    def _generate_quality_alerts(
        self, 
        daily_averages: Dict, 
        hourly_averages: Dict, 
        quality_issues: Dict
    ) -> List[Dict]:
        """Generate quality alerts based on analysis"""
        alerts = []
        
        # Check for recent quality degradation
        recent_days = sorted(daily_averages.keys())[-3:]  # Last 3 days
        if len(recent_days) >= 2:
            recent_avg = statistics.mean([daily_averages[day] for day in recent_days])
            if recent_avg < 0.6:
                alerts.append({
                    'type': 'quality_degradation',
                    'severity': 'high',
                    'message': f'Image quality has dropped to {recent_avg:.2f} in recent days',
                    'recommendation': 'Check camera settings and lighting conditions'
                })
        
        # Check for problematic hours
        low_quality_hours = [
            hour for hour, avg_quality in hourly_averages.items() 
            if avg_quality < 0.5
        ]
        if low_quality_hours:
            alerts.append({
                'type': 'time_based_issues',
                'severity': 'medium',
                'message': f'Poor quality during hours: {low_quality_hours}',
                'recommendation': 'Check lighting conditions during these hours'
            })
        
        # Check for high issue rates
        total_images = sum(quality_issues.values()) if quality_issues else 1
        for issue, count in quality_issues.items():
            if count / total_images > 0.2:  # More than 20% of images have this issue
                alerts.append({
                    'type': 'high_issue_rate',
                    'severity': 'medium',
                    'message': f'High rate of {issue}: {count} ({count/total_images*100:.1f}%)',
                    'recommendation': f'Address root cause of {issue}'
                })
        
        return alerts
    
    def _generate_realtime_quality_recommendations(
        self, 
        hourly_averages: Dict, 
        quality_issues: Dict
    ) -> List[Dict]:
        """Generate recommendations for real-time quality improvement"""
        recommendations = []
        
        # Time-based recommendations
        if hourly_averages:
            best_hours = [h for h, q in hourly_averages.items() if q > 0.8]
            worst_hours = [h for h, q in hourly_averages.items() if q < 0.5]
            
            if best_hours and worst_hours:
                recommendations.append({
                    'type': 'timing_optimization',
                    'title': 'Optimize Attendance Times',
                    'description': f'Quality is best during hours {best_hours}, worst during {worst_hours}',
                    'actions': [
                        'Encourage attendance during high-quality hours',
                        'Improve lighting during problematic hours',
                        'Consider camera adjustments for different times'
                    ]
                })
        
        # Issue-specific recommendations
        if quality_issues:
            top_issue = max(quality_issues.items(), key=lambda x: x[1])
            issue_name, issue_count = top_issue
            
            issue_recommendations = {
                'blurry': ['Use image stabilization', 'Ensure users stay still', 'Check camera focus'],
                'too_dark': ['Improve lighting', 'Add supplementary lights', 'Adjust camera settings'],
                'too_bright': ['Reduce direct lighting', 'Use diffused light', 'Adjust exposure'],
                'no_face_detected': ['Improve user guidance', 'Adjust camera angle', 'Add face detection feedback'],
                'face_too_small': ['Adjust camera distance', 'Guide users to proper position', 'Use zoom if available']
            }
            
            if issue_name in issue_recommendations:
                recommendations.append({
                    'type': 'issue_specific',
                    'title': f'Fix {issue_name.replace("_", " ").title()}',
                    'description': f'Most common issue: {issue_name} ({issue_count} occurrences)',
                    'actions': issue_recommendations[issue_name]
                })
        
        return recommendations
    
    def generate_quality_report(self, days: int = 30) -> Dict:
        """Generate comprehensive data quality report"""
        
        logger.info(f"Generating comprehensive quality report for {days} days")
        
        # Analyze training data quality
        training_analysis = self.analyze_training_data_quality()
        
        # Analyze real-time quality trends
        realtime_analysis = self.analyze_real_time_quality_trends(days)
        
        # Get historical quality metrics from database
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        historical_metrics = self.db.query(DataQualityMetrics).filter(
            DataQualityMetrics.date >= cutoff_date
        ).order_by(DataQualityMetrics.date).all()
        
        # Calculate overall scores
        overall_scores = self._calculate_overall_quality_scores(
            training_analysis, realtime_analysis, historical_metrics
        )
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            overall_scores, training_analysis, realtime_analysis
        )
        
        return {
            'report_metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'analysis_period_days': days,
                'report_type': 'comprehensive_quality_analysis'
            },
            'executive_summary': executive_summary,
            'overall_scores': overall_scores,
            'training_data_analysis': training_analysis,
            'realtime_quality_analysis': realtime_analysis,
            'historical_trends': [
                {
                    'date': metric.date.isoformat(),
                    'avg_quality': metric.avg_image_quality_score,
                    'total_images': metric.total_images_processed,
                    'high_quality_ratio': metric.high_quality_images / max(metric.total_images_processed, 1)
                } for metric in historical_metrics
            ],
            'recommendations': self._consolidate_recommendations(
                training_analysis.get('recommendations', []),
                realtime_analysis.get('recommendations', [])
            )
        }
    
    def _calculate_overall_quality_scores(
        self, 
        training_analysis: Dict, 
        realtime_analysis: Dict, 
        historical_metrics: List
    ) -> Dict:
        """Calculate overall quality scores across all dimensions"""
        
        scores = {
            'training_data_quality': 0.0,
            'realtime_quality': 0.0,
            'quality_consistency': 0.0,
            'data_completeness': 0.0,
            'overall_score': 0.0
        }
        
        # Training data quality score
        if training_analysis.get('status') == 'success':
            scores['training_data_quality'] = training_analysis['overall_metrics']['average_quality']
            scores['data_completeness'] = training_analysis['overall_metrics']['completeness_score']
        
        # Real-time quality score
        if realtime_analysis.get('status') == 'success':
            daily_qualities = list(realtime_analysis['daily_quality_trends'].values())
            if daily_qualities:
                scores['realtime_quality'] = statistics.mean(daily_qualities)
                scores['quality_consistency'] = 1.0 - (statistics.stdev(daily_qualities) if len(daily_qualities) > 1 else 0)
        
        # Overall score (weighted average)
        weights = {
            'training_data_quality': 0.3,
            'realtime_quality': 0.4,
            'quality_consistency': 0.2,
            'data_completeness': 0.1
        }
        
        scores['overall_score'] = sum(
            scores[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return scores
    
    def _generate_executive_summary(
        self, 
        overall_scores: Dict, 
        training_analysis: Dict, 
        realtime_analysis: Dict
    ) -> Dict:
        """Generate executive summary of quality analysis"""
        
        # Determine overall health
        overall_score = overall_scores['overall_score']
        if overall_score >= 0.8:
            health_status = 'excellent'
            health_color = 'green'
        elif overall_score >= 0.6:
            health_status = 'good'
            health_color = 'yellow'
        else:
            health_status = 'needs_attention'
            health_color = 'red'
        
        # Count issues and recommendations
        total_recommendations = 0
        critical_issues = 0
        
        if training_analysis.get('recommendations'):
            total_recommendations += len(training_analysis['recommendations'])
            critical_issues += len([r for r in training_analysis['recommendations'] if r.get('priority') == 'high'])
        
        if realtime_analysis.get('recommendations'):
            total_recommendations += len(realtime_analysis['recommendations'])
        
        if realtime_analysis.get('alerts'):
            critical_issues += len([a for a in realtime_analysis['alerts'] if a.get('severity') == 'high'])
        
        return {
            'overall_health': {
                'status': health_status,
                'score': overall_score,
                'color': health_color
            },
            'key_metrics': {
                'training_quality': f"{overall_scores['training_data_quality']:.2f}",
                'realtime_quality': f"{overall_scores['realtime_quality']:.2f}",
                'consistency': f"{overall_scores['quality_consistency']:.2f}",
                'completeness': f"{overall_scores['data_completeness']:.2f}"
            },
            'issue_summary': {
                'total_recommendations': total_recommendations,
                'critical_issues': critical_issues,
                'status': 'attention_required' if critical_issues > 0 else 'good'
            },
            'top_priorities': self._extract_top_priorities(training_analysis, realtime_analysis)
        }
    
    def _extract_top_priorities(self, training_analysis: Dict, realtime_analysis: Dict) -> List[str]:
        """Extract top 3 priority actions"""
        priorities = []
        
        # From training analysis
        if training_analysis.get('recommendations'):
            high_priority = [r for r in training_analysis['recommendations'] if r.get('priority') == 'high']
            priorities.extend([r['title'] for r in high_priority[:2]])
        
        # From realtime analysis
        if realtime_analysis.get('alerts'):
            high_severity = [a for a in realtime_analysis['alerts'] if a.get('severity') == 'high']
            priorities.extend([a['message'] for a in high_severity[:1]])
        
        return priorities[:3]
    
    def _consolidate_recommendations(self, training_recs: List, realtime_recs: List) -> List[Dict]:
        """Consolidate and prioritize all recommendations"""
        
        all_recommendations = []
        
        # Add training recommendations with source
        for rec in training_recs:
            rec['source'] = 'training_data'
            all_recommendations.append(rec)
        
        # Add realtime recommendations with source
        for rec in realtime_recs:
            rec['source'] = 'realtime_analysis'
            if 'priority' not in rec:
                rec['priority'] = 'medium'
            all_recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
        
        return all_recommendations



