"""
Cached Face Recognition Service
Enhanced face recognition with Redis caching for improved performance
"""

import numpy as np
import hashlib
import json
import time
from typing import Dict, List, Optional
from app.core.redis_service import RedisCacheService
from app.face_recognition.deepface_service import deepface_service
from app.core.config import settings
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CachedFaceRecognitionService:
    """Face recognition service with Redis caching"""
    
    def __init__(self, redis_service: RedisCacheService, db_session=None):
        self.redis = redis_service
        self.db = db_session
        self.similarity_threshold = settings.recognition_threshold
        self.high_confidence_threshold = settings.high_confidence_threshold
        
        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.recognition_times = []
    
    def recognize_face_cached(self, image_path: str) -> Dict:
        """Face recognition with Redis caching"""
        self.total_requests += 1
        start_time = self._get_timestamp()
        
        logger.info(f"🎯 REDIS CACHE - Face Recognition: Starting recognition request #{self.total_requests}")
        
        try:
            # Step 1: Check if we have a cached result for this image
            image_hash = self._calculate_image_hash(image_path)
            logger.info(f"🔍 REDIS CACHE - Face Recognition: Checking for cached result (hash: {image_hash[:8]}...)")
            
            cached_result = self.redis.get_cached_recognition_result(image_hash)
            
            if cached_result:
                self.cache_hits += 1
                processing_time = self._get_timestamp() - start_time
                logger.info(f"🟢 REDIS CACHE - Face Recognition: Using cached recognition result in {processing_time:.3f}s (Hit #{self.cache_hits})")
                cached_result['cache_hit'] = True
                return cached_result
            
            logger.info(f"🟡 REDIS CACHE - Face Recognition: No cached result found, proceeding with recognition")
            
            # Step 2: Extract embedding from input image
            embedding_start = time.time()
            input_embedding = deepface_service.extract_embedding(image_path)
            embedding_time = time.time() - embedding_start
            
            if input_embedding is None:
                logger.warning(f"🔴 REDIS CACHE - Face Recognition: No face detected in image after {embedding_time:.3f}s")
                result = {
                    'success': False,
                    'error': 'No face detected in image',
                    'confidence': 0.0,
                    'cache_hit': False,
                    'processing_time': self._get_timestamp() - start_time
                }
                return result
            
            logger.info(f"✅ REDIS CACHE - Face Recognition: Face embedding extracted in {embedding_time:.3f}s")
            
            # Step 3: Try to get embeddings from cache
            cache_start = time.time()
            cached_embeddings = self.redis.get_all_embeddings()
            cache_retrieval_time = time.time() - cache_start
            cache_hit = cached_embeddings is not None
            
            if cache_hit:
                self.cache_hits += 1
                logger.info(f"🟢 REDIS CACHE - Face Recognition: Retrieved {len(cached_embeddings)} embeddings from cache in {cache_retrieval_time:.3f}s (Hit #{self.cache_hits})")
                embeddings = cached_embeddings
            else:
                # Fallback to database
                logger.warning(f"🟠 REDIS CACHE - Face Recognition: Cache miss, loading from database")
                db_start = time.time()
                embeddings = self._load_embeddings_from_db()
                db_time = time.time() - db_start
                
                logger.info(f"🐌 REDIS CACHE - Face Recognition: Loaded {len(embeddings) if embeddings else 0} embeddings from database in {db_time:.3f}s")
                
                # Cache the embeddings for next time
                if embeddings:
                    cache_store_start = time.time()
                    cache_success = self.redis.cache_all_embeddings(embeddings)
                    cache_store_time = time.time() - cache_store_start
                    
                    if cache_success:
                        logger.info(f"🟢 REDIS CACHE - Face Recognition: Cached {len(embeddings)} embeddings for future use in {cache_store_time:.3f}s")
                    else:
                        logger.warning(f"🟠 REDIS CACHE - Face Recognition: Failed to cache embeddings after {cache_store_time:.3f}s")
                else:
                    logger.error(f"🔴 REDIS CACHE - Face Recognition: No employee embeddings found in database")
                    result = {
                        'success': False,
                        'error': 'No employee embeddings found',
                        'confidence': 0.0,
                        'cache_hit': False,
                        'processing_time': self._get_timestamp() - start_time
                    }
                    return result
            
            # Step 4: Perform similarity search
            similarity_start = time.time()
            best_match = self._find_best_match(input_embedding, embeddings)
            similarity_time = time.time() - similarity_start
            processing_time = self._get_timestamp() - start_time
            self.recognition_times.append(processing_time)
            
            logger.info(f"🔍 REDIS CACHE - Face Recognition: Similarity search completed in {similarity_time:.3f}s, compared against {len(embeddings)} embeddings")
            
            # Step 5: Prepare result
            if best_match and best_match['similarity'] >= self.similarity_threshold:
                logger.info(f"🟢 REDIS CACHE - Face Recognition: Match found! Employee: {best_match['employee_id']}, Confidence: {best_match['similarity']:.3f}, Total time: {processing_time:.3f}s")
                result = {
                    'success': True,
                    'employee_id': best_match['employee_id'],
                    'confidence': round(best_match['similarity'], 3),
                    'cache_hit': cache_hit,
                    'processing_time': processing_time,
                    'metadata': {
                        'total_embeddings_compared': len(embeddings),
                        'recognition_method': 'cached' if cache_hit else 'database',
                        'confidence_level': self._get_confidence_level(best_match['similarity']),
                        'embedding_extraction_time': embedding_time,
                        'similarity_search_time': similarity_time,
                        'cache_retrieval_time': cache_retrieval_time if cache_hit else None
                    }
                }
            else:
                confidence = best_match['similarity'] if best_match else 0.0
                logger.warning(f"🔴 REDIS CACHE - Face Recognition: No match found. Best confidence: {confidence:.3f} (threshold: {self.similarity_threshold}), Total time: {processing_time:.3f}s")
                result = {
                    'success': False,
                    'error': 'No matching face found',
                    'confidence': round(confidence, 3),
                    'cache_hit': cache_hit,
                    'processing_time': processing_time,
                    'suggested_actions': self._get_suggestions(confidence)
                }
            
            # Step 6: Cache the result for a short time
            if settings.enable_redis_cache:
                cache_result_start = time.time()
                result_cached = self.redis.cache_recognition_result(image_hash, result, ttl=300)  # 5 minutes
                cache_result_time = time.time() - cache_result_start
                
                if result_cached:
                    logger.info(f"🟢 REDIS CACHE - Face Recognition: Recognition result cached for 5 minutes in {cache_result_time:.3f}s")
                else:
                    logger.warning(f"🟠 REDIS CACHE - Face Recognition: Failed to cache recognition result after {cache_result_time:.3f}s")
            
            return result
                
        except Exception as e:
            processing_time = self._get_timestamp() - start_time
            logger.error(f"🔴 REDIS CACHE - Face Recognition: Recognition failed after {processing_time:.3f}s: {str(e)}")
            return {
                'success': False,
                'error': f'Recognition error: {str(e)}',
                'confidence': 0.0,
                'cache_hit': False,
                'processing_time': processing_time
            }
    
    def _load_embeddings_from_db(self) -> List[Dict]:
        """Load embeddings from database"""
        try:
            if self.db is None:
                # Import here to avoid circular imports
                from app.employees.service import EmployeeService
                from app.core.database import SessionLocal
                
                db = SessionLocal()
                try:
                    employee_service = EmployeeService(db)
                    return employee_service.get_all_embeddings()
                finally:
                    db.close()
            else:
                from app.employees.service import EmployeeService
                employee_service = EmployeeService(self.db)
                return employee_service.get_all_embeddings()
                
        except Exception as e:
            logger.error(f"Error loading embeddings from database: {str(e)}")
            return []
    
    def _find_best_match(self, input_embedding: np.ndarray, cached_embeddings: List[Dict]) -> Optional[Dict]:
        """Find the best matching face using cached embeddings"""
        best_match = None
        best_similarity = 0.0
        
        for emb_data in cached_embeddings:
            try:
                # Get embedding (should already be deserialized from cache)
                stored_embedding = np.array(emb_data['embedding'])
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(input_embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'employee_id': emb_data['employee_id'],
                        'similarity': similarity,
                        'image_path': emb_data.get('image_path', '')
                    }
                    
            except Exception as e:
                logger.warning(f"Error processing embedding: {str(e)}")
                continue
        
        return best_match
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are numpy arrays
            vec1 = np.array(embedding1).flatten()
            vec2 = np.array(embedding2).flatten()
            
            # Calculate norms
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash for image to use as cache key"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return hashlib.md5(image_data).hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating image hash: {str(e)}")
            return hashlib.md5(image_path.encode()).hexdigest()
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description"""
        if confidence >= self.high_confidence_threshold:
            return "high"
        elif confidence >= self.similarity_threshold:
            return "medium"
        elif confidence >= settings.low_confidence_threshold:
            return "low"
        else:
            return "very_low"
    
    def _get_suggestions(self, confidence: float) -> List[str]:
        """Get suggestions based on confidence level"""
        if confidence >= settings.low_confidence_threshold:
            return [
                "Try better lighting",
                "Look directly at camera",
                "Remove glasses if possible",
                "Ensure face is clearly visible"
            ]
        else:
            return [
                "Ensure you are registered in the system",
                "Try different lighting conditions",
                "Look directly at camera",
                "Contact HR if issue persists"
            ]
    
    def _get_timestamp(self) -> float:
        """Get current timestamp in seconds"""
        import time
        return time.time()
    
    def warm_cache(self) -> bool:
        """Pre-load all embeddings into cache"""
        try:
            # Load embeddings from database
            embeddings = self._load_embeddings_from_db()
            
            if embeddings:
                # Cache all embeddings
                success = self.redis.cache_all_embeddings(embeddings)
                
                # Also cache individual employee embeddings
                employee_embeddings_map = {}
                for emb in embeddings:
                    employee_id = emb['employee_id']
                    if employee_id not in employee_embeddings_map:
                        employee_embeddings_map[employee_id] = []
                    employee_embeddings_map[employee_id].append(emb)
                
                # Cache individual employee embeddings
                for employee_id, emp_embeddings in employee_embeddings_map.items():
                    self.redis.cache_employee_embeddings(employee_id, emp_embeddings)
                
                logger.info(f"Cache warmed with {len(embeddings)} embeddings for {len(employee_embeddings_map)} employees")
                return success
            else:
                logger.warning("No embeddings found to warm cache")
                return False
                
        except Exception as e:
            logger.error(f"Error warming cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get caching performance statistics"""
        cache_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0.0
        avg_processing_time = sum(self.recognition_times) / len(self.recognition_times) if self.recognition_times else 0.0
        
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.total_requests - self.cache_hits,
            'cache_hit_rate': round(cache_hit_rate, 2),
            'average_processing_time': round(avg_processing_time, 3),
            'redis_available': self.redis.is_available(),
            'redis_stats': self.redis.get_cache_stats()
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_requests = 0
        self.cache_hits = 0
        self.recognition_times = []

# Global cached recognition service instance
cached_recognition_service = CachedFaceRecognitionService(
    redis_service=None  # Will be initialized when needed
)
