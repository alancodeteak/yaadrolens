

import redis
import json
import numpy as np
import pickle
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import time
from app.core.config import settings

# Configure logger with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisCacheService:
    """Redis-based caching service for face embeddings"""
    
    def __init__(self, redis_url: str = None, password: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self.password = password or settings.redis_password
        
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                password=self.password,
                decode_responses=False,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_socket_connect_timeout,
                max_connections=settings.redis_max_connections
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None
        
        self.embedding_dim = 512  # ArcFace dimension
        self.cache_ttl = settings.cache_ttl
        self.bulk_cache_ttl = settings.bulk_cache_ttl
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.redis_client is not None
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding for Redis storage"""
        try:
            return pickle.dumps(embedding)
        except Exception as e:
            logger.error(f"Error serializing embedding: {str(e)}")
            return b""
    
    def _deserialize_embedding(self, data: bytes) -> List[float]:
        """Deserialize embedding from Redis"""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializing embedding: {str(e)}")
            return []
    
    def cache_employee_embeddings(self, employee_id: str, embeddings: List[Dict]) -> bool:
        """Cache embeddings for a specific employee"""
        operation_start = time.time()
        
        if not self.is_available():
            logger.warning(f"🔴 REDIS CACHE - Employee Registration: Redis unavailable for employee {employee_id}")
            return False
            
        try:
            key = f"embeddings:employee:{employee_id}"
            logger.info(f"🟡 REDIS CACHE - Employee Registration: Starting cache operation for employee {employee_id}")
            
            # Prepare data for caching
            cache_data = {
                'employee_id': employee_id,
                'embeddings': [],
                'cached_at': datetime.now().isoformat(),
                'count': len(embeddings)
            }
            
            # Process embeddings
            processed_count = 0
            for emb in embeddings:
                try:
                    # Parse JSON embedding if it's a string
                    if isinstance(emb['embedding'], str):
                        embedding_list = json.loads(emb['embedding'])
                    else:
                        embedding_list = emb['embedding']
                    
                    cache_data['embeddings'].append({
                        'id': str(emb.get('id', '')),
                        'embedding': self._serialize_embedding(embedding_list),
                        'image_path': emb.get('image_path', ''),
                        'created_at': emb.get('created_at', datetime.now().isoformat())
                    })
                    processed_count += 1
                except Exception as e:
                    logger.warning(f"🟠 REDIS CACHE - Employee Registration: Error processing embedding {emb.get('id', 'unknown')} for employee {employee_id}: {str(e)}")
                    continue
            
            # Store in Redis
            self.redis_client.setex(
                key, 
                self.cache_ttl, 
                pickle.dumps(cache_data)
            )
            
            operation_time = time.time() - operation_start
            logger.info(f"🟢 REDIS CACHE - Employee Registration: Successfully cached {processed_count}/{len(embeddings)} embeddings for employee {employee_id} in {operation_time:.3f}s")
            
            # Also invalidate bulk cache to ensure consistency
            self.redis_client.delete("embeddings:all")
            logger.info(f"🟡 REDIS CACHE - Employee Registration: Invalidated bulk cache for consistency")
            
            return True
            
        except Exception as e:
            operation_time = time.time() - operation_start
            logger.error(f"🔴 REDIS CACHE - Employee Registration: Failed to cache embeddings for employee {employee_id} after {operation_time:.3f}s: {str(e)}")
            return False
    
    def get_employee_embeddings(self, employee_id: str) -> Optional[Dict]:
        """Get cached embeddings for an employee"""
        if not self.is_available():
            return None
            
        try:
            key = f"embeddings:employee:{employee_id}"
            data = self.redis_client.get(key)
            
            if data:
                cache_data = pickle.loads(data)
                # Deserialize embeddings
                for emb in cache_data['embeddings']:
                    emb['embedding'] = self._deserialize_embedding(emb['embedding'])
                return cache_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting employee embeddings from cache: {str(e)}")
            return None
    
    def cache_all_embeddings(self, embeddings: List[Dict]) -> bool:
        """Cache all active embeddings for fast recognition"""
        if not self.is_available():
            return False
            
        try:
            key = "embeddings:all"
            
            # Prepare bulk cache data
            cache_data = {
                'embeddings': [],
                'cached_at': datetime.now().isoformat(),
                'count': len(embeddings)
            }
            
            # Process embeddings
            for emb in embeddings:
                try:
                    # Parse JSON embedding if it's a string
                    if isinstance(emb['embedding'], str):
                        embedding_list = json.loads(emb['embedding'])
                    else:
                        embedding_list = emb['embedding']
                    
                    cache_data['embeddings'].append({
                        'employee_id': str(emb['employee_id']),
                        'embedding': self._serialize_embedding(embedding_list),
                        'image_path': emb.get('image_path', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing embedding: {str(e)}")
                    continue
            
            # Store in Redis
            self.redis_client.setex(
                key,
                self.bulk_cache_ttl,
                pickle.dumps(cache_data)
            )
            
            logger.info(f"Cached {len(cache_data['embeddings'])} embeddings for bulk recognition")
            return True
            
        except Exception as e:
            logger.error(f"Error caching all embeddings: {str(e)}")
            return False
    
    def get_all_embeddings(self) -> Optional[List[Dict]]:
        """Get all cached embeddings"""
        if not self.is_available():
            return None
            
        try:
            key = "embeddings:all"
            data = self.redis_client.get(key)
            
            if data:
                cache_data = pickle.loads(data)
                # Deserialize embeddings
                for emb in cache_data['embeddings']:
                    emb['embedding'] = self._deserialize_embedding(emb['embedding'])
                return cache_data['embeddings']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting all embeddings from cache: {str(e)}")
            return None
    
    def invalidate_employee_cache(self, employee_id: str) -> bool:
        """Invalidate cache for specific employee"""
        if not self.is_available():
            return False
            
        try:
            key = f"embeddings:employee:{employee_id}"
            self.redis_client.delete(key)
            
            # Also invalidate bulk cache
            self.redis_client.delete("embeddings:all")
            
            logger.info(f"Invalidated cache for employee {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating employee cache: {str(e)}")
            return False
    
    def invalidate_all_cache(self) -> bool:
        """Invalidate all embedding caches"""
        if not self.is_available():
            return False
            
        try:
            # Get all embedding keys
            keys = self.redis_client.keys("embeddings:*")
            if keys:
                self.redis_client.delete(*keys)
            
            logger.info("Invalidated all embedding caches")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating all caches: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.is_available():
            return {'status': 'unavailable'}
            
        try:
            info = self.redis_client.info()
            return {
                'status': 'available',
                'redis_version': info.get('redis_version'),
                'used_memory': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(info),
                'embedding_keys': len(self.redis_client.keys("embeddings:*")),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return round((hits / total * 100), 2) if total > 0 else 0.0
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        if not self.is_available():
            return False
            
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False
    
    def cache_recognition_result(self, image_hash: str, result: Dict, ttl: int = 300) -> bool:
        """Cache recognition result for a short time (5 minutes default)"""
        if not self.is_available():
            return False
            
        try:
            key = f"recognition:result:{image_hash}"
            self.redis_client.setex(key, ttl, json.dumps(result))
            return True
        except Exception as e:
            logger.error(f"Error caching recognition result: {str(e)}")
            return False
    
    def get_cached_recognition_result(self, image_hash: str) -> Optional[Dict]:
        """Get cached recognition result"""
        if not self.is_available():
            return None
            
        try:
            key = f"recognition:result:{image_hash}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Error getting cached recognition result: {str(e)}")
            return None

# Global Redis service instance
redis_service = RedisCacheService()
