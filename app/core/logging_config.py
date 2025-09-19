"""
Logging Configuration for Face Recognition Attendance System
Provides comprehensive logging for Redis cache operations and system performance
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

class RedisLogFormatter(logging.Formatter):
    """Custom formatter for Redis cache operations with color coding"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to log level
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        # Add emoji indicators for Redis operations
        if "REDIS CACHE" in formatted:
            if "🟢" in formatted:  # Success
                pass
            elif "🔴" in formatted:  # Error
                pass
            elif "🟡" in formatted:  # In Progress
                pass
            elif "🟠" in formatted:  # Warning
                pass
        
        return formatted

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file_rotation: bool = True
):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    console_formatter = RedisLogFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        if enable_file_rotation:
            # Rotating file handler (10MB max, keep 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create specific loggers for different components
    loggers = [
        'app.core.redis_service',
        'app.face_recognition.cached_recognition_service',
        'app.employees.service',
        'app.attendance.service',
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)
    
    return root_logger

def log_redis_operation(
    operation: str,
    employee_id: Optional[str] = None,
    duration: Optional[float] = None,
    success: bool = True,
    details: Optional[dict] = None,
    logger: Optional[logging.Logger] = None
):
    """Log Redis cache operations with standardized format"""
    
    if logger is None:
        logger = logging.getLogger('app.core.redis_service')
    
    # Prepare log message
    status_emoji = "🟢" if success else "🔴"
    operation_type = operation.upper().replace("_", " ")
    
    message_parts = [
        f"{status_emoji} REDIS CACHE - {operation_type}:"
    ]
    
    if employee_id:
        message_parts.append(f"Employee: {employee_id}")
    
    if duration is not None:
        message_parts.append(f"Duration: {duration:.3f}s")
    
    if details:
        for key, value in details.items():
            message_parts.append(f"{key}: {value}")
    
    message = " | ".join(message_parts)
    
    if success:
        logger.info(message)
    else:
        logger.error(message)

def log_performance_metrics(
    operation: str,
    metrics: dict,
    logger: Optional[logging.Logger] = None
):
    """Log performance metrics for Redis operations"""
    
    if logger is None:
        logger = logging.getLogger('app.core.redis_service')
    
    message = f"📊 REDIS PERFORMANCE - {operation.upper()}: "
    metric_parts = []
    
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_parts.append(f"{key}: {value:.3f}")
        else:
            metric_parts.append(f"{key}: {value}")
    
    message += " | ".join(metric_parts)
    logger.info(message)

class RedisOperationLogger:
    """Context manager for logging Redis operations"""
    
    def __init__(
        self,
        operation: str,
        employee_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.operation = operation
        self.employee_id = employee_id
        self.logger = logger or logging.getLogger('app.core.redis_service')
        self.start_time = None
        self.success = False
        self.details = {}
    
    def __enter__(self):
        self.start_time = datetime.now().timestamp()
        self.logger.info(f"🟡 REDIS CACHE - {self.operation.upper()}: Starting operation for employee {self.employee_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now().timestamp() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            self.success = True
        else:
            self.details['error'] = str(exc_val)
        
        log_redis_operation(
            self.operation,
            self.employee_id,
            duration,
            self.success,
            self.details,
            self.logger
        )
    
    def add_detail(self, key: str, value):
        """Add additional details to the log"""
        self.details[key] = value
    
    def set_success(self, success: bool):
        """Set operation success status"""
        self.success = success

# Initialize logging on import
def init_logging():
    """Initialize logging configuration"""
    log_file = "logs/face_recognition_cache.log"
    setup_logging(
        log_level="INFO",
        log_file=log_file,
        enable_console=True,
        enable_file_rotation=True
    )

# Auto-initialize logging
init_logging()
