"""
DeepSeek-OCR API 服务
提供RESTful API接口进行图片识别
"""

__version__ = "1.0.0"
__author__ = "DeepSeek-OCR Team"

from .config import config
from .models import (
    ImageSubmitRequest,
    TaskSubmitResponse,
    TaskStatusResponse,
    OCRResultData,
    ErrorResponse,
    HealthResponse
)

__all__ = [
    "config",
    "ImageSubmitRequest",
    "TaskSubmitResponse",
    "TaskStatusResponse",
    "OCRResultData",
    "ErrorResponse",
    "HealthResponse",
]
