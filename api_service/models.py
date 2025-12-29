"""
Pydantic数据模型定义
定义所有请求和响应的数据结构
"""
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, field_validator
import re


class ImageSubmitRequest(BaseModel):
    """
    图片识别请求模型
    """
    image_url: str = Field(..., description="图片URL（阿里云OSS）")
    image_name: Optional[str] = Field(None, description="图片名称/标题")
    image_background: Optional[str] = Field(None, description="背景信息描述")
    custom_prompt: Optional[str] = Field(None, description="自定义提示词")
    historical_context: Optional[str] = Field(None, description="历史背景信息（适用于人物画、文物等）")
    artistic_notes: Optional[str] = Field(None, description="艺术技法说明（适用于画作分析）")
    language: str = Field("zh", description="语言代码，默认zh")

    @field_validator('image_url')
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """验证URL格式"""
        if not v or not v.strip():
            raise ValueError("image_url 不能为空")

        # 基本URL格式验证
        url_pattern = re.compile(
            r'^https?://'  # http:// 或 https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # 域名
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # 端口
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        if not url_pattern.match(v):
            raise ValueError(f"无效的URL格式: {v}")

        return v.strip()

    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """验证语言代码"""
        supported_languages = ['zh', 'en']
        v = v.lower().strip()
        if v not in supported_languages:
            raise ValueError(f"不支持的语言代码: {v}，支持的值: {supported_languages}")
        return v


class TaskSubmitResponse(BaseModel):
    """
    任务提交响应模型
    """
    success: bool = True
    task_id: str = Field(..., description="任务ID")
    status: str = Field("pending", description="任务状态")
    queue_position: int = Field(0, description="队列位置，0表示立即处理")
    message: str = Field("任务已提交", description="响应消息")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "status": "pending",
                "queue_position": 0,
                "message": "任务已提交"
            }
        }


class OCRResultData(BaseModel):
    """
    OCR识别结果数据模型（根据级别不同字段会有差异）
    """
    text: str = Field(..., description="识别的文本内容")
    layout_info: Optional[List[Dict[str, Any]]] = Field(None, description="版面信息（middle和max级别）")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="实体信息（middle和max级别）")
    regions: Optional[List[Dict[str, Any]]] = Field(None, description="区域详情（仅max级别）")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "识别的文本内容",
                "layout_info": [{"type": "paragraph", "bbox": [0, 0, 100, 50]}],
                "entities": [{"type": "person", "text": "张三", "confidence": 0.95}],
                "regions": [{"type": "person", "description": "...", "location": "..."}]
            }
        }


class TaskStatusResponse(BaseModel):
    """
    任务状态查询响应模型
    """
    success: bool = True
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态: pending/processing/completed/failed")
    queue_position: int = Field(0, description="队列位置")
    result: Optional[OCRResultData] = Field(None, description="识别结果（仅completed状态）")
    error: Optional[str] = Field(None, description="错误信息（仅failed状态）")
    error_code: Optional[str] = Field(None, description="错误码（仅failed状态）")
    created_at: str = Field(..., description="任务创建时间（ISO 8601）")
    completed_at: Optional[str] = Field(None, description="任务完成时间（ISO 8601）")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "status": "completed",
                "queue_position": 0,
                "result": {
                    "text": "识别的文本内容"
                },
                "error": None,
                "error_code": None,
                "created_at": "2025-12-28T12:00:00Z",
                "completed_at": "2025-12-28T12:00:10Z"
            }
        }


class ErrorResponse(BaseModel):
    """
    错误响应模型
    """
    success: bool = False
    error: str = Field(..., description="错误信息")
    error_code: str = Field(..., description="错误码")
    detail: Optional[str] = Field(None, description="详细错误信息")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "success": False,
                    "error": "未授权的API Key",
                    "error_code": "UNAUTHORIZED",
                    "detail": None
                },
                {
                    "success": False,
                    "error": "图片下载失败",
                    "error_code": "IMAGE_DOWNLOAD_ERROR",
                    "detail": "URL: https://example.com/image.jpg"
                }
            ]
        }


class HealthResponse(BaseModel):
    """
    健康检查响应模型
    """
    status: str = Field(..., description="服务状态: healthy/unhealthy")
    model_loaded: bool = Field(..., description="模型是否已成功加载")
    queue_size: int = Field(..., description="当前队列中待处理的任务数量")
    gpu_memory_utilization: float = Field(..., description="GPU内存利用率配置")
    version: str = Field("1.0.0", description="API版本")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "queue_size": 5,
                "gpu_memory_utilization": 0.75,
                "version": "1.0.0"
            }
        }
