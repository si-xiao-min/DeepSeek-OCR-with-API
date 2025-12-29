"""
API服务配置管理模块
从环境变量读取配置，提供默认值和验证
"""
import os
from typing import List
from pathlib import Path


class Config:
    """API服务配置类"""

    # API配置
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # 模型配置
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/hy-tmp/deepseek-ocr-model/")
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.75"))
    MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "8192"))
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"

    # API Keys配置
    API_KEYS: List[str] = []
    _api_keys_str = os.getenv("DEEPSEEK_OCR_API_KEYS", "1228")
    if _api_keys_str:
        API_KEYS = [key.strip() for key in _api_keys_str.split(",") if key.strip()]

    # OSS配置
    ALLOWED_OSS_DOMAINS: List[str] = []
    _oss_domains = os.getenv(
        "ALLOWED_OSS_DOMAINS",
        ".oss-cn-wuhan-lr.aliyuncs.com,suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com"
    )
    if _oss_domains:
        ALLOWED_OSS_DOMAINS = [domain.strip() for domain in _oss_domains.split(",") if domain.strip()]

    # 图片配置
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "50"))
    IMAGE_DOWNLOAD_TIMEOUT: int = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "30"))
    ALLOWED_IMAGE_FORMATS: List[str] = ["image/jpeg", "image/png", "image/webp"]

    # 任务队列配置
    MAX_TASKS_IN_MEMORY: int = int(os.getenv("MAX_TASKS_IN_MEMORY", "1000"))
    TASK_TTL_SECONDS: int = int(os.getenv("TASK_TTL_SECONDS", "3600"))  # 1小时

    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "api_service.log")

    # CORS配置（宽松策略）
    CORS_ORIGINS: List[str] = ["*"]  # 允许所有源
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]  # 允许所有方法
    CORS_ALLOW_HEADERS: List[str] = ["*"]  # 允许所有头
    CORS_EXPOSE_HEADERS: List[str] = ["*"]  # 暴露所有响应头

    @classmethod
    def validate(cls) -> None:
        """验证配置是否合法"""
        if not cls.API_KEYS:
            print("警告: 未配置API_KEY，将使用默认值 '1228'")
            cls.API_KEYS = ["1228"]

        if cls.GPU_MEMORY_UTILIZATION <= 0 or cls.GPU_MEMORY_UTILIZATION > 1:
            raise ValueError(f"GPU_MEMORY_UTILIZATION 必须在 (0, 1] 范围内，当前值: {cls.GPU_MEMORY_UTILIZATION}")

        if cls.MAX_IMAGE_SIZE_MB <= 0:
            raise ValueError(f"MAX_IMAGE_SIZE_MB 必须大于0，当前值: {cls.MAX_IMAGE_SIZE_MB}")

        if not Path(cls.MODEL_PATH).exists():
            print(f"警告: 模型路径不存在: {cls.MODEL_PATH}")
            print("如果模型需要从HuggingFace下载，请忽略此警告")

    @classmethod
    def print_config(cls) -> None:
        """打印配置信息（隐藏敏感信息）"""
        print("=" * 60)
        print("DeepSeek-OCR API 服务配置")
        print("=" * 60)
        print(f"API 地址: {cls.API_HOST}:{cls.API_PORT}")
        print(f"模型路径: {cls.MODEL_PATH}")
        print(f"GPU 内存利用率: {cls.GPU_MEMORY_UTILIZATION}")
        print(f"最大模型长度: {cls.MAX_MODEL_LEN}")
        print(f"API Keys 数量: {len(cls.API_KEYS)}")
        print(f"允许的OSS域名: {cls.ALLOWED_OSS_DOMAINS}")
        print(f"最大图片大小: {cls.MAX_IMAGE_SIZE_MB}MB")
        print(f"任务TTL: {cls.TASK_TTL_SECONDS}秒")
        print(f"CORS策略: 允许所有源、方法和头")
        print("=" * 60)


# 创建全局配置实例
config = Config()

if __name__ == "__main__":
    # 测试配置
    config.validate()
    config.print_config()
