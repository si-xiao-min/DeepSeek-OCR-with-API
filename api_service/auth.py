"""
API Key认证模块
实现基础的API访问控制
"""
import logging
from typing import Optional
from fastapi import Header, HTTPException, status
from .config import config

logger = logging.getLogger(__name__)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> bool:
    """
    验证API Key的FastAPI依赖注入函数

    Args:
        x_api_key: 从请求头"X-API-Key"中获取的API Key

    Returns:
        True（如果验证通过）

    Raises:
        HTTPException: 如果API Key缺失或无效（401 Unauthorized）
    """
    # 检查是否配置了API Keys
    if not config.API_KEYS:
        logger.warning("未配置API_KEY，允许所有请求（仅用于开发）")
        return True

    # 检查请求头中是否包含API Key
    if not x_api_key:
        logger.warning(f"请求缺少API Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # 验证API Key是否有效
    if x_api_key not in config.API_KEYS:
        logger.warning(f"无效的API Key: {x_api_key[:4]}****（已隐藏）")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # API Key验证通过
    logger.debug(f"API Key验证通过: {x_api_key[:4]}****")
    return True


async def verify_api_key_optional(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> bool:
    """
    可选的API Key验证（用于某些公开端点）
    如果提供了API Key，会验证其有效性，但不强制要求

    Args:
        x_api_key: 从请求头"X-API-Key"中获取的API Key（可选）

    Returns:
        True（始终返回True）

    Raises:
        HTTPException: 如果提供了无效的API Key（401 Unauthorized）
    """
    # 如果没有配置API Keys，直接通过
    if not config.API_KEYS:
        return True

    # 如果没有提供API Key，直接通过（因为这是可选验证）
    if not x_api_key:
        return True

    # 如果提供了API Key，验证其有效性
    if x_api_key not in config.API_KEYS:
        logger.warning(f"无效的API Key（可选验证）: {x_api_key[:4]}****")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True


def get_api_key_info(x_api_key: Optional[str] = None) -> dict:
    """
    获取API Key信息（用于日志或审计）

    Args:
        x_api_key: API Key

    Returns:
        包含API Key信息的字典（敏感信息已脱敏）
    """
    if not x_api_key:
        return {"provided": False}

    return {
        "provided": True,
        "prefix": x_api_key[:4] if len(x_api_key) > 4 else x_api_key,
        "length": len(x_api_key),
    }


def generate_api_key(prefix: str = "dso_", length: int = 16) -> str:
    """
    生成随机API Key

    Args:
        prefix: API Key前缀
        length: 随机部分长度

    Returns:
        生成的API Key
    """
    import secrets
    import string

    # 生成随机字符串
    alphabet = string.ascii_lowercase + string.digits
    random_part = ''.join(secrets.choice(alphabet) for _ in range(length))

    api_key = f"{prefix}{random_part}"

    logger.info(f"生成新的API Key: {api_key[:8]}****（已隐藏）")
    return api_key


if __name__ == "__main__":
    # 测试代码
    print("生成示例API Key:")
    for i in range(5):
        key = generate_api_key()
        print(f"  {i+1}. {key}")
