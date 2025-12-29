"""
工具函数模块
提供各种辅助功能
"""
import re
import uuid
import logging
from typing import Optional, List
from urllib.parse import urlparse
import aiohttp
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from .config import config

logger = logging.getLogger(__name__)


def generate_task_id() -> str:
    """
    生成唯一的任务ID

    Returns:
        UUID字符串
    """
    return str(uuid.uuid4())


def validate_url(url: str) -> bool:
    """
    验证URL格式

    Args:
        url: URL字符串

    Returns:
        是否为有效的URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_domain_allowed(url: str, allowed_domains: List[str]) -> bool:
    """
    检查URL域名是否在白名单中

    Args:
        url: URL字符串
        allowed_domains: 允许的域名列表（支持通配符，如.example.com）

    Returns:
        域名是否允许
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # 移除端口号
        if ':' in domain:
            domain = domain.split(':')[0]

        # 检查是否在白名单中
        for allowed in allowed_domains:
            allowed = allowed.lower().strip()

            # 支持通配符（.example.com匹配example.com及其所有子域名）
            if allowed.startswith('.'):
                if domain == allowed[1:] or domain.endswith(allowed):
                    return True
            # 精确匹配
            elif domain == allowed:
                return True

        return False

    except Exception as e:
        logger.error(f"域名检查失败: {e}")
        return False


async def get_file_size_from_url(url: str, timeout: int = 10) -> Optional[int]:
    """
    获取远程文件大小（通过HEAD请求）

    Args:
        url: 文件URL
        timeout: 超时时间（秒）

    Returns:
        文件大小（字节），如果获取失败返回None
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=timeout) as response:
                if response.status == 200:
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        return int(content_length)
    except Exception as e:
        logger.debug(f"无法获取文件大小: {e}")

    return None


def is_valid_image_format(mime_type: str) -> bool:
    """
    验证图片MIME类型

    Args:
        mime_type: MIME类型字符串

    Returns:
        是否为支持的图片格式
    """
    return mime_type.lower() in config.ALLOWED_IMAGE_FORMATS


async def download_image(url: str, timeout: int = None) -> Image.Image:
    """
    异步下载图片

    Args:
        url: 图片URL
        timeout: 超时时间（秒），默认使用配置值

    Returns:
        PIL Image对象（RGB格式）

    Raises:
        ValueError: URL无效或域名不在白名单
        aiohttp.ClientError: 下载失败
        UnidentifiedImageError: 图片格式不支持
        RuntimeError: 图片过大
    """
    if timeout is None:
        timeout = config.IMAGE_DOWNLOAD_TIMEOUT

    # 验证URL
    if not validate_url(url):
        raise ValueError(f"无效的URL: {url}")

    # 检查域名白名单
    if not is_domain_allowed(url, config.ALLOWED_OSS_DOMAINS):
        raise ValueError(f"URL域名不在白名单中: {url}")

    try:
        async with aiohttp.ClientSession() as session:
            # 先获取文件大小
            file_size = await get_file_size_from_url(url, timeout)
            if file_size:
                size_mb = file_size / (1024 * 1024)
                if size_mb > config.MAX_IMAGE_SIZE_MB:
                    raise RuntimeError(
                        f"图片过大: {size_mb:.2f}MB (最大允许: {config.MAX_IMAGE_SIZE_MB}MB)"
                    )

            # 下载图片
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(f"HTTP {response.status}: {url}")

                # 检查Content-Type
                content_type = response.headers.get('Content-Type', '')
                if content_type and not is_valid_image_format(content_type):
                    # 尝试解析，让PIL来决定是否支持
                    pass

                # 读取图片数据
                image_data = await response.read()

                # 验证大小（再次检查）
                size_mb = len(image_data) / (1024 * 1024)
                if size_mb > config.MAX_IMAGE_SIZE_MB:
                    raise RuntimeError(
                        f"图片过大: {size_mb:.2f}MB (最大允许: {config.MAX_IMAGE_SIZE_MB}MB)"
                    )

                # 打开图片
                image = Image.open(BytesIO(image_data))

                # 转换为RGB格式
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                logger.info(f"成功下载图片: {url} ({image.size[0]}x{image.size[1]})")
                return image

    except aiohttp.ClientError as e:
        logger.error(f"图片下载失败: {url}, 错误: {e}")
        raise
    except UnidentifiedImageError as e:
        logger.error(f"不支持的图片格式: {url}, 错误: {e}")
        raise
    except Exception as e:
        logger.error(f"下载图片时发生未知错误: {url}, 错误: {e}")
        raise


def get_error_code_from_exception(e: Exception) -> str:
    """
    将异常映射到错误码

    Args:
        e: 异常对象

    Returns:
        错误码字符串
    """
    error_mapping = {
        ValueError: "INVALID_REQUEST",
        aiohttp.ClientError: "IMAGE_DOWNLOAD_ERROR",
        UnidentifiedImageError: "UNSUPPORTED_IMAGE_FORMAT",
        RuntimeError: "IMAGE_TOO_LARGE",
        # torch.cuda.OutOfMemoryError: "GPU_OUT_OF_MEMORY",  # 在model_manager中处理
    }

    for exception_type, error_code in error_mapping.items():
        if isinstance(e, exception_type):
            return error_code

    # 未匹配的异常
    logger.warning(f"未映射的异常类型: {type(e).__name__}")
    return "INTERNAL_SERVER_ERROR"


def parse_ocr_result(raw_result: str, level: str) -> dict:
    """
    解析和格式化OCR原始结果

    Args:
        raw_result: 模型输出的原始文本
        level: 识别级别（min/middle/max）

    Returns:
        格式化后的结果字典
    """
    result = {"text": raw_result}

    # 尝试解析JSON（如果输出是JSON格式）
    if raw_result.strip().startswith('{'):
        try:
            import json
            parsed = json.loads(raw_result)

            # 根据级别过滤字段
            if level == "min":
                # min级别只返回text
                if "text" in parsed:
                    result["text"] = parsed["text"]
            elif level == "middle":
                # middle级别返回text和layout_info
                result["text"] = parsed.get("text", raw_result)
                if "layout_info" in parsed:
                    result["layout_info"] = parsed["layout_info"]
            elif level == "max":
                # max级别返回所有字段
                result = parsed

        except json.JSONDecodeError:
            # JSON解析失败，返回原始文本
            logger.debug("JSON解析失败，返回原始文本")
            result["text"] = raw_result

    return result


def setup_logging(log_level: str = None, log_file: str = None):
    """
    配置日志系统

    Args:
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_file: 日志文件路径
    """
    if log_level is None:
        log_level = config.LOG_LEVEL
    if log_file is None:
        log_file = config.LOG_FILE

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除现有处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"日志系统已配置: 级别={log_level}, 文件={log_file}")


if __name__ == "__main__":
    # 测试代码
    import asyncio

    async def test_download():
        # 测试图片下载（使用示例URL）
        test_url = "https://picsum.photos/800/600"
        try:
            image = await download_image(test_url)
            print(f"成功下载图片: {image.size}, 模式: {image.mode}")
        except Exception as e:
            print(f"下载失败: {e}")

    # 测试任务ID生成
    print("生成任务ID:")
    for i in range(5):
        print(f"  {i+1}. {generate_task_id()}")

    # 测试URL验证
    print("\nURL验证测试:")
    test_urls = [
        "https://example.com/image.jpg",
        "invalid-url",
        "http://test.oss-cn-wuhan-lr.aliyuncs.com/file.png",
    ]
    for url in test_urls:
        print(f"  {url}: {validate_url(url)}")

    # 测试域名白名单
    print("\n域名白名单测试:")
    test_domains = [
        "https://test.oss-cn-wuhan-lr.aliyuncs.com/file.png",
        "https://suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com/image.jpg",
        "https://evil.com/image.jpg",
    ]
    for url in test_domains:
        allowed = is_domain_allowed(url, config.ALLOWED_OSS_DOMAINS)
        print(f"  {url}: {'允许' if allowed else '拒绝'}")
