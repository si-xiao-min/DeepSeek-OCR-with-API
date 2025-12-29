"""
OCR处理器模块
实现图片处理和OCR推理逻辑
"""
import logging
from typing import Dict, Any
from PIL import Image
from .model_manager import get_model_manager
from .prompts import build_prompt
from .utils import download_image, parse_ocr_result, get_error_code_from_exception
from .config import config

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR处理器类
    负责图片下载、提示词构建、模型推理和结果格式化
    """

    def __init__(self):
        """初始化OCR处理器"""
        self.model_manager = get_model_manager()
        logger.info("OCR处理器初始化完成")

    async def process(self, task_data: Dict[str, Any], level: str = "middle") -> Dict[str, Any]:
        """
        处理OCR任务

        Args:
            task_data: 任务数据，包含：
                - image_url: 图片URL
                - image_name: 图片名称（可选）
                - image_background: 背景信息（可选）
                - custom_prompt: 自定义提示词（可选）
                - historical_context: 历史背景（可选）
                - artistic_notes: 艺术技法说明（可选）
                - language: 语言代码
            level: 识别级别（min/middle/max）

        Returns:
            处理结果字典：
            {
                "success": True/False,
                "result": {...},  # 如果成功
                "error": "...",   # 如果失败
                "error_code": "..."  # 如果失败
            }
        """
        try:
            logger.info(f"开始处理OCR任务: level={level}, url={task_data.get('image_url')}")

            # 1. 下载图片
            image = await self._download_image(task_data["image_url"])

            # 2. 构建提示词
            prompt = self._build_prompt(task_data, level)
            logger.debug(f"提示词: {prompt[:100]}...")

            # 3. 执行OCR推理
            raw_result = await self._run_inference(image, prompt)
            logger.debug(f"原始结果长度: {len(raw_result)} 字符")

            # 4. 格式化结果
            formatted_result = self._format_result(raw_result, level, task_data)

            logger.info(f"OCR任务处理成功: level={level}")
            return {
                "success": True,
                "result": formatted_result
            }

        except Exception as e:
            error_code = get_error_code_from_exception(e)
            error_message = str(e)
            logger.error(f"OCR任务处理失败: {error_code} - {error_message}")

            return {
                "success": False,
                "error": error_message,
                "error_code": error_code
            }

    async def _download_image(self, url: str) -> Image.Image:
        """
        下载图片

        Args:
            url: 图片URL

        Returns:
            PIL Image对象

        Raises:
            各种下载和图片处理异常
        """
        logger.info(f"开始下载图片: {url}")
        image = await download_image(url)
        logger.info(f"图片下载成功: {image.size[0]}x{image.size[1]}")
        return image

    def _build_prompt(self, task_data: Dict[str, Any], level: str) -> str:
        """
        构建提示词

        Args:
            task_data: 任务数据
            level: 识别级别

        Returns:
            提示词字符串
        """
        # 将task_data转换为ImageSubmitRequest格式
        from .models import ImageSubmitRequest

        request = ImageSubmitRequest(**task_data)
        prompt = build_prompt(request, level)

        return prompt

    async def _run_inference(self, image: Image.Image, prompt: str) -> str:
        """
        运行OCR推理

        Args:
            image: PIL Image对象
            prompt: 提示词

        Returns:
            模型输出的文本

        Raises:
            RuntimeError: 推理失败
        """
        logger.info("开始OCR推理...")

        # 确保模型已加载
        if not self.model_manager.is_loaded():
            raise RuntimeError("模型尚未加载")

        # 执行推理
        result = await self.model_manager.generate(
            image=image,
            prompt=prompt,
            max_tokens=config.MAX_MODEL_LEN,
            temperature=0.0,
        )

        logger.info("OCR推理完成")
        return result

    def _format_result(self, raw_result: str, level: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化OCR结果

        Args:
            raw_result: 模型原始输出
            level: 识别级别
            task_data: 原始任务数据

        Returns:
            格式化后的结果字典
        """
        # 解析原始结果
        parsed = parse_ocr_result(raw_result, level)

        # 添加元数据
        formatted = {
            "text": parsed.get("text", raw_result),
        }

        # 根据级别添加额外字段
        if level in ["middle", "max"]:
            if "layout_info" in parsed:
                formatted["layout_info"] = parsed["layout_info"]

        if level == "max":
            if "entities" in parsed:
                formatted["entities"] = parsed["entities"]
            if "regions" in parsed:
                formatted["regions"] = parsed["regions"]

        # 添加元数据
        formatted["metadata"] = {
            "image_name": task_data.get("image_name"),
            "recognition_level": level,
            "language": task_data.get("language", "zh"),
        }

        return formatted


# 全局OCR处理器实例
_ocr_processor: OCRProcessor = None


def get_ocr_processor() -> OCRProcessor:
    """
    获取全局OCR处理器实例

    Returns:
        OCRProcessor单例
    """
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor()
    return _ocr_processor
