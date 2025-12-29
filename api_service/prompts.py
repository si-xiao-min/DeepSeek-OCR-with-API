"""
提示词管理模块
实现三级提示词模板管理
"""
import logging
from typing import Optional, Dict, Any
from .models import ImageSubmitRequest

logger = logging.getLogger(__name__)


def build_min_prompt(request: ImageSubmitRequest) -> str:
    """
    构建最小级别提示词
    只返回纯文本，不包含任何结构化信息

    Args:
        request: 图片提交请求

    Returns:
        提示词字符串
    """
    # 使用DeepSeek-OCR的标准提示词格式
    if request.language == "zh":
        prompt = "<image>\n请识别图片中的所有文字内容，按阅读顺序输出。"
    else:
        prompt = "<image>\nExtract all visible text content from the image."

    return prompt


def build_middle_prompt(request: ImageSubmitRequest) -> str:
    """
    构建中间级别提示词
    返回文本 + 基础版面信息（段落、标题等）

    Args:
        request: 图片提交请求

    Returns:
        提示词字符串
    """
    language = "Chinese" if request.language == "zh" else "English"

    prompt_parts = [
        "<image>",
        f"Perform OCR and extract text content in {language}.",
        "Return the result as a JSON object with the following structure:",
        "{",
        '  "text": "extracted text content",',
        '  "layout_info": [',
        '    {"type": "paragraph", "content": "text", "bbox": [x1, y1, x2, y2]},',
        '    {"type": "title", "content": "text", "bbox": [x1, y1, x2, y2]}',
        "  ]",
        "}",
    ]

    # 添加图片背景信息
    if request.image_background:
        prompt_parts.append(f"\nContext: {request.image_background}")

    prompt = "\n".join(prompt_parts)

    return prompt


def build_max_prompt(request: ImageSubmitRequest) -> str:
    """
    构建最大级别提示词
    返回文本 + 详细版面信息 + 实体识别 + 区域定位
    特别针对中国文化图片（人物画、山水画、文物等）

    Args:
        request: 图片提交请求

    Returns:
        提示词字符串
    """
    # 构建基础提示词
    if request.language == "zh":
        prompt = "<image>\n<|grounding|>请详细识别这张图片中的所有内容：\n\n"
        prompt += "1. 文字识别：识别所有可见的文字，包括题跋、印章、标题等\n"
        prompt += "2. 内容描述：如果是画作，描述画面内容和构图；如果是文物，描述其特征\n"
        prompt += "3. 格式要求：按从右到左、从上到下的顺序排列文字内容"
    else:
        prompt = "<image>\n<|grounding|>Please identify all content in this image:\n\n"
        prompt += "1. Text recognition: Extract all visible text including inscriptions and seals\n"
        prompt += "2. Content description: Describe the scene and composition\n"
        prompt += "3. Format: Organize text in proper reading order"

    # 添加历史背景信息（适用于人物画、文物等）
    if request.historical_context:
        prompt += f"\n\n历史背景：{request.historical_context}"

    # 添加艺术技法说明（适用于画作分析）
    if request.artistic_notes:
        prompt += f"\n\n艺术技法：{request.artistic_notes}"

    # 添加图片背景信息
    if request.image_background:
        prompt += f"\n\n背景描述：{request.image_background}"

    return prompt


def build_prompt(request: ImageSubmitRequest, level: str = "middle") -> str:
    """
    统一的提示词构建入口

    Args:
        request: 图片提交请求
        level: 识别级别（"min", "middle", "max"）

    Returns:
        提示词字符串

    Raises:
        ValueError: 如果level参数无效
    """
    # 如果用户提供了自定义提示词，优先使用
    if request.custom_prompt:
        logger.info("使用自定义提示词")
        custom_prompt = request.custom_prompt

        # 确保自定义提示词包含<image>标记
        if "<image>" not in custom_prompt:
            custom_prompt = "<image>\n" + custom_prompt

        # 对于max级别，如果没有grounding标记，添加它
        if level == "max" and "<|grounding|>" not in custom_prompt:
            custom_prompt = custom_prompt.replace("<image>", "<image>\n<|grounding|>")

        return custom_prompt

    # 根据级别构建提示词
    if level == "min":
        return build_min_prompt(request)
    elif level == "middle":
        return build_middle_prompt(request)
    elif level == "max":
        return build_max_prompt(request)
    else:
        raise ValueError(f"无效的识别级别: {level}，支持的值: 'min', 'middle', 'max'")


def extract_prompt_info(prompt: str) -> Dict[str, Any]:
    """
    从提示词中提取信息（用于日志和调试）

    Args:
        prompt: 提示词字符串

    Returns:
        包含提示词信息的字典
    """
    info = {
        "length": len(prompt),
        "has_image_token": "<image>" in prompt,
        "has_grounding": "<|grounding|>" in prompt,
        "has_custom_content": False,
        "language": "unknown",
    }

    # 简单的语言检测
    if any(char in prompt for char in "中文请识别文字内容书法题跋"):
        info["language"] = "zh"
    elif any(char in prompt for char in "Extract text content OCR English"):
        info["language"] = "en"

    # 检测是否包含自定义内容
    if "Context:" in prompt or "背景" in prompt or "historical" in prompt:
        info["has_custom_content"] = True

    return info


# 预定义的提示词模板（用于常见场景）
PREDEFINED_PROMPTS = {
    "chinese_portrait": {
        "level": "max",
        "template": "<image>\n<|grounding|>\n请识别这幅中国人物画中的所有人物，包括他们的位置、服饰、姿态和面部特征。如果有题跋或印章，也请识别其中的文字。",
        "description": "中国人物画识别"
    },
    "chinese_landscape": {
        "level": "max",
        "template": "<image>\n<|grounding|>\n请分析这幅中国山水画的构图，包括山石、水流、树木、建筑等元素的位置和特点。识别所有题跋和印章文字。",
        "description": "中国山水画识别"
    },
    "chinese_calligraphy": {
        "level": "middle",
        "template": "<image>\n请识别这幅书法作品的所有文字内容，按从右到左、从上到下的顺序排列。",
        "description": "中国书法识别"
    },
    "cultural_relic": {
        "level": "max",
        "template": "<image>\n<|grounding|>\n请识别这件文物的所有文字信息和铭文，并描述其外形特征、纹饰和工艺细节。",
        "description": "文物识别"
    },
    "document_ocr": {
        "level": "middle",
        "template": "<image>\n请将这个文档转换为markdown格式，保留原有的标题、段落和列表结构。",
        "description": "文档OCR"
    },
}


def get_predefined_prompt(name: str) -> Optional[str]:
    """
    获取预定义的提示词模板

    Args:
        name: 模板名称

    Returns:
        提示词模板，如果不存在返回None
    """
    if name in PREDEFINED_PROMPTS:
        return PREDEFINED_PROMPTS[name]["template"]
    return None


def list_predefined_prompts() -> Dict[str, str]:
    """
    列出所有预定义的提示词模板

    Returns:
        模板名称到描述的映射
    """
    return {
        name: info["description"]
        for name, info in PREDEFINED_PROMPTS.items()
    }


if __name__ == "__main__":
    # 测试代码
    from .models import ImageSubmitRequest

    print("=" * 60)
    print("测试提示词生成")
    print("=" * 60)

    # 创建测试请求
    test_request = ImageSubmitRequest(
        image_url="https://example.com/test.jpg",
        image_name="测试图片",
        image_background="这是一幅唐代人物画",
        historical_context="虢国夫人游春图，唐代张萱作品",
        artistic_notes="工笔重彩，线条流畅",
        language="zh"
    )

    # 测试三个级别
    for level in ["min", "middle", "max"]:
        print(f"\n{'=' * 60}")
        print(f"级别: {level}")
        print("=" * 60)
        prompt = build_prompt(test_request, level)
        print(prompt)
        print(f"\n提示词长度: {len(prompt)} 字符")

    # 测试自定义提示词
    print(f"\n{'=' * 60}")
    print("自定义提示词测试")
    print("=" * 60)
    custom_request = ImageSubmitRequest(
        image_url="https://example.com/test.jpg",
        custom_prompt="<image>\n请识别这张图片中的所有文字"
    )
    custom_prompt = build_prompt(custom_request, "max")
    print(custom_prompt)
