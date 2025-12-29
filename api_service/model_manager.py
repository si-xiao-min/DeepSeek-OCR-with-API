"""
模型管理器模块
实现全局单例模式管理vLLM模型
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import torch
from PIL import Image
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

# 添加父目录到系统路径，以便导入deepseek_ocr模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from deepseek_ocr import DeepseekOCRForCausalLM
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from .config import config

# 配置日志
logger = logging.getLogger(__name__)


class ModelManager:
    """
    模型管理器类（单例模式）
    负责模型的加载、推理和健康检查
    """

    _instance: Optional['ModelManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> 'ModelManager':
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化模型管理器（只执行一次）"""
        # 防止重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self.engine: Optional[AsyncLLMEngine] = None
        self.processor: Optional[DeepseekOCRProcessor] = None
        self.model_loaded = False
        self.load_lock = asyncio.Lock()
        self._load_start_time: Optional[float] = None

        logger.info("模型管理器已创建（单例模式）")

    async def _load_model(self) -> None:
        """
        内部方法：加载模型
        需要在异步上下文中调用
        """
        if self.model_loaded:
            return

        async with self.load_lock:
            # 双重检查
            if self.model_loaded:
                return

            try:
                self._load_start_time = time.time()
                logger.info(f"开始加载模型: {config.MODEL_PATH}")

                # 配置CUDA环境
                if torch.version.cuda == '11.8':
                    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

                os.environ['VLLM_USE_V1'] = '0'

                # 注册模型
                ModelRegistry.register_model(
                    "DeepseekOCRForCausalLM",
                    DeepseekOCRForCausalLM
                )
                logger.info("模型注册完成")

                # 初始化引擎
                engine_args = AsyncEngineArgs(
                    model=config.MODEL_PATH,
                    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                    block_size=256,
                    max_model_len=config.MAX_MODEL_LEN,
                    enforce_eager=False,
                    trust_remote_code=config.TRUST_REMOTE_CODE,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
                )

                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                logger.info("vLLM引擎初始化完成")

                # 初始化图片处理器
                self.processor = DeepseekOCRProcessor()
                logger.info("图片处理器初始化完成")

                self.model_loaded = True
                load_time = time.time() - self._load_start_time
                logger.info(f"✅ 模型加载成功！耗时: {load_time:.2f}秒")

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ GPU内存不足: {e}")
                logger.error(f"建议：降低GPU_MEMORY_UTILIZATION（当前值: {config.GPU_MEMORY_UTILIZATION}）")
                raise RuntimeError(f"GPU内存不足，请降低gpu_memory_utilization配置") from e

            except Exception as e:
                logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
                raise RuntimeError(f"模型加载失败: {e}") from e

    async def initialize(self) -> None:
        """
        公共方法：初始化模型（如果尚未加载）
        可以在服务启动时调用
        """
        if not self.model_loaded:
            await self._load_model()

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model_loaded and self.engine is not None

    async def generate(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> str:
        """
        生成文本（异步方法）

        Args:
            image: PIL Image对象（RGB格式）
            prompt: 提示词（应包含<image>标记）
            max_tokens: 最大生成token数
            temperature: 温度参数（0.0表示贪婪解码）

        Returns:
            生成的文本

        Raises:
            RuntimeError: 如果模型未加载或推理失败
        """
        if not self.is_loaded():
            raise RuntimeError("模型尚未加载，请先调用 initialize()")

        try:
            # 准备logits处理器（防止重复）
            logits_processors = [
                NoRepeatNGramLogitsProcessor(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822}  # <td>, </td>
                )
            ]

            # 配置采样参数
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                logits_processors=logits_processors,
                skip_special_tokens=False,
            )

            # 生成请求ID
            request_id = f"api-request-{int(time.time() * 1000)}"

            # 准备请求数据
            if image and '<image>' in prompt:
                # 预处理图片
                from .config import config as api_config
                image_features = self.processor.tokenize_with_images(
                    images=[image],
                    bos=True,
                    eos=True,
                    cropping=True  # 使用动态裁剪
                )
                request = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_features}
                }
            elif prompt:
                request = {"prompt": prompt}
            else:
                raise ValueError("prompt不能为空")

            # 执行推理
            full_text = ""
            async for request_output in self.engine.generate(
                request, sampling_params, request_id
            ):
                if request_output.outputs:
                    full_text = request_output.outputs[0].text

            return full_text

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"推理时GPU内存不足: {e}")
            raise RuntimeError("GPU内存不足，请降低并发数或gpu_memory_utilization") from e

        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            raise RuntimeError(f"推理失败: {e}") from e

    def get_load_time(self) -> Optional[float]:
        """获取模型加载耗时（秒）"""
        if self._load_start_time:
            return time.time() - self._load_start_time
        return None

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            包含健康状态信息的字典
        """
        return {
            "model_loaded": self.is_loaded(),
            "model_path": config.MODEL_PATH,
            "gpu_memory_utilization": config.GPU_MEMORY_UTILIZATION,
            "max_model_len": config.MAX_MODEL_LEN,
            "load_time_seconds": self.get_load_time(),
        }


# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    获取全局模型管理器实例（线程安全）

    Returns:
        ModelManager单例实例
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
