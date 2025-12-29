"""
DeepSeek-OCR API 服务主应用
提供RESTful API接口进行图片识别
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 添加父目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import config
from .models import (
    ImageSubmitRequest,
    TaskSubmitResponse,
    TaskStatusResponse,
    ErrorResponse,
    HealthResponse,
)
from .auth import verify_api_key
from .model_manager import get_model_manager
from .task_manager import get_task_manager
from .utils import generate_task_id, setup_logging

# 配置日志
setup_logging(config.LOG_LEVEL, config.LOG_FILE)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时初始化模型和worker，关闭时清理资源
    """
    # 启动阶段
    logger.info("=" * 60)
    logger.info("DeepSeek-OCR API 服务启动中...")
    logger.info("=" * 60)

    # 打印配置信息
    config.validate()
    config.print_config()

    # 初始化模型管理器
    model_manager = get_model_manager()
    logger.info("正在加载模型...")
    try:
        await model_manager.initialize()
        logger.info("✅ 模型加载成功")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise RuntimeError(f"模型加载失败: {e}")

    # 初始化任务管理器并启动worker
    task_manager = get_task_manager()
    await task_manager.start_worker()
    logger.info("✅ 任务处理worker已启动")

    logger.info("=" * 60)
    logger.info("🚀 DeepSeek-OCR API 服务已启动")
    logger.info(f"📍 API地址: http://{config.API_HOST}:{config.API_PORT}")
    logger.info(f"📚 文档地址: http://{config.API_HOST}:{config.API_PORT}/docs")
    logger.info("=" * 60)

    yield

    # 关闭阶段
    logger.info("DeepSeek-OCR API 服务关闭中...")
    await task_manager.stop_worker()
    logger.info("✅ 任务处理worker已停止")
    logger.info("DeepSeek-OCR API 服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="DeepSeek-OCR API",
    description="基于DeepSeek-OCR的多模态OCR识别服务",
    version="1.0.0",
    lifespan=lifespan,
)


# 配置CORS中间件（宽松策略）
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # ["*"] 允许所有源
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,  # ["*"] 允许所有方法
    allow_headers=config.CORS_ALLOW_HEADERS,  # ["*"] 允许所有头
    expose_headers=config.CORS_EXPOSE_HEADERS,  # ["*"] 暴露所有响应头
)


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "detail": str(exc) if config.LOG_LEVEL == "DEBUG" else None
        }
    )


# 根路径
@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "DeepSeek-OCR API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# 健康检查接口
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    健康检查接口

    返回服务健康状态，包括：
    - 服务状态
    - 模型是否加载
    - 当前队列大小
    - GPU内存利用率配置
    - API版本
    """
    model_manager = get_model_manager()
    task_manager = get_task_manager()

    health_info = await model_manager.health_check()

    return HealthResponse(
        status="healthy" if model_manager.is_loaded() else "unhealthy",
        model_loaded=model_manager.is_loaded(),
        queue_size=task_manager.get_queue_size(),
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        version="1.0.0"
    )


# 提交OCR任务接口 - Min级别
@app.post(
    "/image/min",
    response_model=TaskSubmitResponse,
    summary="提交最小级别OCR任务",
    description="提交一个最小级别的OCR识别任务，只返回纯文本，不包含结构化信息",
    tags=["OCR"]
)
async def submit_min_task(
    request: ImageSubmitRequest,
    verified: bool = Depends(verify_api_key)
):
    """
    提交最小级别OCR任务

    - **image_url**: 图片URL（必填，阿里云OSS）
    - **image_name**: 图片名称（可选）
    - **image_background**: 背景信息（可选）
    - **custom_prompt**: 自定义提示词（可选）
    - **language**: 语言代码（默认zh）

    返回任务ID和状态，可以通过 /tasks/{task_id} 查询结果
    """
    try:
        # 生成任务ID
        task_id = generate_task_id()

        # 转换请求数据为字典
        task_data = request.model_dump()

        # 提交任务
        task_manager = get_task_manager()
        task_manager.submit_task(task_id, task_data, level="min")

        # 获取队列位置
        queue_position = task_manager.get_queue_position(task_id)

        return TaskSubmitResponse(
            success=True,
            task_id=task_id,
            status="pending",
            queue_position=queue_position,
            message="任务已提交"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"提交任务失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="提交任务失败"
        )


# 提交OCR任务接口 - Middle级别
@app.post(
    "/image/middle",
    response_model=TaskSubmitResponse,
    summary="提交中间级别OCR任务",
    description="提交一个中间级别的OCR识别任务，返回文本和基础版面信息",
    tags=["OCR"]
)
async def submit_middle_task(
    request: ImageSubmitRequest,
    verified: bool = Depends(verify_api_key)
):
    """
    提交中间级别OCR任务

    返回文本 + 基础版面信息（段落、标题等）
    """
    try:
        task_id = generate_task_id()
        task_data = request.model_dump()

        task_manager = get_task_manager()
        task_manager.submit_task(task_id, task_data, level="middle")

        queue_position = task_manager.get_queue_position(task_id)

        return TaskSubmitResponse(
            success=True,
            task_id=task_id,
            status="pending",
            queue_position=queue_position,
            message="任务已提交"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"提交任务失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="提交任务失败"
        )


# 提交OCR任务接口 - Max级别
@app.post(
    "/image/max",
    response_model=TaskSubmitResponse,
    summary="提交最大级别OCR任务",
    description="提交一个最大级别的OCR识别任务，返回文本、详细版面、实体识别和区域定位",
    tags=["OCR"]
)
async def submit_max_task(
    request: ImageSubmitRequest,
    verified: bool = Depends(verify_api_key)
):
    """
    提交最大级别OCR任务

    特别针对中国文化图片（人物画、山水画、文物等），提供详细识别和定位
    """
    try:
        task_id = generate_task_id()
        task_data = request.model_dump()

        task_manager = get_task_manager()
        task_manager.submit_task(task_id, task_data, level="max")

        queue_position = task_manager.get_queue_position(task_id)

        return TaskSubmitResponse(
            success=True,
            task_id=task_id,
            status="pending",
            queue_position=queue_position,
            message="任务已提交"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"提交任务失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="提交任务失败"
        )


# 查询任务状态接口
@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="查询任务状态",
    description="根据任务ID查询OCR任务的执行状态和结果",
    tags=["Tasks"]
)
async def get_task_status(
    task_id: str,
    verified: bool = Depends(verify_api_key)
):
    """
    查询任务状态

    返回任务的当前状态：
    - **pending**: 任务在队列中等待处理
    - **processing**: 任务正在处理中
    - **completed**: 任务处理完成，result字段包含识别结果
    - **failed**: 任务处理失败，error字段包含错误信息
    """
    task_manager = get_task_manager()
    task_info = task_manager.get_task(task_id)

    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {task_id}"
        )

    # 获取队列位置
    queue_position = task_manager.get_queue_position(task_id)

    # 构建响应
    response = TaskStatusResponse(
        success=True,
        task_id=task_info["task_id"],
        status=task_info["status"],
        queue_position=queue_position,
        result=task_info.get("result"),
        error=task_info.get("error"),
        error_code=task_info.get("error_code"),
        created_at=task_info["created_at"],
        completed_at=task_info.get("completed_at")
    )

    return response


# 主函数
def main():
    """主函数：启动API服务"""
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,  # 生产环境不要启用reload
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
