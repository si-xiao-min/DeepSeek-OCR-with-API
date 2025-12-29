"""
任务管理器模块
实现任务队列和状态管理
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from .ocr_processor import get_ocr_processor
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    任务数据类
    """
    task_id: str
    status: str  # pending/processing/completed/failed
    created_at: datetime
    data: Dict[str, Any]
    level: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "error_code": self.error_code,
        }


class TaskManager:
    """
    任务管理器类
    管理任务队列和状态
    """

    def __init__(self):
        """初始化任务管理器"""
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}
        self.max_tasks = config.MAX_TASKS_IN_MEMORY
        self.task_ttl = config.TASK_TTL_SECONDS
        self.worker_task: Optional[asyncio.Task] = None
        self.ocr_processor = get_ocr_processor()
        logger.info("任务管理器初始化完成")

    def submit_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        level: str = "middle"
    ) -> str:
        """
        提交任务到队列

        Args:
            task_id: 任务ID
            task_data: 任务数据
            level: 识别级别

        Returns:
            任务ID

        Raises:
            ValueError: 如果任务数量超过上限
        """
        # 检查任务数量上限
        if len(self.tasks) >= self.max_tasks:
            logger.warning(f"任务数量达到上限: {self.max_tasks}")
            raise ValueError(f"任务队列已满，请稍后重试")

        # 创建任务
        task = Task(
            task_id=task_id,
            status="pending",
            created_at=datetime.now(),
            data=task_data,
            level=level
        )

        # 保存任务
        self.tasks[task_id] = task

        # 加入队列
        self.queue.put_nowait(task_id)

        logger.info(f"任务已提交: {task_id}, 当前队列大小: {self.queue.qsize()}")
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        查询任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务信息字典，如果任务不存在返回None
        """
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        return None

    def get_queue_position(self, task_id: str) -> int:
        """
        获取任务在队列中的位置

        Args:
            task_id: 任务ID

        Returns:
            队列位置：
            - -1: 任务不存在或已完成/失败
            - 0: 任务正在处理
            - >0: 任务在队列中的位置（数字越小越靠前）
        """
        task = self.tasks.get(task_id)
        if not task:
            return -1

        if task.status == "processing":
            return 0

        if task.status == "pending":
            # 计算在队列中的位置
            queue_list = list(self.queue._queue)
            try:
                return queue_list.index(task_id)
            except ValueError:
                return -1

        return -1  # completed 或 failed

    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return self.queue.qsize()

    async def worker(self):
        """
        工作协程：持续处理队列中的任务
        """
        logger.info("任务处理worker已启动")

        while True:
            try:
                # 从队列中获取任务ID
                task_id = await self.queue.get()

                # 获取任务
                task = self.tasks.get(task_id)
                if not task:
                    logger.warning(f"任务不存在: {task_id}")
                    continue

                # 更新任务状态
                task.status = "processing"
                logger.info(f"开始处理任务: {task_id}")

                try:
                    # 调用OCR处理器
                    result = await self.ocr_processor.process(
                        task_data=task.data,
                        level=task.level
                    )

                    if result["success"]:
                        # 处理成功
                        task.status = "completed"
                        task.result = result["result"]
                        logger.info(f"任务处理成功: {task_id}")
                    else:
                        # 处理失败
                        task.status = "failed"
                        task.error = result.get("error", "Unknown error")
                        task.error_code = result.get("error_code", "UNKNOWN_ERROR")
                        logger.error(f"任务处理失败: {task_id} - {task.error}")

                except Exception as e:
                    # 捕获未预期的异常
                    task.status = "failed"
                    task.error = str(e)
                    task.error_code = "INTERNAL_ERROR"
                    logger.error(f"任务处理异常: {task_id} - {e}", exc_info=True)

                finally:
                    # 更新完成时间
                    task.completed_at = datetime.now()
                    self.queue.task_done()

                    # 触发清理检查
                    await self._cleanup_old_tasks()

            except asyncio.CancelledError:
                logger.info("worker被取消，退出")
                break
            except Exception as e:
                logger.error(f"worker发生异常: {e}", exc_info=True)
                # 继续运行，不退出

    async def start_worker(self):
        """启动worker协程"""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.worker())
            logger.info("worker协程已创建")

    async def stop_worker(self):
        """停止worker协程"""
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            logger.info("worker协程已停止")

    async def _cleanup_old_tasks(self):
        """
        清理过期任务
        删除超过TTL时间且状态为completed/failed的任务
        """
        try:
            now = datetime.now()
            expired_tasks = []

            for task_id, task in self.tasks.items():
                # 只清理已完成的任务
                if task.status in ["completed", "failed"]:
                    # 检查是否过期
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds()
                        if age > self.task_ttl:
                            expired_tasks.append(task_id)

            # 删除过期任务
            for task_id in expired_tasks:
                del self.tasks[task_id]
                logger.debug(f"已清理过期任务: {task_id}")

            if expired_tasks:
                logger.info(f"清理了 {len(expired_tasks)} 个过期任务，当前任务数: {len(self.tasks)}")

        except Exception as e:
            logger.error(f"清理任务时发生异常: {e}", exc_info=True)

    async def cleanup_all_tasks(self):
        """
        清理所有任务（用于服务重启）
        """
        count = len(self.tasks)
        self.tasks.clear()
        # 清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"已清理所有任务: {count} 个")


# 全局任务管理器实例
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """
    获取全局任务管理器实例

    Returns:
        TaskManager单例
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
