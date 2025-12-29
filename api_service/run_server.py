#!/usr/bin/env python
"""
DeepSeek-OCR API 服务启动脚本
"""
import sys
from pathlib import Path

# 添加父目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 使用uvicorn直接运行
if __name__ == "__main__":
    import uvicorn
    from api_service.config import config

    uvicorn.run(
        "api_service.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )
