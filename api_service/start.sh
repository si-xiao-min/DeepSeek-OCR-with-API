#!/bin/bash

# DeepSeek-OCR API 服务启动脚本

echo "========================================"
echo "  DeepSeek-OCR API 服务启动中..."
echo "========================================"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查是否存在 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  警告: .env 文件不存在"
    echo "正在从 .env.template 创建 .env 文件..."
    cp .env.template .env
    echo "✅ .env 文件已创建，请根据实际情况修改配置"
    echo ""
fi

# 加载环境变量
if [ -f ".env" ]; then
    echo "📋 加载环境变量..."
    export $(grep -v '^#' .env | xargs)
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: Python 未安装或不在PATH中"
    exit 1
fi

echo ""
echo "配置信息:"
echo "  - API 地址: ${API_HOST}:${API_PORT}"
echo "  - 模型路径: ${MODEL_PATH}"
echo "  - GPU 内存利用率: ${GPU_MEMORY_UTILIZATION}"
echo "  - 日志级别: ${LOG_LEVEL}"
echo ""
echo "========================================"
echo "🚀 启动 API 服务..."
echo "========================================"
echo ""

# 启动服务
python run_server.py
