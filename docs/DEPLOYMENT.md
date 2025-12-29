# DeepSeek-OCR API 服务部署指南

本文档提供详细的部署步骤和配置说明。

## 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [配置说明](#配置说明)
- [启动服务](#启动服务)
- [验证部署](#验证部署)
- [生产环境部署](#生产环境部署)
- [故障排查](#故障排查)

---

## 环境要求

### 硬件要求

- **GPU**: NVIDIA GPU，显存 >= 40GB（推荐 A100/A800）
- **内存**: >= 64GB
- **磁盘**: >= 100GB 可用空间（用于存储模型）

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **CUDA**: 11.8
- **Python**: 3.12.9
- **Conda**: Miniconda 或 Anaconda

### 网络要求

- 能够访问阿里云 OSS（图片下载）
- 开放 API 端口（默认 8000）

---

## 安装步骤

### 1. 检查 CUDA

```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 CUDA 工具链
nvcc --version
```

预期输出：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.8    |
...
```

### 2. 创建 Conda 环境

```bash
# 创建环境
conda create -n deepseek-ocr python=3.12.9 -y

# 激活环境
conda activate deepseek-ocr
```

### 3. 安装 PyTorch 和 vLLM

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

# 安装 vLLM (下载对应的 wheel 文件)
# 从 https://github.com/vllm-project/vllm/releases 下载
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 或直接从 pip 安装（如果可用）
pip install vllm==0.8.5
```

### 4. 安装 Flash Attention

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

### 5. 安装 API 服务依赖

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm/api_service
pip install -r requirements.txt
```

### 6. 下载模型

```bash
# 方式1：从 HuggingFace 自动下载（推荐）
export MODEL_PATH=deepseek-ai/DeepSeek-OCR

# 方式2：使用已下载的本地模型
export MODEL_PATH=/hy-tmp/deepseek-ocr-model/
```

---

## 配置说明

### 1. 创建配置文件

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm/api_service
cp .env.template .env
```

### 2. 编辑配置文件

```bash
vim .env
```

### 3. 关键配置项

#### API 配置

```bash
API_HOST=0.0.0.0          # 监听地址（0.0.0.0 表示所有网卡）
API_PORT=8000             # 监听端口
```

#### 模型配置

```bash
MODEL_PATH=/hy-tmp/deepseek-ocr-model/  # 模型路径
GPU_MEMORY_UTILIZATION=0.75             # GPU 内存利用率
MAX_MODEL_LEN=8192                      # 最大 token 长度
TRUST_REMOTE_CODE=true                  # 信任远程代码
```

#### API Key 配置

```bash
# 单个 API Key
DEEPSEEK_OCR_API_KEYS=1228

# 多个 API Key（逗号分隔）
DEEPSEEK_OCR_API_KEYS=dso_key1,dso_key2,dso_key3
```

#### OSS 白名单配置

```bash
# 支持通配符（.example.com 匹配所有子域名）
ALLOWED_OSS_DOMAINS=.oss-cn-wuhan-lr.aliyuncs.com,suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com
```

#### 图片配置

```bash
MAX_IMAGE_SIZE_MB=50         # 最大图片大小
IMAGE_DOWNLOAD_TIMEOUT=30    # 下载超时时间（秒）
```

#### 任务队列配置

```bash
MAX_TASKS_IN_MEMORY=1000    # 内存中最大任务数
TASK_TTL_SECONDS=3600       # 任务 TTL（1小时）
```

#### 日志配置

```bash
LOG_LEVEL=INFO              # 日志级别（DEBUG/INFO/WARNING/ERROR）
LOG_FILE=api_service.log    # 日志文件路径
```

---

## 启动服务

### 方式1：使用启动脚本（推荐）

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm/api_service
./start.sh
```

### 方式2：直接运行

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm/api_service

# 加载环境变量
source .env

# 启动服务
python run_server.py
```

### 方式3：使用 Uvicorn

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

uvicorn api_service.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info
```

---

## 验证部署

### 1. 检查服务启动日志

服务启动成功后会显示：

```
========================================
🚀 DeepSeek-OCR API 服务已启动
========================================
📍 API地址: http://0.0.0.0:8000
📚 文档地址: http://0.0.0.0:8000/docs
========================================
```

### 2. 健康检查

```bash
curl http://localhost:8000/health
```

预期响应：

```json
{
  "status": "healthy",
  "model_loaded": true,
  "queue_size": 0,
  "gpu_memory_utilization": 0.75,
  "version": "1.0.0"
}
```

### 3. 测试 API 接口

```bash
curl -X POST "http://localhost:8000/image/min" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 1228" \
  -d '{
    "image_url": "https://suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com/test.jpg",
    "language": "zh"
  }'
```

### 4. 访问 API 文档

打开浏览器：`http://localhost:8000/docs`

---

## 生产环境部署

### 1. 使用 Systemd 管理

创建 systemd 服务文件：

```bash
sudo vim /etc/systemd/system/deepseek-ocr-api.service
```

内容：

```ini
[Unit]
Description=DeepSeek-OCR API Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/DeepSeek-OCR-master/DeepSeek-OCR-vllm/api_service
Environment="CUDA_VISIBLE_DEVICES=0"
EnvironmentFile=/path/to/.env
ExecStart=/usr/bin/python3 run_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
# 重载 systemd 配置
sudo systemctl daemon-reload

# 启用服务（开机自启）
sudo systemctl enable deepseek-ocr-api

# 启动服务
sudo systemctl start deepseek-ocr-api

# 查看状态
sudo systemctl status deepseek-ocr-api

# 查看日志
sudo journalctl -u deepseek-ocr-api -f
```

### 2. 使用 Nginx 反向代理

安装 Nginx：

```bash
sudo apt install nginx -y
```

配置 Nginx：

```bash
sudo vim /etc/nginx/sites-available/deepseek-ocr-api
```

内容：

```nginx
upstream deepseek_ocr_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://deepseek_ocr_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS 头（可选，如果应用层已配置）
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type, X-API-Key";
    }
}
```

启用配置：

```bash
# 创建符号链接
sudo ln -s /etc/nginx/sites-available/deepseek-ocr-api /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

### 3. 配置 HTTPS（使用 Let's Encrypt）

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx -y

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo certbot renew --dry-run
```

### 4. 监控和日志

#### 日志轮转

创建 logrotate 配置：

```bash
sudo vim /etc/logrotate.d/deepseek-ocr-api
```

内容：

```
/path/to/api_service/api_service.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 your-username your-username
}
```

#### 监控 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 记录到日志
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 5 > gpu_monitor.log
```

---

## 故障排查

### 问题1：服务启动失败

**症状**：
```
❌ 模型加载失败: GPU内存不足
```

**解决方案**：

1. 降低 GPU 内存利用率
   ```bash
   # 编辑 .env
   GPU_MEMORY_UTILIZATION=0.5
   ```

2. 减小最大 token 长度
   ```bash
   MAX_MODEL_LEN=4096
   ```

3. 检查 GPU 显存
   ```bash
   nvidia-smi
   ```

### 问题2：API 请求超时

**症状**：任务一直处于 pending 状态

**解决方案**：

1. 检查服务健康状态
   ```bash
   curl http://localhost:8000/health
   ```

2. 查看服务日志
   ```bash
   tail -f api_service.log
   ```

3. 检查 GPU 状态
   ```bash
   nvidia-smi
   ```

### 问题3：跨域问题

**症状**：浏览器报 CORS 错误

**解决方案**：

已配置宽松的 CORS 策略（允许所有源），如果仍有问题：

1. 检查 CORS 中间件配置
2. 检查 Nginx 配置（如果使用反向代理）
3. 确保请求头包含 `X-API-Key`

### 问题4：图片下载失败

**症状**：
```json
{
  "error": "图片下载失败",
  "error_code": "IMAGE_DOWNLOAD_ERROR"
}
```

**解决方案**：

1. 检查图片 URL 是否可访问
2. 检查 OSS 白名单配置
3. 检查网络连接

---

## 性能优化建议

1. **GPU 内存利用率**：根据实际显存调整（0.5-0.9）
2. **并发控制**：默认支持约 10 个并发任务
3. **图片优化**：使用适当分辨率的图片（建议 1024-2048px）
4. **日志级别**：生产环境使用 INFO 或 WARNING

---

## 备份和恢复

### 备份配置文件

```bash
tar -czf deepseek-ocr-config-backup.tar.gz .env config.py
```

### 恢复配置

```bash
tar -xzf deepseek-ocr-config-backup.tar.gz
```

---

## 更新和升级

### 更新代码

```bash
cd /path/to/DeepSeek-OCR
git pull
```

### 更新依赖

```bash
cd api_service
pip install -r requirements.txt --upgrade
```

### 重启服务

```bash
sudo systemctl restart deepseek-ocr-api
```

---

## 安全建议

1. **API Key 管理**：使用强随机 API Key，定期更换
2. **OSS 白名单**：严格限制允许的域名
3. **防火墙**：只开放必要的端口
4. **HTTPS**：生产环境使用 HTTPS
5. **日志审计**：定期检查访问日志

---

## 联系支持

如有问题，请查看：
- 服务日志：`api_service.log`
- 系统日志：`journalctl -u deepseek-ocr-api`
- GPU 状态：`nvidia-smi`
