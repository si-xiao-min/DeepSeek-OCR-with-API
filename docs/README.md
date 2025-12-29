# DeepSeek-OCR API 服务

基于 DeepSeek-OCR 的多模态 OCR 识别 HTTP API 服务，支持人物画、山水画、文物等中国传统文化图片的智能识别。

## 功能特点

- ✅ **三级识别接口**：`/image/min`、`/image/middle`、`/image/max` 满足不同精度需求
- 🎨 **中国文化优化**：针对人物画、山水画、文物等场景特别优化
- 🚀 **异步任务队列**：支持高并发请求，异步处理任务
- 🔒 **API Key 认证**：简单有效的访问控制
- 🌐 **宽松 CORS 策略**：允许跨域访问，易于前端集成
- 📊 **任务状态查询**：实时查询任务进度和结果

## 快速开始

### 1. 环境准备

```bash
# 确保已安装 CUDA 11.8
nvidia-smi

# 激活 conda 环境
conda activate deepseek-ocr
```

### 2. 安装依赖

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm/api_service
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制配置模板
cp .env.template .env

# 编辑配置文件（根据实际情况修改）
vim .env
```

关键配置项：
```bash
MODEL_PATH=/hy-tmp/deepseek-ocr-model/  # 模型路径
DEEPSEEK_OCR_API_KEYS=1228               # API Key
API_PORT=8000                            # API 端口
GPU_MEMORY_UTILIZATION=0.75              # GPU 内存利用率
```

### 4. 启动服务

```bash
# 方式1：使用启动脚本（推荐）
./start.sh

# 方式2：直接运行
python run_server.py
```

服务启动后会显示：
```
========================================
🚀 DeepSeek-OCR API 服务已启动
========================================
📍 API地址: http://0.0.0.0:8000
📚 文档地址: http://0.0.0.0:8000/docs
========================================
```

### 5. 访问 API 文档

打开浏览器访问：`http://localhost:8000/docs`

## API 接口说明

### 提交 OCR 任务

#### Min 级别（纯文本）

```bash
curl -X POST "http://localhost:8000/image/min" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 1228" \
  -d '{
    "image_url": "https://suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com/test.jpg",
    "language": "zh"
  }'
```

#### Middle 级别（文本 + 版面信息）

```bash
curl -X POST "http://localhost:8000/image/middle" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 1228" \
  -d '{
    "image_url": "https://suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com/test.jpg",
    "image_name": "测试图片",
    "language": "zh"
  }'
```

#### Max 级别（详细识别 + 定位）

```bash
curl -X POST "http://localhost:8000/image/max" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 1228" \
  -d '{
    "image_url": "https://suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com/test.jpg",
    "image_name": "虢国夫人游春图",
    "historical_context": "唐代人物画，张萱作品",
    "artistic_notes": "工笔重彩，线条流畅",
    "language": "zh"
  }'
```

### 查询任务状态

```bash
curl -X GET "http://localhost:8000/tasks/{task_id}" \
  -H "X-API-Key: 1228"
```

响应示例：
```json
{
  "success": true,
  "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "completed",
  "queue_position": 0,
  "result": {
    "text": "识别的文本内容",
    "layout_info": [...],
    "entities": [...],
    "regions": [...]
  },
  "created_at": "2025-12-28T12:00:00",
  "completed_at": "2025-12-28T12:00:10"
}
```

### 健康检查

```bash
curl "http://localhost:8000/health"
```

## 请求参数说明

### 通用参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_url` | string | ✅ | 图片 URL（阿里云 OSS） |
| `image_name` | string | ❌ | 图片名称/标题 |
| `image_background` | string | ❌ | 背景信息描述 |
| `custom_prompt` | string | ❌ | 自定义提示词 |
| `historical_context` | string | ❌ | 历史背景（适用于人物画、文物） |
| `artistic_notes` | string | ❌ | 艺术技法说明（适用于画作） |
| `language` | string | ❌ | 语言代码（zh/en，默认 zh） |

### 响应级别差异

| 级别 | 返回字段 | 适用场景 |
|------|---------|---------|
| `min` | `text` | 纯文本提取 |
| `middle` | `text` + `layout_info` | 文档识别、版面分析 |
| `max` | `text` + `layout_info` + `entities` + `regions` | 人物画、文物等详细识别 |

## Node.js 前端集成示例

```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:8000';
const API_KEY = '1228';

// 提交 OCR 任务
async function submitOCRTask(imageUrl, level = 'middle') {
  try {
    const response = await axios.post(
      `${API_BASE}/image/${level}`,
      {
        image_url: imageUrl,
        image_name: '测试图片',
        language: 'zh'
      },
      {
        headers: {
          'X-API-Key': API_KEY,
          'Content-Type': 'application/json'
        }
      }
    );
    return response.data; // { task_id, status, ... }
  } catch (error) {
    console.error('提交任务失败:', error.response?.data || error.message);
    throw error;
  }
}

// 查询任务状态（轮询）
async function pollTaskStatus(taskId, maxAttempts = 60) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await axios.get(
        `${API_BASE}/tasks/${taskId}`,
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      const { status, result, error } = response.data;

      if (status === 'completed') {
        return result;
      } else if (status === 'failed') {
        throw new Error(error || '任务处理失败');
      }

      // 等待 1 秒后重试
      await new Promise(resolve => setTimeout(resolve, 1000));
    } catch (error) {
      console.error('查询状态失败:', error.response?.data || error.message);
      throw error;
    }
  }
  throw new Error('任务超时');
}

// 完整流程
async function processImage(imageUrl) {
  console.log('提交任务...');
  const { task_id } = await submitOCRTask(imageUrl, 'max');
  console.log('任务ID:', task_id);

  console.log('等待处理...');
  const result = await pollTaskStatus(task_id);
  console.log('识别结果:', result);

  return result;
}

// 使用示例
processImage('https://suxiaomin-tuil.oss-cn-wuhan-lr.aliyuncs.com/test.jpg')
  .then(result => {
    console.log('✅ 处理成功');
    console.log('文本:', result.text);
  })
  .catch(error => {
    console.error('❌ 处理失败:', error.message);
  });
```

## 错误码说明

| 错误码 | 说明 | 处理建议 |
|--------|------|---------|
| `UNAUTHORIZED` | API Key 无效或缺失 | 检查 X-API-Key 请求头 |
| `INVALID_REQUEST` | 请求参数无效 | 检查请求参数格式 |
| `IMAGE_DOWNLOAD_ERROR` | 图片下载失败 | 检查图片 URL 是否可访问 |
| `UNSUPPORTED_IMAGE_FORMAT` | 不支持的图片格式 | 使用 JPG/PNG/WEBP 格式 |
| `IMAGE_TOO_LARGE` | 图片过大 | 限制图片大小在 50MB 以内 |
| `DOMAIN_NOT_ALLOWED` | URL 域名不在白名单 | 检查 ALLOWED_OSS_DOMAINS 配置 |
| `GPU_OUT_OF_MEMORY` | GPU 内存不足 | 降低 gpu_memory_utilization |
| `INTERNAL_SERVER_ERROR` | 服务器内部错误 | 查看服务日志 |

## 性能优化建议

1. **GPU 内存利用率**
   - 默认：0.75（75%）
   - 如果遇到 OOM：降低到 0.5
   - 如果 GPU 内存充足：提高到 0.9

2. **并发处理**
   - 默认支持约 10 个并发任务
   - 超过部分会在队列中等待

3. **图片优化**
   - 使用适当的图片分辨率（建议 1024-2048px）
   - 压缩图片大小（建议 < 10MB）

## 故障排查

### 问题1：模型加载失败

```
❌ 模型加载失败: GPU内存不足
```

**解决方案**：
```bash
# 编辑 .env 文件
GPU_MEMORY_UTILIZATION=0.5  # 降低到 50%
```

### 问题2：API 跨域问题

已配置宽松的 CORS 策略，允许所有源访问。如果仍有问题：

```python
# 在 main.py 中检查 CORS 配置
allow_origins=["*"],  # 允许所有源
```

### 问题3：任务一直 pending

```bash
# 检查健康状态
curl http://localhost:8000/health

# 检查日志
tail -f api_service.log
```

## 配置文件说明

详细配置请参考 `.env.template` 文件，包含以下配置项：

- API 配置（地址、端口）
- 模型配置（路径、GPU 利用率）
- API Keys
- OSS 白名单
- 图片大小限制
- 任务队列配置
- 日志配置

## 许可证

本项目遵循原 DeepSeek-OCR 项目的许可证。

## 联系方式

如有问题或建议，欢迎提交 Issue。
