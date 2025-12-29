# DeepSeek-OCR API 服务架构文档

## 目录

- [系统概述](#系统概述)
- [整体架构](#整体架构)
- [核心模块](#核心模块)
- [数据流](#数据流)
- [技术栈](#技术栈)
- [设计决策](#设计决策)

---

## 系统概述

DeepSeek-OCR API 服务是一个基于 vLLM 的异步 OCR 识别 HTTP 服务，专门用于处理中国传统文化图片（人物画、山水画、文物等）的文本提取和结构化识别。

### 主要特性

- **异步任务队列**：使用 asyncio 实现非阻塞任务处理
- **三级识别精度**：提供 min/middle/max 三个级别的识别接口
- **双编码器架构**：SAM + CLIP 视觉编码器组合
- **动态分辨率处理**：支持不同尺寸图片的智能分块
- **单例模型管理**：全局共享模型实例，节省内存

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端请求                              │
│              (Web/Mobile/CLI Applications)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP/JSON
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 应用层                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   /image/min │  │/image/middle │  │  /image/max  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │                │
│  ┌──────▼─────────────────▼─────────────────▼───────┐      │
│  │         FastAPI 路由 + CORS 中间件                  │      │
│  └──────────────────────┬────────────────────────────┘      │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      业务逻辑层                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TaskManager (任务管理器)                 │   │
│  │   - 异步任务队列 (asyncio.Queue)                      │   │
│  │   - 任务状态管理 (pending/processing/completed)       │   │
│  │   - Worker 协程 (后台任务处理)                        │   │
│  └──────────────┬──────────────────────────────────────┘   │
│                 │                                           │
│  ┌──────────────▼──────────────────────────────────────┐   │
│  │            OCRProcessor (OCR 处理器)                  │   │
│  │   - 图片下载 (aiohttp)                                │   │
│  │   - 提示词构建 (Prompts)                              │   │
│  │   - 结果解析 (Utils)                                  │   │
│  └──────────────┬──────────────────────────────────────┘   │
└─────────────────┼───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      模型推理层                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          ModelManager (模型管理器 - 单例)             │   │
│  │   - vLLM AsyncLLMEngine                               │   │
│  │   - DeepseekOCRProcessor (图像预处理)                 │   │
│  │   - 采样参数 + N-gram 防重复处理器                    │   │
│  └──────────────┬──────────────────────────────────────┘   │
└─────────────────┼───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     DeepSeek-OCR 模型                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            DeepseekOCRForCausalLM                     │   │
│  │  ┌────────────┐         ┌────────────┐              │   │
│  │  │ SAM 编码器  │         │CLIP 编码器  │              │   │
│  │  │ (Segment   │         │ (Vision    │              │   │
│  │  │  Anything) │         │  Encoder)  │              │   │
│  │  └─────┬──────┘         └─────┬──────┘              │   │
│  │        │                     │                       │   │
│  │        └────────┬────────────┘                       │   │
│  │                 ▼                                    │   │
│  │        ┌──────────────────┐                          │   │
│  │        │ MLP 投影层       │                          │   │
│  │        │ (2048→1280D)     │                          │   │
│  │        └────────┬─────────┘                          │   │
│  │                 ▼                                    │   │
│  │        ┌──────────────────┐                          │   │
│  │        │ LLM 推理引擎      │                          │   │
│  │        │ (文本生成)       │                          │   │
│  │        └──────────────────┘                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                        GPU 显存                              │
│                  (NVIDIA A100/A800)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. FastAPI 应用层 (`main.py`)

**职责**：
- 定义 RESTful API 路由
- 处理 HTTP 请求和响应
- 配置 CORS 中间件
- 全局异常处理

**关键组件**：
```python
app = FastAPI(
    title="DeepSeek-OCR API",
    version="1.0.0",
    lifespan=lifespan,  # 生命周期管理
)

# 路由示例
@app.post("/image/min")
@app.post("/image/middle")
@app.post("/image/max")
@app.get("/tasks/{task_id}")
@app.get("/health")
```

**生命周期管理**：
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：
    # 1. 初始化 ModelManager
    # 2. 加载模型到 GPU
    # 3. 启动 TaskManager worker
    yield
    # 关闭时：
    # 1. 停止 worker
    # 2. 清理资源
```

---

### 2. TaskManager (任务管理器 - `task_manager.py`)

**职责**：
- 管理异步任务队列
- 维护任务状态
- 协调 worker 协程

**数据结构**：
```python
class Task:
    task_id: str
    status: str  # pending/processing/completed/failed
    created_at: datetime
    data: Dict[str, Any]
    level: str  # min/middle/max
    result: Optional[Dict]
    error: Optional[str]

class TaskManager:
    queue: asyncio.Queue        # 任务队列
    tasks: Dict[str, Task]      # 任务存储
    max_tasks: int              # 最大任务数
    task_ttl: int               # 任务生存时间
    worker_task: asyncio.Task   # worker 协程
```

**Worker 工作流**：
```python
async def worker(self):
    while True:
        task_id = await self.queue.get()
        task = self.tasks[task_id]
        task.status = "processing"

        try:
            result = await self.ocr_processor.process(
                task.data, task.level
            )
            task.status = "completed"
            task.result = result["result"]
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
        finally:
            await self._cleanup_old_tasks()
```

---

### 3. OCRProcessor (OCR 处理器 - `ocr_processor.py`)

**职责**：
- 下载图片
- 构建提示词
- 调用模型推理
- 解析结果

**处理流程**：
```python
async def process(self, task_data, level):
    # 1. 下载图片
    image = await self._download_image(task_data["image_url"])

    # 2. 构建提示词
    prompt = self._build_prompt(task_data, level)

    # 3. 推理
    raw_result = await self._run_inference(image, prompt)

    # 4. 格式化结果
    formatted_result = self._format_result(raw_result, level, task_data)

    return {"success": True, "result": formatted_result}
```

---

### 4. ModelManager (模型管理器 - `model_manager.py`)

**职责**：
- 管理 vLLM 模型生命周期
- 提供推理接口
- 单例模式确保全局唯一实例

**单例实现**：
```python
class ModelManager:
    _instance: Optional['ModelManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**模型初始化**：
```python
async def _load_model(self):
    # 1. 注册自定义模型
    ModelRegistry.register_model(
        "DeepseekOCRForCausalLM",
        DeepseekOCRForCausalLM
    )

    # 2. 初始化 vLLM 引擎
    engine_args = AsyncEngineArgs(
        model=config.MODEL_PATH,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_model_len=config.MAX_MODEL_LEN,
        ...
    )
    self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 3. 初始化图像处理器
    self.processor = DeepseekOCRProcessor()
```

---

### 5. DeepseekOCRForCausalLM (模型定义 - `deepseek_ocr.py`)

**职责**：
- 实现双编码器架构
- 处理视觉特征
- 文本生成推理

**架构图**：
```
输入图片
    │
    ▼
┌─────────────────────────────────────────┐
│   视觉编码层                              │
│  ┌────────────┐      ┌────────────┐    │
│  │   SAM      │      │   CLIP     │    │
│  │  Encoder   │      │  Encoder   │    │
│  └─────┬──────┘      └─────┬──────┘    │
│        │                  │            │
│        └────────┬─────────┘            │
│                 ▼                      │
│        ┌───────────────┐               │
│        │  MLP Projector│               │
│        │  (2048→1280)  │               │
│        └───────┬───────┘               │
└────────────────┼───────────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │  LLM (Text Gen)   │
        │  - Input Embed    │
        │  - Transformer    │
        │  - Output Head    │
        └───────────────────┘
```

**关键方法**：
```python
def forward(self, input_ids, pixel_values, attention_mask):
    # 1. 视觉特征提取
    image_features = self.encode_images(pixel_values)

    # 2. 特征嵌入
    inputs_embeds = self.merge_text_vision(input_ids, image_features)

    # 3. 文本生成
    outputs = self.model(inputs_embeds, attention_mask)
    return outputs
```

---

### 6. DeepseekOCRProcessor (图像预处理 - `process/image_process.py`)

**职责**：
- 图片加载和预处理
- 动态分块（crop_mode）
- Tokenization

**动态分辨率处理**：
```python
def tokenize_with_images(self, images, cropping=True):
    all_tiles = []

    for image in images:
        # 全局视图（缩小到 base_size）
        global_view = resize(image, base_size)

        # 局部视图（动态分块）
        if cropping and image.size > (image_size, image_size):
            tiles = self._crop_image(image, image_size)
            all_tiles.extend(tiles)
        else:
            all_tiles.append(global_view)

    # 2D token 组织
    features = self._encode_tiles(all_tiles)
    return self._arrange_2d_tokens(features)
```

---

## 数据流

### 1. 完整请求流程

```
[客户端]
    │ POST /image/middle
    │ { "image_url": "https://...", "language": "zh" }
    ▼
[FastAPI 路由层]
    │ 验证 API Key
    │ 生成 task_id
    ▼
[TaskManager]
    │ 创建 Task
    │ 加入 queue
    ▼
[Worker 协程]
    │ 从 queue 获取 task_id
    ▼
[OCRProcessor]
    │ 1. 下载图片 (aiohttp)
    │ 2. 构建提示词
    ▼
[ModelManager]
    │ 3. 预处理图片
    │ 4. vLLM 推理
    ▼
[DeepSeek-OCR Model]
    │ SAM + CLIP 编码
    │ LLM 生成文本
    ▼
[OCRProcessor]
    │ 5. 解析结果
    │ 6. 格式化输出
    ▼
[TaskManager]
    │ 更新 Task.status = "completed"
    │ 存储 Task.result
    ▼
[客户端]
    │ GET /tasks/{task_id}
    │ 轮询查询结果
    ▼
[返回最终结果]
```

### 2. 任务状态转换

```
创建任务
    │
    ▼
[pending] ─────► 队列等待
    │
    │ (worker 获取任务)
    ▼
[processing] ──► 推理中
    │
    ├─► 成功 ──► [completed]
    │
    └─► 失败 ──► [failed]

(completed/failed) ──► TTL 过期 ──► 清理
```

---

## 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| **Web 框架** | FastAPI | 0.115.0 |
| **ASGI 服务器** | Uvicorn | 0.32.0 |
| **数据验证** | Pydantic | 2.9.0 |
| **HTTP 客户端** | aiohttp | 3.10.0 |
| **深度学习框架** | PyTorch | 2.6.0 |
| **推理引擎** | vLLM | 0.8.5 |
| **注意力优化** | Flash Attention | 2.7.3 |
| **图像处理** | Pillow | 11.0.0 |
| **CUDA** | NVIDIA CUDA | 11.8 |

---

## 设计决策

### 1. 为什么使用异步架构？

**原因**：
- OCR 推理耗时较长（1-10 秒）
- 同步处理会导致请求阻塞
- 异步可以支持高并发

**实现**：
```python
# FastAPI 异步路由
@app.post("/image/middle")
async def submit_task(...):
    task_id = generate_task_id()
    task_manager.submit_task(task_id, task_data)
    return {"task_id": task_id, "status": "pending"}

# Worker 协程后台处理
async def worker(self):
    while True:
        task_id = await self.queue.get()
        await self._process_task(task_id)
```

---

### 2. 为什么使用任务队列？

**优势**：
- **解耦提交和处理**：客户端无需等待推理完成
- **流量控制**：避免并发过多导致 OOM
- **容错性**：任务失败不影响服务可用性
- **可观测性**：实时查询任务状态

**实现细节**：
```python
# 使用 asyncio.Queue 实现
self.queue: asyncio.Queue = asyncio.Queue()

# 任务 TTL 机制
async def _cleanup_old_tasks(self):
    for task_id, task in self.tasks.items():
        if task.status in ["completed", "failed"]:
            if (now - task.completed_at) > TTL:
                del self.tasks[task_id]
```

---

### 3. 为什么使用单例模式管理模型？

**原因**：
- GPU 显存有限，无法加载多个模型实例
- vLLM 引擎本身支持批量处理
- 全局共享提高资源利用率

**实现**：
```python
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 全局访问
model_manager = get_model_manager()
```

---

### 4. 为什么提供三级识别接口？

| 级别 | 延迟 | 输出内容 | 适用场景 |
|------|------|---------|---------|
| `min` | ~1-2s | 纯文本 | 简单文档提取 |
| `middle` | ~3-5s | 文本 + 版面 | 复杂文档识别 |
| `max` | ~5-10s | 全部信息 | 人物画、文物等 |

**设计思路**：
- 不同场景对精度和延迟的要求不同
- 给用户选择权，灵活平衡性能和效果
- 提示词工程实现差异化输出

---

### 5. 为什么使用 SAM + CLIP 双编码器？

**SAM (Segment Anything Model)**：
- 强大的区域分割能力
- 提供精细的空间信息
- 适合处理复杂版面

**CLIP (Vision Transformer)**：
- 强大的语义理解能力
- 多模态对齐训练
- 适合文本识别

**组合优势**：
```
SAM → 空间特征 (where)
CLIP → 语义特征 (what)
MLP → 融合 → LLM
```

---

## 性能指标

### 延迟

| 级别 | 平均延迟 | P50 | P95 | P99 |
|------|---------|-----|-----|-----|
| min | 1.5s | 1.2s | 2.0s | 3.5s |
| middle | 3.2s | 2.8s | 4.5s | 7.0s |
| max | 6.5s | 5.5s | 9.0s | 15.0s |

### 吞吐量

- **并发数**：约 10 个任务同时处理
- **GPU 利用率**：75-85%
- **显存占用**：约 30GB / 40GB (A100)

---

## 扩展性设计

### 1. 水平扩展

```python
# 部署多个实例
实例1: http://host1:8000
实例2: http://host2:8000
实例3: http://host3:8000

# 使用 Nginx 负载均衡
upstream deepseek_ocr_cluster {
    server host1:8000;
    server host2:8000;
    server host3:8000;
}
```

### 2. 功能扩展

- 添加新的识别级别（如 `ultra`）
- 支持批量图片处理
- 支持 PDF 文档处理
- 添加结果缓存层

---

## 安全性设计

### 1. API Key 认证

```python
async def verify_api_key(api_key: str = Header(...)):
    if api_key not in config.API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True
```

### 2. OSS 域名白名单

```python
def is_allowed_domain(url: str) -> bool:
    parsed = urlparse(url)
    for allowed_domain in config.ALLOWED_OSS_DOMAINS:
        if parsed.netloc.endswith(allowed_domain.lstrip('.')):
            return True
    return False
```

### 3. 输入验证

- 图片格式限制（JPG/PNG/WEBP）
- 图片大小限制（50MB）
- 参数类型验证（Pydantic）

---

## 监控和可观测性

### 1. 健康检查

```bash
GET /health
{
  "status": "healthy",
  "model_loaded": true,
  "queue_size": 5,
  "gpu_memory_utilization": 0.75
}
```

### 2. 日志

- 结构化日志（JSON 格式）
- 分级日志（DEBUG/INFO/WARNING/ERROR）
- 关键操作日志（任务提交/完成/失败）

### 3. 指标（建议接入 Prometheus）

- 任务处理速率
- 任务延迟分布
- GPU 利用率
- 错误率

---

## 总结

DeepSeek-OCR API 服务采用现代化的异步微服务架构，通过合理的模块划分和设计模式，实现了高性能、高可用的 OCR 识别服务。

**关键优势**：
- ✅ 异步非阻塞，支持高并发
- ✅ 三级接口，灵活满足不同需求
- ✅ 单例模型管理，资源利用率高
- ✅ 任务队列解耦，易于扩展
- ✅ 完善的错误处理和日志

**适用场景**：
- 文档数字化
- 文化遗产保护
- 艺术作品识别
- 多模态内容理解
