# DeepSeek-OCR API 服务项目结构说明

## 项目概述

DeepSeek-OCR API 服务是一个基于 DeepSeek-OCR 多模态模型的高性能 OCR（光学字符识别）服务。该项目采用异步架构，提供 RESTful API 接口，支持三种不同级别的识别精度，特别针对中文文档、书法、绘画等中国文化内容的识别进行了优化。

### 核心特性

- **双编码器架构**：结合 SAM（Segment Anything Model）和 CLIP 视觉编码器
- **动态分辨率处理**：支持图像分块裁剪，处理大尺寸图像
- **三级识别模式**：min（纯文本）、middle（文本+版面）、max（详细识别）
- **异步任务队列**：基于 asyncio 的任务调度和处理
- **vLLM 推理引擎**：高性能模型推理，支持 GPU 加速

---

## 项目目录结构

```
deepseek-ocr-api-service/
├── api_service/          # API 服务包
├── deepencoder/          # 视觉编码器包
├── process/              # 图像处理包
├── docs/                 # 文档目录
├── config.py             # 根级配置文件
├── deepseek_ocr.py       # 主模型实现
├── requirements.txt      # Python 依赖
├── README.md             # 项目说明
├── CLAUDE.md             # Claude Code 指导文档
└── .env.template         # 环境变量模板
```

---

## 根目录文件说明

### `config.py`
**功能**：模型运行参数配置

**主要配置项**：
- `BASE_SIZE = 1024`：全局视图的基础图像尺寸
- `IMAGE_SIZE = 640`：局部裁剪块的图像尺寸
- `CROP_MODE = True`：是否启用动态裁剪模式
- `MIN_CROPS = 2`：最小裁剪块数量
- `MAX_CROPS = 6`：最大裁剪块数量（GPU 内存受限时可降低）
- `MAX_CONCURRENCY = 100`：最大并发处理数量
- `NUM_WORKERS = 64`：图像预处理工作线程数
- `MODEL_PATH`：模型文件路径
- `PROMPT`：默认 OCR 提示词模板
- `TOKENIZER`：分词器实例

**常用提示词模板**：
```python
# 文档转 Markdown
'<image>\n<|grounding|>Convert the document to markdown.'

# 自由 OCR（无版面信息）
'<image>\nFree OCR.'

# 图像 OCR（带版面）
'<image>\n<|grounding|>OCR this image.'

# 图表解析
'<image>\nParse the figure.'

# 详细描述
'<image>\nDescribe this image in detail.'

# 对象定位
'<image>\nLocate <|ref|>xxxx<|/ref|> in the image.'
```

### `deepseek_ocr.py`
**功能**：DeepSeek-OCR 主模型实现，兼容 vLLM 框架

**核心类**：
- `DeepseekOCRProcessingInfo`：处理信息配置类
  - 计算图像 token 数量
  - 获取图像尺寸配置

- `DeepseekOCRDummyInputsBuilder`：虚拟输入构建器
  - 用于模型初始化和 profiling

- `DeepseekOCRMultiModalProcessor`：多模态处理器
  - 处理图像和文本的融合输入
  - 管理 `<image>` token 替换

- `DeepseekOCRForCausalLM`：主模型类
  - 初始化双编码器（SAM + CLIP）
  - 实现前向传播逻辑
  - 处理全局视图和局部裁剪块
  - 集成语言模型（DeepSeek-V2/V3）

**架构特点**：
1. **双编码器融合**：SAM 提取空间特征，CLIP 提取语义特征
2. **动态分块处理**：大图自动裁剪成多个 640×640 块
3. **2D 空间标记**：使用 `image_newline` 和 `view_seperator` 保留空间关系
4. **权重映射**：自动将 HuggingFace 权重映射到 vLLM 格式

---

## `api_service/` 包说明

API 服务包是整个项目的核心业务逻辑层，提供 RESTful API 接口和任务管理。

### `main.py`
**功能**：FastAPI 主应用入口

**核心职责**：
- 应用生命周期管理（启动/关闭）
- 路由定义和端点注册
- 中间件配置（CORS、异常处理）
- API 文档生成

**主要端点**：
| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 |
| `/image/min` | POST | 提交最小级别 OCR 任务 |
| `/image/middle` | POST | 提交中间级别 OCR 任务 |
| `/image/max` | POST | 提交最大级别 OCR 任务 |
| `/tasks/{task_id}` | GET | 查询任务状态 |

**启动流程**：
1. 配置验证和打印
2. 初始化模型管理器
3. 加载 DeepSeek-OCR 模型
4. 启动任务处理 worker
5. 监听 API 请求

### `model_manager.py`
**功能**：vLLM 模型管理器（单例模式）

**核心类**：`ModelManager`

**主要方法**：
- `initialize()`：初始化模型（异步加载）
- `generate(image, prompt, ...)`：执行 OCR 推理
- `is_loaded()`：检查模型是否已加载
- `health_check()`：获取模型健康状态

**技术细节**：
- 使用 AsyncLLMEngine 支持异步推理
- 注册自定义模型到 vLLM ModelRegistry
- 配置 CUDA 环境和 Triton 路径
- 集成 NoRepeatNGramLogitsProcessor 防止重复生成
- 支持动态裁剪模式（`cropping=True`）

### `task_manager.py`
**功能**：异步任务队列管理器

**核心类**：
- `Task`：任务数据类
  - 字段：task_id, status, data, level, result, error, created_at, completed_at

- `TaskManager`：任务管理器
  - 维护任务队列（asyncio.Queue）
  - 任务状态管理（pending/processing/completed/failed）
  - 自动清理过期任务（TTL 机制）
  - 队列位置查询

**Worker 机制**：
```python
async def worker():
    while True:
        task_id = await queue.get()
        # 处理任务
        result = await ocr_processor.process(...)
        # 更新状态
        # 清理过期任务
```

### `ocr_processor.py`
**功能**：OCR 处理器，协调整个识别流程

**核心类**：`OCRProcessor`

**处理流程**：
1. **下载图像**：从 URL 下载图片（OSS 白名单验证）
2. **构建提示词**：根据级别和用户输入构建 prompt
3. **执行推理**：调用 ModelManager 进行 OCR
4. **格式化结果**：解析原始输出，添加元数据

**三级输出格式**：
- `min`：纯文本
- `middle`：文本 + layout_info（段落、标题等）
- `max`：文本 + layout_info + entities + regions（详细定位）

### `models.py`
**功能**：Pydantic 数据模型定义

**请求模型**：
- `ImageSubmitRequest`：图像提交请求
  - 字段：image_url, image_name, image_background, custom_prompt, historical_context, artistic_notes, language
  - 验证：URL 格式、语言代码

**响应模型**：
- `TaskSubmitResponse`：任务提交响应
- `TaskStatusResponse`：任务状态查询响应
- `OCRResultData`：OCR 识别结果数据
- `ErrorResponse`：错误响应
- `HealthResponse`：健康检查响应

### `config.py`
**功能**：API 服务配置管理

**配置项**：
| 类别 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| API | API_HOST | 0.0.0.0 | 监听地址 |
| API | API_PORT | 8080 | 监听端口 |
| 模型 | MODEL_PATH | - | 模型路径 |
| 模型 | GPU_MEMORY_UTILIZATION | 0.75 | GPU 内存利用率 |
| 模型 | MAX_MODEL_LEN | 8192 | 最大模型长度 |
| 安全 | API_KEYS | ["1228"] | API 密钥列表 |
| OSS | ALLOWED_OSS_DOMAINS | - | 允许的 OSS 域名 |
| 图像 | MAX_IMAGE_SIZE_MB | 50 | 最大图片大小 |
| 任务 | MAX_TASKS_IN_MEMORY | 1000 | 最大任务数 |
| 任务 | TASK_TTL_SECONDS | 3600 | 任务存活时间 |

### `prompts.py`
**功能**：三级提示词模板管理

**核心函数**：
- `build_min_prompt()`：最小级别（纯文本）
- `build_middle_prompt()`：中间级别（JSON 格式，包含版面信息）
- `build_max_prompt()`：最大级别（详细识别，支持 grounding）
- `build_prompt()`：统一入口，支持自定义提示词
- `get_predefined_prompt()`：获取预定义模板

**预定义模板**：
- `chinese_portrait`：中国人物画识别
- `chinese_landscape`：中国山水画识别
- `chinese_calligraphy`：中国书法识别
- `cultural_relic`：文物识别
- `document_ocr`：文档 OCR

### `utils.py`
**功能**：工具函数集合

**核心函数**：
- `generate_task_id()`：生成唯一任务 ID（UUID4）
- `validate_url()`：验证 URL 格式
- `is_domain_allowed()`：检查域名白名单
- `download_image()`：异步下载图片（支持大小检查、格式验证）
- `parse_ocr_result()`：解析和格式化 OCR 结果
- `get_error_code_from_exception()`：异常到错误码映射
- `setup_logging()`：配置日志系统

**错误码映射**：
| 错误码 | 异常类型 |
|--------|----------|
| INVALID_REQUEST | ValueError |
| IMAGE_DOWNLOAD_ERROR | aiohttp.ClientError |
| UNSUPPORTED_IMAGE_FORMAT | UnidentifiedImageError |
| IMAGE_TOO_LARGE | RuntimeError |
| INTERNAL_SERVER_ERROR | 其他异常 |

### `auth.py`
**功能**：API Key 认证模块

**核心函数**：
- `verify_api_key()`：强制 API Key 验证（FastAPI 依赖注入）
- `verify_api_key_optional()`：可选 API Key 验证
- `generate_api_key()`：生成随机 API Key
- `get_api_key_info()`：获取 API Key 信息（脱敏）

**验证流程**：
1. 检查请求头 `X-API-Key`
2. 验证是否在配置的 API_KEYS 列表中
3. 返回 True 或抛出 HTTPException（401）

### `run_server.py`
**功能**：服务启动脚本

**作用**：
- 简单的 uvicorn 启动封装
- 用于直接运行服务（开发/生产环境）

---

## `deepencoder/` 包说明

视觉编码器包，实现双编码器架构（SAM + CLIP）。

### `sam_vary_sdpa.py`
**功能**：SAM (Segment Anything Model) ViT-B 编码器

**核心类**：
- `ImageEncoderViT`：SAM 图像编码器
  - 基于 Vision Transformer (ViT)
  - 支持窗口注意力（window_size=14）
  - 支持相对位置编码
  - 输出 1024 维特征

**主要组件**：
- `PatchEmbed`：图像分块嵌入（16×16 patch）
- `Block`：Transformer 块
  - LayerNorm
  - Multi-head Attention（支持窗口注意力）
  - MLP（GELU 激活）
  - 残差连接
- `Attention`：多头自注意力层
  - 使用 Scaled Dot-Product Attention
  - 支持相对位置编码

**网络结构**：
```
Input (1024×1024×3)
    ↓
Patch Embed (16×16, 768-dim)
    ↓
Positional Encoding
    ↓
12 × Transformer Blocks (window_attn + global_attn)
    ↓
Neck (Conv + LayerNorm)
    ↓
Conv2D: 256 → 512 (stride=2)
    ↓
Conv2D: 512 → 1024 (stride=2)
    ↓
Output: 1024-dim features
```

**特点**：
- 全局注意力索引：[2, 5, 8, 11]
- 下采样总倍数：16（patch）× 4（conv）= 64
- 最终特征图：16×16×1024

### `clip_sdpa.py`
**功能**：CLIP ViT-L/14 视觉编码器

**核心类**：
- `CLIPVisionEmbeddings`：CLIP 视觉嵌入层
  - Class token + Patch embeddings
  - 可学习的位置编码
  - 支持位置编码插值（适应不同分辨率）

- `NoTPAttention`：无张量并行的注意力层
  - 支持 Flash Attention
  - 使用 GELU 激活

- `NoTPTransformerBlock`：Transformer 块
- `NoTPTransformer`：24 层 Transformer
- `VitModel`：完整的 ViT 模型

**配置参数**：
```python
vit_model_cfg = {
    "num_layers": 24,
    "hidden_size": 1024,
    "num_heads": 16,
    "ffn_hidden_size": 4096,
    "image_size": 224,
    "patch_size": 14,
    "use_flash_attn": False
}
```

**网络结构**：
```
Input (224×224×3)
    ↓
Patch Embed (14×14, 1024-dim)
    ↓
Add Class Token + Positional Encoding
    ↓
Layer Normalization
    ↓
24 × Transformer Blocks
    ↓
Output: 257 × 1024 (1 class token + 256 patch tokens)
```

**特点**：
- 输入可接收 SAM 的 patch embeds（双编码器协同）
- 支持 fp32 LayerNorm（可选）
- 快速 GELU 激活函数

### `build_linear.py`
**功能**：MLP 投影器，融合双编码器特征

**核心类**：`MlpProjector`

**支持的投影类型**：
1. `linear`：单层线性映射（2048 → 1280）
2. `mlp_gelu`：多层 MLP + GELU
3. `downsample_mlp_gelu`：下采样 + MLP
4. `normlayer_downsample_mlp_gelu`：LayerNorm + 下采样 + MLP
5. `low_high_hybrid_split_mlp_gelu`：高低频分离双路径
6. `hybrid_split_feature_mlp_gelu`：特征分割双路径
7. `low_high_split_mlp_gelu`：高低层独立处理

**当前配置**：
```python
projector_type = "linear"
input_dim = 2048  # SAM 1024 + CLIP 1024
n_embed = 1280    # 匹配语言模型维度
```

**作用**：
- 将 SAM (1024-dim) 和 CLIP (1024-dim) 特征拼接
- 投影到 1280-dim，匹配 DeepSeek 语言模型嵌入维度
- 简单线性映射，高效且有效

---

## `process/` 包说明

图像处理包，负责图像预处理和 N-gram 重复防止。

### `image_process.py`
**功能**：图像预处理和 tokenization

**核心函数**：
- `find_closest_aspect_ratio()`：查找最接近的宽高比
- `count_tiles()`：计算图像需要裁剪成多少块
- `dynamic_preprocess()`：动态裁剪图像

**核心类**：`DeepseekOCRProcessor`

**主要方法**：
- `tokenize_with_images()`：将图像转换为模型输入
  1. 分割 prompt（按 `<image>` 标记）
  2. 处理全局视图（pad 到 base_size）
  3. 动态裁剪局部视图（如果需要）
  4. 编码文本（添加 bos/eos）
  5. 生成图像 tokens（占位符）
  6. 返回完整的输入数据

**输出格式**：
```python
[
    input_ids,           # 文本 + 图像 token IDs
    pixel_values,        # 全局视图像素 [1, 3, 1024, 1024]
    images_crop,         # 局部裁剪块 [N, 3, 640, 640]
    images_seq_mask,     # 区分文本/图像 token
    images_spatial_crop, # 裁剪空间信息 [num_width_tiles, num_height_tiles]
    num_image_tokens,    # 图像 token 数量
    image_shapes         # 原始图像尺寸
]
```

**处理流程示例**：
```
原图: 2000×1500
    ↓
动态裁剪: 3×2 = 6 个 640×640 块
    ↓
全局视图: pad 到 1024×1024
    ↓
局部视图: 6 个 640×640 块
    ↓
特征提取: SAM + CLIP
    ↓
投影: 2048 → 1280
    ↓
空间标记: 添加 image_newline 和 view_seperator
```

### `ngram_norepeat.py`
**功能**：N-gram 重复防止 Logits 处理器

**核心类**：`NoRepeatNGramLogitsProcessor`

**参数**：
- `ngram_size`：N-gram 大小（默认 30）
- `window_size`：搜索窗口大小（默认 90）
- `whitelist_token_ids`：白名单 token IDs（如 `<td>`, `</td>`）

**工作原理**：
1. 在生成过程中，检查最后 (ngram_size-1) 个 token
2. 在滑动窗口（window_size）内搜索相同前缀
3. 禁止生成会导致重复的 token
4. 白名单 token 不受限制

**示例**：
```
已生成: "这是一段文字，这是"
窗口内发现: "这是一段文字，这是" + "一段"
禁止生成: "一段"（会导致重复）
```

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         客户端请求                                    │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                                   │
│  - 验证 API Key (auth.py)                                            │
│  - 验证请求数据 (models.py)                                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Task Manager (task_manager.py)                                      │
│  - 生成任务 ID (utils.py)                                           │
│  - 加入任务队列                                                       │
│  - 返回队列位置                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Worker (异步处理)                                                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  OCR Processor (ocr_processor.py)                              │  │
│  │  1. 下载图像 (utils.py: download_image)                       │  │
│  │  2. 构建提示词 (prompts.py: build_prompt)                      │  │
│  │  3. 执行推理 (model_manager.py: generate)                      │  │
│  │  4. 格式化结果 (utils.py: parse_ocr_result)                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Model Manager (model_manager.py)                                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  vLLM Engine (AsyncLLMEngine)                                  │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  DeepseekOCRForCausalLM (deepseek_ocr.py)                │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │  图像处理 (process/image_process.py)               │  │  │  │
│  │  │  │  - 动态裁剪 (dynamic_preprocess)                   │  │  │  │
│  │  │  │  - 全局视图 (pad to 1024)                         │  │  │  │
│  │  │  │  - 局部视图 (crop 640×640)                         │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │  SAM 编码器 (deepencoder/sam_vary_sdpa.py)         │  │  │  │
│  │  │  │  - ViT-B/16 (12 layers)                            │  │  │  │
│  │  │  │  - 输出: 1024-dim features                         │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │  CLIP 编码器 (deepencoder/clip_sdpa.py)            │  │  │  │
│  │  │  │  - ViT-L/14 (24 layers)                            │  │  │  │
│  │  │  │  - 输出: 1024-dim features                         │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │  投影器 (deepencoder/build_linear.py)              │  │  │  │
│  │  │  │  - MLP Projector (2048 → 1280)                    │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │  语言模型 (DeepSeek-V2/V3)                         │  │  │  │
│  │  │  │  - 文本生成                                       │  │  │  │
│  │  │  │  - N-gram 重复防止                                 │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  更新任务状态 (completed/failed)                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         客户端轮询结果                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## API 使用示例

### 提交 OCR 任务（Min 级别）

```bash
curl -X POST "http://localhost:8080/image/min" \
  -H "X-API-Key: 1228" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "image_name": "测试图片",
    "language": "zh"
  }'
```

**响应**：
```json
{
  "success": true,
  "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "pending",
  "queue_position": 0,
  "message": "任务已提交"
}
```

### 查询任务状态

```bash
curl -X GET "http://localhost:8080/tasks/{task_id}" \
  -H "X-API-Key: 1228"
```

**响应（completed）**：
```json
{
  "success": true,
  "task_id": "...",
  "status": "completed",
  "queue_position": -1,
  "result": {
    "text": "识别的文本内容"
  },
  "error": null,
  "error_code": null,
  "created_at": "2025-12-29T12:00:00Z",
  "completed_at": "2025-12-29T12:00:10Z"
}
```

### Max 级别（带详细上下文）

```bash
curl -X POST "http://localhost:8080/image/max" \
  -H "X-API-Key: 1228" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/painting.jpg",
    "image_name": "虢国夫人游春图",
    "image_background": "唐代人物画，描绘虢国夫人游春场景",
    "historical_context": "唐代张萱作品，描绘杨贵妃姐妹虢国夫人等游春场景",
    "artistic_notes": "工笔重彩，线条流畅，色彩艳丽",
    "language": "zh"
  }'
```

---

## 环境要求

### 硬件要求
- **GPU**：NVIDIA GPU，支持 CUDA 11.8
- **显存**：建议 ≥ 16GB（可调整 `GPU_MEMORY_UTILIZATION`）
- **内存**：建议 ≥ 32GB

### 软件依赖
```
torch==2.6.0
torchvision==0.21.0
vllm==0.8.5
flash-attn==2.7.3
transformers
fastapi
uvicorn
aiohttp
pillow
pydantic
python-multipart
```

### Conda 环境创建
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## 配置建议

### GPU 内存受限场景
1. 降低 `GPU_MEMORY_UTILIZATION`（0.5-0.7）
2. 减少 `MAX_CROPS`（4-6）
3. 降低 `MAX_CONCURRENCY`（10-50）

### 精度要求
- **文档 OCR**：使用 Middle 级别
- **文化图片识别**：使用 Max 级别 + 历史背景信息
- **简单文字提取**：使用 Min 级别

### 性能优化
1. 启用 prefix caching（如果 vLLM 支持）
2. 调整 `MAX_MODEL_LEN` 根据实际需求
3. 增加 `NUM_WORKERS` 加速图像预处理

---

## 故障排查

### 常见错误

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| GPU内存不足 | GPU_MEMORY_UTILIZATION 过高 | 降低到 0.5-0.7 |
| 模型加载失败 | MODEL_PATH 错误 | 检查模型路径 |
| 图片下载失败 | URL 不在白名单 | 检查 ALLOWED_OSS_DOMAINS |
| 任务队列已满 | 并发任务过多 | 增大 MAX_TASKS_IN_MEMORY |
| API Key 验证失败 | API Key 错误 | 检查 X-API-Key 请求头 |

---

## 总结

DeepSeek-OCR API 服务是一个功能完整、架构清晰的多模态 OCR 服务，具有以下优势：

1. **高性能**：基于 vLLM 的异步推理引擎
2. **高精度**：双编码器架构 + 动态分辨率处理
3. **易扩展**：模块化设计，支持自定义提示词
4. **易部署**：完整的 Docker/Conda 部署方案
5. **针对性强**：特别优化中国文化内容识别

适用于文档数字化、文化遗存保护、智能内容理解等多种场景。
