# DeepSeek-OCR API 服务迁移问题总结

本文档记录了从原始 DeepSeek-OCR 项目迁移到独立 API 服务仓库时遇到的问题及其解决方案。

---

## 目录

- [问题概览](#问题概览)
- [核心问题：config.py 文件错误](#1-核心问题configpy-文件错误)
- [TOKENIZER 对象引用错误](#2tokenizer-对象引用错误)
- [Conda 环境未激活](#3conda-环境未激活)
- [GPU 内存占用问题](#4gpu-内存占用问题)
- [根本原因分析](#根本原因分析)
- [经验教训](#经验教训)
- [最佳实践](#最佳实践)

---

## 问题概览

在迁移和启动 DeepSeek-OCR API 服务的过程中，主要遇到了以下问题：

| # | 问题 | 错误信息 | 严重程度 |
|---|------|---------|---------|
| 1 | `config.py` 文件内容错误 | 导入失败 | 🔴 严重 |
| 2 | TOKENIZER 类型错误 | `AttributeError: 'str' object has no attribute 'padding_side'` | 🔴 严重 |
| 3 | Conda 环境未激活 | `ModuleNotFoundError: No module named 'vllm'` | 🟡 中等 |
| 4 | GPU 内存被占用 | `CUDA out of memory` | 🟡 中等 |

---

## 1. 核心问题：config.py 文件错误

### 问题描述

在创建项目辅助文件（.gitignore、LICENSE 等）时，**误写**了根目录的 `config.py` 文件，将其替换成了 API 服务配置内容，导致模型配置参数丢失。

### 错误内容

**我错误创建的内容：**
```python
"""
API服务配置管理模块
从环境变量读取配置，提供默认值和验证
"""
import os
from typing import List
from pathlib import Path

class Config:
    """API服务配置类"""
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8080"))
    # ... 更多 API 配置
```

### 正确内容

**应该包含的模型配置：**
```python
# DeepSeek-OCR 模型配置
# 定义图像处理和模型推理所需的参数

# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100  # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64  # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = '/hy-tmp/deepseek-ocr-model'  # change to your model path

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py;
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py;
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = ''
OUTPUT_PATH = ''

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
```

### 影响范围

- ❌ `process/image_process.py` 无法导入配置参数
- ❌ `deepseek_ocr.py` 无法导入 `IMAGE_SIZE`, `BASE_SIZE` 等
- ❌ 整个项目无法正常初始化

---

## 2. TOKENIZER 对象引用错误

### 问题描述

即使 config.py 存在，如果 TOKENIZER 被定义为字符串而非实际对象，会导致属性访问错误。

### 错误信息

```
Traceback (most recent call last):
  File "/root/deepseek-ocr-api-service/process/image_process.py", line 149, in __init__
    self.tokenizer.padding_side = 'left'
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'padding_side'
```

### 错误原因

**错误的定义方式：**
```python
# ❌ 错误：TOKENIZER 是字符串
TOKENIZER = "deepseek-ai/DeepSeek-OCR"
```

**正确的定义方式：**
```python
# ✅ 正确：TOKENIZER 是实际对象
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
```

### 代码依赖关系

`process/image_process.py` 中的 `DeepseekOCRProcessor` 类：

```python
class DeepseekOCRProcessor(ProcessorMixin):
    def __init__(
        self,
        tokenizer: LlamaTokenizerFast = TOKENIZER,  # 期望对象，不是字符串
        # ... 其他参数
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'  # ← 这里失败
```

---

## 3. Conda 环境未激活

### 问题描述

直接使用系统 Python 运行服务，导致找不到 `vllm` 模块。

### 错误信息

```
Traceback (most recent call last):
  File "/root/deepseek-ocr-api-service/api_service/run_server.py", line 16, in <module>
    uvicorn.run(
  ...
  File "/root/deepseek-ocr-api-service/api_service/model_manager.py", line 15, in <module>
    from vllm import AsyncLLMEngine, SamplingParams
ModuleNotFoundError: No module named 'vllm'
```

### 原因分析

- `vllm`、`flash-attn` 等依赖安装在 conda 环境 `deepseek-ocr` 中
- 直接使用 `python run_server.py` 会使用系统默认 Python
- 系统环境中没有安装这些深度学习依赖

### 解决方案

**启动服务时必须激活环境：**
```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr
cd /root/deepseek-ocr-api-service/api_service
python run_server.py
```

**或者在启动脚本中添加：**
```bash
#!/bin/bash
# start.sh

# 激活 conda 环境
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

# 启动服务
python run_server.py
```

---

## 4. GPU 内存占用问题

### 问题描述

首次启动时遇到 GPU 内存不足错误，但实际上 GPU 有 24GB，模型只需要约 17GB。

### 错误信息

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 906.00 MiB.
GPU 0 has a total capacity of 23.69 GiB of which 775.00 MiB is free.
Process 2789321 has 10.13 GiB memory in use.
Process 2813720 has 12.80 GiB memory in use.
```

### 原因分析

**不是真正的内存不足，而是有其他进程占用：**
- Process 2789321: 10.13 GiB
- Process 2813720: 12.80 GiB
- 总计约 23GB，几乎占满 24GB

**可能是：**
1. 之前启动失败的模型实例没有正确清理
2. 其他服务正在使用 GPU
3. Jupyter Notebook 或其他交互式会话占用

### 解决方案

**检查 GPU 使用情况：**
```bash
nvidia-smi
```

**等待进程释放内存或手动终止：**
```bash
# 查看占用 GPU 的 Python 进程
ps aux | grep python

# 如果确认可以终止，使用 kill 命令
kill <PID>
```

**验证 GPU 已释放：**
```bash
nvidia-smi
# 应该看到 "No running processes found" 或很小的内存占用
```

---

## 根本原因分析

### 项目结构理解不足

**原始项目结构：**
```
/hy-tmp/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/
├── config.py              # ← 模型配置（被遗漏）
├── deepseek_ocr.py
├── process/
│   ├── __init__.py
│   ├── image_process.py   # 依赖 config.py
│   └── ngram_norepeat.py
├── deepencoder/
│   ├── __init__.py
│   ├── build_linear.py
│   ├── clip_sdpa.py
│   └── sam_vary_sdpa.py
└── api_service/
    ├── config.py          # ← API 配置（不同文件）
    ├── main.py
    ├── model_manager.py
    └── ...
```

### 两个 config.py 的区别

| 特性 | 根目录 `config.py` | `api_service/config.py` |
|------|-------------------|----------------------|
| **用途** | 模型配置 | API 服务配置 |
| **内容** | IMAGE_SIZE, TOKENIZER, PROMPT 等 | API_PORT, API_KEYS 等 |
| **导入者** | process/, deepseek_ocr.py | api_service/ 模块 |
| **初始化时机** | 模块加载时 | 服务启动时 |
| **类型** | 全局变量 | Config 类 |

### 迁移时的疏漏

**错误的操作流程：**
1. ✅ 复制了 `api_service/` 目录
2. ✅ 复制了 `process/` 目录
3. ✅ 复制了 `deepencoder/` 目录
4. ✅ 复制了 `deepseek_ocr.py`
5. ❌ **忘记**复制根目录的 `config.py`
6. ❌ 后来创建 `.env.template` 时**误写**了 `config.py`

**应该的操作流程：**
```bash
# 完整复制所有必要文件
cp /hy-tmp/DeepSeek-OCR/.../config.py /root/.../config.py
cp /hy-tmp/DeepSeek-OCR/.../deepseek_ocr.py /root/.../
cp -r /hy-tmp/DeepSeek-OCR/.../process /root/.../
cp -r /hy-tmp/DeepSeek-OCR/.../api_service /root/.../
```

---

## 经验教训

### 1. 配置文件要明确命名

避免使用通用名称，使用更具描述性的文件名：

```
config.py              # → model_config.py
api_config.py          # → api_service_config.py
```

### 2. 迁移时检查依赖关系

**使用工具检查导入依赖：**
```bash
# 查找所有导入 config 的文件
grep -r "from config import" --include="*.py"

# 查找所有导入 TOKENIZER 的文件
grep -r "TOKENIZER" --include="*.py"
```

### 3. 关键文件要备份

```bash
# 迁移前备份
tar -czf deepseek-ocr-backup.tar.gz /hy-tmp/DeepSeek-OCR/

# 对比文件差异
diff /hy-tmp/.../config.py /root/.../config.py
```

### 4. 分步验证启动

**不要一次性启动整个服务，而是逐步验证：**

```bash
# 第 1 步：验证环境
python -c "import vllm; print('vLLM OK')"
python -c "import torch; print('PyTorch OK', torch.__version__)"

# 第 2 步：验证配置
python -c "from config import TOKENIZER; print(type(TOKENIZER))"
# 输出应该是: <class 'transformers.models.llama.tokenization_llama.LlamaTokenizerFast'>

# 第 3 步：验证模型导入
python -c "from deepseek_ocr import DeepseekOCRForCausalLM; print('Model OK')"

# 第 4 步：验证处理器
python -c "from process.image_process import DeepseekOCRProcessor; print('Processor OK')"

# 第 5 步：启动服务
python run_server.py
```

### 5. 使用版本控制避免丢失

```bash
# 添加所有文件到 Git
git add config.py process/ deepencoder/ deepseek_ocr.py

# 提交前检查状态
git status

# 查看即将提交的内容
git diff --cached

# 确认无误后提交
git commit -m "feat: add model files"
```

---

## 最佳实践

### 迁移检查清单

- [ ] **环境准备**
  - [ ] 创建/激活正确的 conda 环境
  - [ ] 验证依赖包版本
  - [ ] 检查 CUDA 版本兼容性

- [ ] **文件复制**
  - [ ] 列出所有需要复制的文件
  - [ ] 检查导入依赖关系
  - [ ] 复制配置文件（注意同名文件）
  - [ ] 复制源代码文件
  - [ ] 复制资源文件（模型权重等）

- [ ] **配置调整**
  - [ ] 更新路径配置
  - [ ] 更新端口配置
  - [ ] 更新 API Keys
  - [ ] 检查环境变量

- [ ] **验证测试**
  - [ ] 测试导入
  - [ ] 测试配置加载
  - [ ] 测试模型加载
  - [ ] 测试 API 端点

### 调试技巧

**1. 快速定位导入错误：**
```python
# test_imports.py
import sys

print("Testing imports...")
print("=" * 60)

try:
    from config import IMAGE_SIZE, BASE_SIZE, TOKENIZER
    print("✅ config.py: OK")
    print(f"   IMAGE_SIZE = {IMAGE_SIZE}")
    print(f"   TOKENIZER type = {type(TOKENIZER)}")
except Exception as e:
    print(f"❌ config.py: {e}")

try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    print("✅ deepseek_ocr.py: OK")
except Exception as e:
    print(f"❌ deepseek_ocr.py: {e}")

try:
    from process.image_process import DeepseekOCRProcessor
    print("✅ process/image_process.py: OK")
except Exception as e:
    print(f"❌ process/image_process.py: {e}")

print("=" * 60)
print("Done.")
```

**2. 检查 GPU 状态：**
```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查找占用 GPU 的进程
fuser -v /dev/nvidia*
```

**3. 清理僵尸进程：**
```bash
# 查找 Python 进程
ps aux | grep python | grep -v grep

# 终止指定进程
kill <PID>

# 如果无法终止，强制杀死
kill -9 <PID>
```

### 文档记录

**迁移过程中应该记录：**
1. 原始项目路径和版本
2. 迁移的目的和范围
3. 修改的文件列表
4. 遇到的问题和解决方案
5. 环境配置信息
6. 启动和验证步骤

**本文档本身就是这种记录的一部分，方便未来参考。**

---

## 总结

本次迁移问题的核心是**遗漏了关键的模型配置文件**，导致了一连串的启动失败。通过逐步调试和对比原始项目，最终找到了问题根源并成功解决。

**关键要点：**
1. 🔍 迁移时要全面检查依赖关系
2. 📝 同名文件要特别注意其内容和用途
3. ✅ 分步验证比一次性启动更可靠
4. 📚 详细记录问题和解决方案有助于未来参考

---

## 附录

### 相关文档

- [README.md](./README.md) - 项目说明文档
- [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md) - 部署指南
- [docs/architecture.md](./docs/architecture.md) - 架构文档

### 有用的命令

```bash
# 检查导入依赖
grep -r "from config import" --include="*.py" -n

# 检查文件差异
diff original_file.py new_file.py

# 监控 GPU 使用
watch -n 1 nvidia-smi

# 测试环境配置
python -c "import sys; print(sys.path)"

# 查找进程
ps aux | grep python

# 端口检查
netstat -tunlp | grep 8080
```

---

**文档版本:** 1.0
**最后更新:** 2025-12-29
**作者:** Claude Code & sixiaomin
