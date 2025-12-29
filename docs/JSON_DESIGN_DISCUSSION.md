# DeepSeek-OCR 输出能力分析与 JSON 设计讨论

## 一、DeepSeek-OCR 实际能返回什么？

### 1.1 核心输出格式

从源码分析来看，DeepSeek-OCR 支持**三种主要输出模式**：

#### 模式 1：纯文本（Free OCR）
```python
prompt = "<image>\nFree OCR."
```
**输出**：纯文本，无任何格式标记
```
这是一段文档中的文字内容。
这是第二段文字。
```

#### 模式 2：Markdown 格式
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```
**输出**：标准 Markdown 格式
```markdown
# 文档标题

这是第一段内容。

## 二级标题

- 列表项1
- 列表项2

| 表格 | 列1 |
|-----|-----|
| 数据 | 值1 |
```

#### 模式 3：Grounding 格式（带定位）
```python
prompt = "<image>\n<|grounding|>OCR this image."
```
**输出**：文本 + 特殊标记（这是关键！）
```
<|ref|>title<|/ref|><|det|>[[[0, 0, 500, 50]]]<|/det|>文档标题

<|ref|>paragraph<|/ref|><|det|>[[[0, 60, 500, 200]]]<|/det|>这是段落内容

<|ref|>table<|/ref|><|det|>[[[0, 220, 400, 350]]]<|/det|>| 列1 | 列2 |
...
```

### 1.2 Grounding 格式详解

**格式结构**：
```
<|ref|>{element_type}<|/ref|><|det|>[[[x1, y1, x2, y2]]]<|/det|>{content}
```

**组成部分**：
- `<|ref|>`：元素类型标记的开始
- `{element_type}`：元素类型（title、paragraph、table、figure、equation 等）
- `<|/ref|>`：元素类型标记的结束
- `<|det|>`：定位信息的开始
- `[[[x1, y1, x2, y2]]]`：边界框坐标（归一化到 0-999）
- `<|/det|>`：定位信息的结束
- `{content}`：实际的文本内容

### 1.3 支持的元素类型（从源码推断）

根据 `run_dpsk_ocr_image.py` 的处理逻辑，模型可以识别以下元素类型：

| 元素类型 | 说明 | 示例 |
|---------|------|------|
| `title` | 标题 | 一级、二级标题 |
| `paragraph` | 段落 | 普通文本段落 |
| `table` | 表格 | 数据表格 |
| `figure` | 图表/图片 | 图片、图表 |
| `equation` | 公式 | 数学公式 |
| `list` | 列表 | 有序/无序列表 |
| `image` | 嵌入图片 | 文档中的图片 |
| `header` | 页眉 | 页眉内容 |
| `footer` | 页脚 | 页脚内容 |
| `footnote` | 脚注 | 脚注内容 |

---

## 二、模型能力的边界

### 2.1 **DeepSeek-OCR 能做的**

✅ **文本提取**：高精度文字识别
✅ **版面结构**：识别标题、段落、列表
✅ **文档元素**：表格、公式、图表
✅ **定位信息**：边界框坐标（归一化 0-999）
✅ **Markdown 转换**：保留文档结构
✅ **中文优化**：特别针对中文文档优化
✅ **动态分块**：处理超大尺寸图片
✅ **Grounding 模式**：输出带定位的文本

### 2.2 **DeepSeek-OCR 不能做的**

❌ **实体识别**：不会自动识别人名、地名、组织名
❌ **情感分析**：不会分析文本情感倾向
❌ **关键词提取**：不会自动提取关键词
❌ **摘要生成**：不会自动生成摘要（除非在 prompt 中要求）
❌ **格式完美 JSON**：模型不保证输出合法的 JSON 格式
❌ **复杂嵌套结构**：如复杂的层级关系
❌ **OCR 之外的理解**：如图片的艺术风格分析（需要额外 prompt）

### 2.3 **关键认识**

**DeepSeek-OCR 本质上是一个"语言模型"**，不是"结构化数据提取器"。

它的输出：
- 主要是**文本**（纯文本或 Markdown）
- Grounding 格式是**文本中的标记**，不是独立的 JSON
- 它的能力边界取决于 **prompt**，而不是固定的 Schema

---

## 三、基于实际能力的 JSON 设计方案

### 3.1 方案 A：忠实于模型输出（推荐）

**设计理念**：只解析模型**明确输出**的信息，不做额外假设。

#### JSON Schema
```json
{
  "text": "纯文本内容",
  "markdown": "Markdown 格式内容（如果使用 Markdown 模式）",
  "layout_info": [
    {
      "type": "title|paragraph|table|figure|equation|list",
      "content": "该元素的文本内容",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ],
  "metadata": {
    "mode": "free|markdown|grounding",
    "image_size": [width, height],
    "num_blocks": 3
  }
}
```

#### 字段说明
| 字段 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `text` | string | 纯文本内容 | 所有模式 |
| `markdown` | string | Markdown 格式 | Markdown 模式 |
| `layout_info` | array | 版面信息 | 仅 Grounding 模式 |
| `layout_info[].type` | string | 元素类型 | 解析 `<|ref|>` |
| `layout_info[].content` | string | 文本内容 | 提取文本 |
| `layout_info[].bbox` | array | 边界框 | 解析 `<|det|>` |
| `metadata.mode` | string | 输出模式 | 根据 prompt 判断 |

#### 实现示例
```python
def parse_ocr_result(raw_result: str, level: str) -> dict:
    """
    基于 DeepSeek-OCR 实际输出能力的解析器
    """
    result = {"text": raw_result}

    # 检测输出模式
    has_grounding = "<|ref|>" in raw_result and "<|det|>" in raw_result
    is_markdown = any(mark in raw_result for mark in ["#", "|", "- ", "```"])

    # 添加元数据
    result["metadata"] = {
        "mode": "grounding" if has_grounding else ("markdown" if is_markdown else "free"),
        "raw_length": len(raw_result)
    }

    # Grounding 模式：解析版面信息
    if has_grounding:
        layout_info = parse_grounding_elements(raw_result)
        result["layout_info"] = layout_info
        # 提取纯文本
        result["text"] = extract_text_from_grounding(raw_result)

    # Markdown 模式：保留原始格式
    elif is_markdown:
        result["markdown"] = raw_result
        # 可选：推断版面结构
        if level in ["middle", "max"]:
            result["layout_info"] = infer_layout_from_markdown(raw_result)

    # Free OCR 模式：只有纯文本
    # result["text"] 已经是纯文本

    return result


def parse_grounding_elements(raw_result: str) -> list:
    """
    解析 grounding 格式的元素
    """
    import re

    # 匹配 <|ref|>type<|/ref|><|det|>[[[x1,y1,x2,y2]]]<|/det|>content
    pattern = r'<\|ref\|>(?P<type>\w+)<\|/ref\|><\|det\|>\[\[\[(?P<bbox>[\d\[\]\,\s]+)\]\]\]<\|/det\|>(?P<content>[^<]+)'

    elements = []
    current_pos = 0

    for match in re.finditer(pattern, raw_result):
        element_type = match.group("type")
        bbox_str = match.group("bbox")
        content = match.group("content").strip()

        try:
            # 解析坐标
            bbox = [int(x.strip()) for x in bbox_str.replace("[", "").replace("]", "").split(",")]

            elements.append({
                "type": element_type,
                "content": content,
                "bbox": bbox
            })
        except:
            continue

    return elements if elements else None


def infer_layout_from_markdown(markdown: str) -> list:
    """
    从 Markdown 推断版面结构（启发式）
    """
    layout_info = []
    lines = markdown.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("# "):
            layout_info.append({"type": "title", "level": 1, "content": line[2:].strip()})
        elif line.startswith("## "):
            layout_info.append({"type": "subtitle", "level": 2, "content": line[3:].strip()})
        elif line.startswith("- ") or line.startswith("* "):
            layout_info.append({"type": "list_item", "content": line[2:].strip()})
        elif line.startswith("|"):
            layout_info.append({"type": "table_row", "content": line})
        else:
            layout_info.append({"type": "paragraph", "content": line})

    return layout_info
```

---

### 3.2 方案 B：增强型 JSON（不推荐）

**设计理念**：在模型输出基础上，添加额外的处理和分析。

```json
{
  "text": "...",
  "layout_info": [...],
  "entities": [
    {
      "type": "PERSON",
      "text": "张三",
      "confidence": 0.85
    }
  ],
  "summary": "文档摘要...",
  "keywords": ["关键词1", "关键词2"],
  "sentiment": "positive"
}
```

**问题**：
- ❌ DeepSeek-OCR **不会自动输出**这些信息
- ❌ 需要额外的 NLP 模型（如 NER、情感分析）
- ❌ 增加系统复杂度和延迟
- ❌ 用户可能误以为是模型直接输出的

---

### 3.3 方案 C：用户指定 Schema（实验性）

**设计理念**：在 prompt 中明确要求模型输出特定格式。

```python
prompt = f"""<image>
请执行 OCR 并以 JSON 格式返回，包含以下字段：
{{
  "text": "提取的文本",
  "layout_info": [
    {{"type": "元素类型", "content": "内容", "bbox": [x1, y1, x2, y2]}}
  ]
}}

只返回 JSON，不要其他内容。"""
```

**问题**：
- ⚠️ 模型可能不遵循格式
- ⚠️ JSON 格式可能不合法
- ⚠️ 需要 robust 的错误处理
- ⚠️ 增加 token 消耗

---

## 四、推荐的三级 API 设计

基于以上分析，我建议采用**忠实于模型输出**的三级 API：

### 4.1 Min 级别：纯文本
```json
{
  "text": "识别的纯文本内容"
}
```

**实现**：
```python
prompt = "<image>\nFree OCR."
# 或
prompt = "<image>\n请识别图片中的文字内容。"
```

### 4.2 Middle 级别：文本 + Markdown
```json
{
  "text": "识别的纯文本内容",
  "markdown": "# 标题\n\n内容...",
  "layout_info": [
    {"type": "title", "level": 1, "content": "标题"},
    {"type": "paragraph", "content": "内容"}
  ]
}
```

**实现**：
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
# 从 Markdown 推断版面结构
```

### 4.3 Max 级别：文本 + 定位信息
```json
{
  "text": "识别的纯文本内容",
  "markdown": "Markdown 格式（如果有）",
  "layout_info": [
    {
      "type": "title",
      "content": "文档标题",
      "bbox": [0, 0, 500, 50]
    },
    {
      "type": "paragraph",
      "content": "这是段落",
      "bbox": [0, 60, 500, 200]
    },
    {
      "type": "table",
      "content": "| 列1 | 列2 |\n...",
      "bbox": [0, 220, 400, 350]
    }
  ],
  "elements": {
    "tables": [...],
    "figures": [...],
    "equations": [...]
  }
}
```

**实现**：
```python
prompt = "<image>\n<|grounding|>OCR this image."
# 解析 grounding 标记
```

---

## 五、具体实现建议

### 5.1 修改 `prompts.py`

```python
def build_min_prompt(request: ImageSubmitRequest) -> str:
    """最小级别：纯文本"""
    if request.language == "zh":
        return "<image>\nFree OCR."
    return "<image>\nExtract all text from the image."

def build_middle_prompt(request: ImageSubmitRequest) -> str:
    """中间级别：Markdown 格式"""
    if request.language == "zh":
        return "<image>\n<|grounding|>将文档转换为 Markdown 格式。"
    return "<image>\n<|grounding|>Convert the document to markdown."

def build_max_prompt(request: ImageSubmitRequest) -> str:
    """最大级别：Grounding 格式（带定位）"""
    if request.language == "zh":
        prompt = "<image>\n<|grounding|>请识别图片中的所有内容，并标注版面结构。"
        if request.historical_context:
            prompt += f"\n\n历史背景：{request.historical_context}"
        if request.artistic_notes:
            prompt += f"\n\n艺术技法：{request.artistic_notes}"
    else:
        prompt = "<image>\n<|grounding|>OCR this image with layout information."
    return prompt
```

### 5.2 修改 `utils.py`

```python
def parse_ocr_result(raw_result: str, level: str) -> dict:
    """
    解析 OCR 结果（忠实于模型输出）
    """
    result = {"text": raw_result}

    # 检测输出模式
    has_grounding = "<|ref|>" in raw_result and "<|det|>" in raw_result
    has_markdown = any(mark in raw_result for mark in ["#", "|", "- ", "```"])

    # Grounding 模式
    if has_grounding:
        result["layout_info"] = parse_grounding_layout(raw_result)
        result["text"] = extract_text_from_grounding(raw_result)
        # 根据 level 添加额外信息
        if level == "max":
            result["elements"] = categorize_elements(result["layout_info"])

    # Markdown 模式
    elif has_markdown and level in ["middle", "max"]:
        result["markdown"] = raw_result
        result["layout_info"] = infer_markdown_layout(raw_result)

    return result


def parse_grounding_layout(raw_result: str) -> list:
    """解析 grounding 格式的版面信息"""
    import re

    pattern = r'<\|ref\|>(?P<type>\w+)<\|/ref\|><\|det\|>\[\[\[(?P<bbox>[\d\[\]\,\s]+)\]\]\]<\|/det\|>(?P<content>[^<]+?)(?=<\|ref\|>|$)'

    layout_info = []
    for match in re.finditer(pattern, raw_result, re.DOTALL):
        try:
            element_type = match.group("type")
            bbox_str = match.group("bbox")
            content = match.group("content").strip()

            bbox = [int(x.strip()) for x in bbox_str.replace("[", "").replace("]", "").split(",")]

            layout_info.append({
                "type": element_type,
                "content": content,
                "bbox": bbox
            })
        except:
            continue

    return layout_info if layout_info else None


def categorize_elements(layout_info: list) -> dict:
    """将元素分类到不同的类别"""
    elements = {
        "tables": [],
        "figures": [],
        "equations": [],
        "titles": [],
        "paragraphs": []
    }

    for item in layout_info or []:
        element_type = item["type"].lower()
        if "table" in element_type:
            elements["tables"].append(item)
        elif "figure" in element_type or "image" in element_type:
            elements["figures"].append(item)
        elif "equation" in element_type or "formula" in element_type:
            elements["equations"].append(item)
        elif "title" in element_type or "header" in element_type:
            elements["titles"].append(item)
        elif "paragraph" in element_type:
            elements["paragraphs"].append(item)

    # 移除空列表
    return {k: v for k, v in elements.items() if v}
```

---

## 六、总结与建议

### 6.1 核心原则

1. **忠实于模型输出**：只解析模型实际输出的信息
2. **不做过度承诺**：不要在 API 中包含模型不能提供的信息
3. **清晰的文档**：明确告诉用户每个级别的输出格式
4. **渐进式增强**：从简单开始，逐步增加复杂度

### 6.2 推荐的实现路径

**阶段 1**（立即实现）：
- ✅ Min：纯文本
- ✅ Middle：Markdown + 基础版面推断
- ✅ Max：Grounding 格式解析

**阶段 2**（可选）：
- 🔄 添加坐标归一化（0-999 → 像素坐标）
- 🔄 添加置信度评分（如果模型输出）
- 🔄 支持自定义 prompt

**阶段 3**（实验性）：
- ⚠️ 尝试让模型输出 JSON
- ⚠️ 添加后处理 NLP 模块

### 6.3 你的代码需要修改的地方

1. **`api_service/prompts.py`**：使用正确的提示词格式
2. **`api_service/utils.py`**：实现 grounding 格式解析
3. **`api_service/models.py`**：更新响应模型（移除不存在的字段）
4. **`docs/`**：更新 API 文档，说明实际的输出格式

---

**你觉得这个设计方案如何？我们是否需要进一步讨论某些细节？**
