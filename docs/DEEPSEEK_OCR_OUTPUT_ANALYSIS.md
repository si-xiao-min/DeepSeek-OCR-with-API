# DeepSeek-OCR 输出格式分析文档

## 问题背景

你提出的问题：
> DeepSeek-OCR 返回的东西究竟是什么？它是不是只能返回 Markdown 或者纯文本？如果是，那么我得到的 JSON 究竟是怎么得到的，为什么永远没有 layout_info？

## 核心答案

### DeepSeek-OCR 模型**只返回文本**

DeepSeek-OCR 是一个**多模态语言模型**，它的输出**永远是文本格式**，具体形式取决于你使用的提示词（prompt）：

| 提示词示例 | 输出格式 |
|-----------|---------|
| `<image>\nFree OCR.` | 纯文本 |
| `<image>\n<|grounding|>Convert the document to markdown.` | Markdown 格式 |
| `<image>\n<|grounding|>OCR this image.` | 带特殊定位标记的文本 |

### 模型支持的输出格式

从源码分析，DeepSeek-OCR 支持以下三种输出模式：

#### 1. 纯文本模式（Free OCR）
```python
prompt = "<image>\nFree OCR."
```
**输出示例**：
```
这是一段文档中的文字内容。
没有版面信息，只有文本。
```

#### 2. Markdown 模式（文档转换）
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```
**输出示例**：
```markdown
# 标题

这是一段文字。

## 子标题

- 列表项1
- 列表项2
```

#### 3. Grounding 模式（带定位信息）
```python
prompt = "<image>\n<|grounding|>OCR this image."
```
**输出示例**：
```
<|ref|>title<|/ref|><|det|>[[[0, 0, 100, 50]]]<|/det|>这是标题
<|ref|>paragraph<|/ref|><|det|>[[[0, 60, 200, 150]]]<|/det|>这是段落内容
```

**关键**：即使是 grounding 模式，定位信息也是**嵌入在文本中的特殊标记**，而不是独立的 JSON 结构。

---

## 你的 API 服务问题分析

### 问题根源

查看你的 API 服务代码（`api_service/utils.py` 第 228-266 行）：

```python
def parse_ocr_result(raw_result: str, level: str) -> dict:
    result = {"text": raw_result}

    # 尝试解析JSON（如果输出是JSON格式）
    if raw_result.strip().startswith('{'):
        try:
            import json
            parsed = json.loads(raw_result)

            # 根据级别过滤字段
            if level == "min":
                if "text" in parsed:
                    result["text"] = parsed["text"]
            elif level == "middle":
                result["text"] = parsed.get("text", raw_result)
                if "layout_info" in parsed:
                    result["layout_info"] = parsed["layout_info"]
            elif level == "max":
                result = parsed

        except json.JSONDecodeError:
            # JSON解析失败，返回原始文本
            logger.debug("JSON解析失败，返回原始文本")
            result["text"] = raw_result

    return result
```

**问题所在**：

1. **模型不输出 JSON**：DeepSeek-OCR 模型**不会自动输出 JSON 格式**
2. **代码假设模型会输出 JSON**：`parse_ocr_result` 函数假设模型返回的是 JSON 字符串
3. **永远解析失败**：因为模型输出的是 Markdown 或纯文本，不满足 `raw_result.strip().startswith('{')` 条件
4. **只返回 `text` 字段**：解析失败时，只返回 `{"text": raw_result}`，所以 `layout_info` 永远是 `None`

### 为什么会这样设计？

你的 API 服务代码参考了其他 OCR 系统（如 GOT-OCR、MinerU），这些系统可能确实输出 JSON 格式的结构化数据。但 **DeepSeek-OCR 采用了不同的设计理念**：

| 其他 OCR 系统 | DeepSeek-OCR |
|--------------|--------------|
| 输出结构化 JSON | 输出 Markdown/纯文本 |
| 强制版面结构 | 自然文本表达 |
| 程序解析优先 | 人类阅读优先 |

---

## 如何获得 layout_info？

### 方案 1：解析 Grounding 标记（推荐）

如果需要版面信息，应该使用 grounding 模式的提示词，然后解析特殊标记：

```python
import re
import json

def parse_grounding_output(raw_result: str) -> dict:
    """
    解析带 grounding 标记的输出
    """
    # 匹配 <|ref|>type<|/ref|><|det|>coordinates<|/det|> 格式
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = re.findall(pattern, raw_result, re.DOTALL)

    layout_info = []
    text_parts = []
    last_end = 0

    for match in matches:
        element_type, coord_str = match
        try:
            # 提取文本内容
            match_start = raw_result.find(match[0], last_end)
            text_before = raw_result[last_end:match_start].strip()
            if text_before:
                text_parts.append(text_before)

            # 解析坐标
            coordinates = eval(coord_str)
            if isinstance(coordinates, str):
                coordinates = eval(coordinates)

            layout_info.append({
                "type": element_type,
                "content": text_before,
                "bbox": coordinates[0] if isinstance(coordinates, list) else coordinates
            })

            last_end = raw_result.find(match[1], last_end) + len(match[1])
        except:
            continue

    # 添加剩余文本
    remaining_text = raw_result[last_end:].strip()
    if remaining_text:
        text_parts.append(remaining_text)

    return {
        "text": "\n".join(text_parts),
        "layout_info": layout_info if layout_info else None
    }
```

### 方案 2：使用 JSON 提示词（不推荐）

可以在提示词中要求模型输出 JSON：

```python
prompt = """<image>
请执行 OCR 并以 JSON 格式返回结果：
{
  "text": "提取的文本",
  "layout_info": [
    {"type": "paragraph", "content": "段落内容", "bbox": [x1, y1, x2, y2]}
  ]
}"""
```

**缺点**：
- 模型可能不遵循 JSON 格式
- 即使输出 JSON，格式也可能不一致
- 会增加 token 消耗

### 方案 3：Markdown 解析（折中方案）

如果使用 Markdown 模式，可以解析 Markdown 结构：

```python
def parse_markdown_layout(markdown_text: str) -> dict:
    """
    从 Markdown 中推断版面结构
    """
    layout_info = []
    lines = markdown_text.split('\n')
    current_y = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('# '):
            layout_info.append({
                "type": "title",
                "content": line[2:],
                "level": 1
            })
        elif line.startswith('## '):
            layout_info.append({
                "type": "subtitle",
                "content": line[3:],
                "level": 2
            })
        elif line.startswith('- '):
            layout_info.append({
                "type": "list_item",
                "content": line[2:],
                "level": 1
            })
        else:
            layout_info.append({
                "type": "paragraph",
                "content": line
            })

    return {
        "text": markdown_text,
        "layout_info": layout_info
    }
```

---

## 修改建议

### 修改 `api_service/utils.py`

将 `parse_ocr_result` 函数替换为：

```python
def parse_ocr_result(raw_result: str, level: str) -> dict:
    """
    解析 OCR 结果

    Args:
        raw_result: 模型原始输出（文本/Markdown/grounding 格式）
        level: 识别级别

    Returns:
        格式化后的结果字典
    """
    result = {"text": raw_result}

    # 检测输出格式
    has_grounding = "<|ref|>" in raw_result and "<|det|>" in raw_result
    is_json = raw_result.strip().startswith('{')

    # JSON 格式（尝试解析）
    if is_json:
        try:
            import json
            parsed = json.loads(raw_result)
            if level == "min":
                result["text"] = parsed.get("text", raw_result)
            elif level == "middle":
                result["text"] = parsed.get("text", raw_result)
                result["layout_info"] = parsed.get("layout_info")
            elif level == "max":
                result = parsed
            return result
        except json.JSONDecodeError:
            pass

    # Grounding 格式（解析版面信息）
    if has_grounding and level in ["middle", "max"]:
        layout_info = parse_grounding_to_layout(raw_result)
        result["layout_info"] = layout_info
        # 提取纯文本
        result["text"] = extract_text_from_grounding(raw_result)

    # Markdown 格式（可选解析）
    if level in ["middle", "max"] and not has_grounding:
        # 可选：从 Markdown 推断版面
        # result["layout_info"] = parse_markdown_to_layout(raw_result)
        pass

    return result


def parse_grounding_to_layout(raw_result: str) -> list:
    """
    解析 grounding 标记为 layout_info
    """
    import re

    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = re.findall(pattern, raw_result, re.DOTALL)

    layout_info = []
    for element_type, coord_str in matches:
        try:
            # 解析坐标
            coord_str = coord_str.strip()
            if coord_str.startswith('['):
                coordinates = eval(coord_str)
                if isinstance(coordinates, list) and len(coordinates) > 0:
                    if isinstance(coordinates[0], list):
                        bbox = coordinates[0]
                    else:
                        bbox = coordinates

                    layout_info.append({
                        "type": element_type,
                        "bbox": bbox
                    })
        except:
            continue

    return layout_info if layout_info else None


def extract_text_from_grounding(raw_result: str) -> str:
    """
    从 grounding 格式中提取纯文本
    """
    import re

    # 移除所有 grounding 标记
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', raw_result)
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    text = re.sub(r'<\|grounding\|>', '', text)

    return text.strip()
```

### 修改 `api_service/prompts.py`

使用 grounding 模式的提示词：

```python
def build_middle_prompt(request: ImageSubmitRequest) -> str:
    """
    构建中间级别提示词（使用 grounding 模式）
    """
    if request.language == "zh":
        prompt = "<image>\n<|grounding|>请识别图片中的文字内容，并标注版面结构。"
    else:
        prompt = "<image>\n<|grounding|>OCR this image with layout information."

    return prompt
```

---

## 实际输出示例

### 示例 1：Free OCR 模式

**提示词**：
```python
"<image>\nFree OCR."
```

**模型输出**：
```
这是一段文档内容。

这是第二段文字。
```

**API 返回**：
```json
{
  "success": true,
  "result": {
    "text": "这是一段文档内容。\n\n这是第二段文字。"
  }
}
```

### 示例 2：Markdown 模式

**提示词**：
```python
"<image>\n<|grounding|>Convert the document to markdown."
```

**模型输出**：
```markdown
# 文档标题

这是第一段文字。

## 二级标题

- 列表项1
- 列表项2
```

**API 返回**：
```json
{
  "success": true,
  "result": {
    "text": "# 文档标题\n\n这是第一段文字。\n\n## 二级标题\n\n- 列表项1\n- 列表项2"
  }
}
```

### 示例 3：Grounding 模式

**提示词**：
```python
"<image>\n<|grounding|>OCR this image."
```

**模型输出**：
```
<|ref|>title<|/ref|><|det|>[[[0, 0, 500, 50]]]<|/det|>文档标题

<|ref|>paragraph<|/ref|><|det|>[[[0, 60, 500, 200]]]<|/det|>这是段落内容，包含多行文字。
```

**API 返回（修改后）**：
```json
{
  "success": true,
  "result": {
    "text": "文档标题\n\n这是段落内容，包含多行文字。",
    "layout_info": [
      {
        "type": "title",
        "bbox": [0, 0, 500, 50]
      },
      {
        "type": "paragraph",
        "bbox": [0, 60, 500, 200]
      }
    ]
  }
}
```

---

## 总结

### 核心要点

1. **DeepSeek-OCR 只输出文本**（纯文本、Markdown、或带特殊标记的文本）
2. **模型不会自动输出 JSON 格式**
3. **`layout_info` 缺失的原因**：代码假设模型输出 JSON，但实际不是
4. **获得版面信息的正确方法**：
   - 使用 grounding 模式提示词
   - 解析 `<|ref|>` 和 `<|det|>` 特殊标记
   - 或者从 Markdown 结构推断版面

### 你的代码需要修改

- `api_service/utils.py` 的 `parse_ocr_result` 函数
- `api_service/prompts.py` 的提示词模板（使用 grounding 模式）

### 参考文档

- DeepSeek-OCR 官方仓库：[https://github.com/deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- vLLM 集成文档：[https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
