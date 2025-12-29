# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

不管什么使用，使用中文回答我，尽可能多停留，询问我的意见，告诉我你在做什么，为什么这么做。指出我的指令中存在的问题，向我确定之后再开始执行。

这个项目使用conda管理环境，运行的时候必须要保证conda的环境被激活。conda activate deepseek-ocr

完成了任务在docs目录下面写一个总结文档。太长的总结性的回答也放在里面，遇到了花费很长时间才解决的问题，也在docs目录下面创建总结文档，因为我需要学习。

回应我，一律使用Markdown。

## Project Overview

DeepSeek-OCR is a multi-modal OCR system that combines large language models with advanced vision encoders for context-aware text extraction from images and documents. The architecture uses a dual-encoder approach (SAM + CLIP) with dynamic resolution processing for handling various image sizes.

## Temporary Research Files

- Canonical location: store all temporary research artifacts (downloaded READMEs, API docs, scratch notes) under `docs/research/`.
- Do not place temporary files at the repository root or outside `docs/research/`.
- Commit policy: avoid committing temporary files unless they are necessary for traceability during `/spec` or `/plan`. If committed, keep the scope minimal and store them under `docs/` only.
- Naming: use descriptive names with date or task context (e.g., `docs/research/2025-09-11-wsl-notes.md`).
- Cleanup: after completing a task (`/do`), evaluate whether each temporary file is still required. Remove unneeded files, or promote essential content into curated docs under `docs/` or into `specs/`/`plans/`.

## Stage-Gated Workflow (spec/plan/do)

- Mode: Opt-in. The workflow applies only when the user explicitly uses `/spec`, `/plan`, or `/do`. Routine Q&A or trivial edits do not require these stages.
- Triggers: A message containing one of `/spec`, `/plan`, or `/do` activates or advances the workflow. Once active, stages must proceed in order with explicit user approval to advance.
- Guardrails:
  - Do not modify source code before `/do`. Documentation/spec files may be edited only in `/spec`.
  - Do not skip stages or proceed without user confirmation once the workflow is active.
  - If scope changes, return to the appropriate prior stage for approval.
  - Respect sandbox/approval settings for all actions.

- When to Use
  - Use the workflow for new features, structural refactors, multi-file changes, or work needing traceability.
  - Skip the workflow (no triggers) for routine Q&A, diagnostics, or one-off trivial edits.

- Entry Points and Prerequisites
  - `/spec` is the canonical entry point for new efforts.
  - `/plan` requires an approved `/spec`. If unclear which spec applies, pause and ask the user to identify the correct file(s) under `specs/`.
  - `/do` requires an approved `/plan`.

- `/spec` (Specifications; docs only)
  - Purpose: Capture a concrete, reviewable specification using spec-kit style.
  - Output: Markdown spec(s) under `specs/` (no code changes). Share a concise diff summary and links to updated files; wait for approval.
  - Style: Specs are canonical and final. Do not include change logs or “correction/更正” notes. Incorporate revisions directly so the document always reflects the current agreed state. Historical context belongs in PR descriptions, commit messages, or the conversation — not in the spec.
  - Recommended contents:
    - Problem statement and context
    - Goals and non-goals
    - Requirements and constraints (functional, UX, performance, security)
    - UX/flows and API/IPC contracts (as applicable)
    - Acceptance criteria and success metrics
    - Alternatives considered and open questions
    - Rollout/backout considerations and telemetry (if relevant)

- `/plan` (High-level Plan; docs only)
  - Purpose: Turn the approved spec into an ordered, verifiable implementation plan.
  - Inputs: Approved spec file(s) in `specs/`.
  - Ambiguity: If the relevant spec is unclear, pause and request clarification before writing the plan.
  - Style: Plans are canonical and should not include change logs or “correction/更正” notes. Incorporate revisions directly so the plan always reflects the current agreed state. Historical notes should live in PR descriptions, commit messages, or the conversation.
  - Output:
    - An ordered plan via `update_plan` (short, verifiable steps; Task is merged into Plan and tracked here).
    - A plan document in `plans/` named `YYYY-MM-DD-short-title.md`, containing:
      - Scope and links to authoritative spec(s)
      - Assumptions and out-of-scope items
      - Phases/milestones mapped to acceptance criteria
      - Impacted areas, dependencies, risks/mitigations
      - Validation strategy (tests/lint/build) and rollout/backout notes
      - Approval status and next stage
  - Handoff: Await user approval of the plan before `/do`.

- `/do` (Execution)
  - Purpose: Implement approved plan steps with minimal, focused changes and file operations.
  - Actions:
    - Use `apply_patch` for file edits; group related changes and keep diffs scoped to approved steps.
    - Provide concise progress updates and a final summary of changes.
    - Validate appropriately: run `pnpm lint`, `pnpm format`, `pnpm build`, and relevant tests.
    - If material changes to the plan are needed, pause and return to `/plan` (or `/spec`) for approval.
  - Output: Implemented changes, validation results, and a concise change summary linked to the plan checklist.

### Plans Directory

- Location: `plans/` at the repository root.
- Filename: `YYYY-MM-DD-short-title.md` (kebab-case title; consistent dating).
- Style: Plan docs are the canonical source of truth for the implementation approach; avoid embedding change logs or “correction/更正” notes. Update the plan in place as decisions evolve.
- Contents:
  - Title and summary
  - Scope and linked specs (paths under `specs/`)
  - Assumptions / Out of scope
  - Step-by-step plan (short, verifiable)
  - Validation strategy (tests/lint/build)
  - Approval status and next stage
- Process:
  - During `/plan`, create or update the relevant file in `plans/` and share a short summary in the conversation. Await approval before `/do`.



## Environment Setup

```bash
# Create conda environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Install dependencies
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl  # Download from vLLM releases
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

**Requirements**: CUDA 11.8, PyTorch 2.6.0, vLLM 0.8.5, Flash Attention 2.7.3

## Running Inference

### vLLM Implementation (Recommended)

Navigate to `DeepSeek-OCR-master/DeepSeek-OCR-vllm/` and edit `config.py` first:

```bash
# Single image with streaming output
python run_dpsk_ocr_image.py

# PDF batch processing (~2500 tokens/s on A100-40G)
python run_dpsk_ocr_pdf.py

# Batch evaluation for benchmarks
python run_dpsk_ocr_eval_batch.py
```

### HuggingFace Transformers Implementation

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

## Configuration

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Resolution modes (choose one):
# Tiny:   base_size=512,  image_size=512,  crop_mode=False
# Small:  base_size=640,  image_size=640,  crop_mode=False
# Base:   base_size=1024, image_size=1024, crop_mode=False
# Large:  base_size=1280, image_size=1280, crop_mode=False
# Gundam: base_size=1024, image_size=640,  crop_mode=True (dynamic)

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # Reduce if GPU memory limited
MAX_CONCURRENCY = 100
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
INPUT_PATH = ''  # Set your input path
OUTPUT_PATH = ''  # Set your output path
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

## Architecture

### Core Components

- **`deepseek_ocr.py`**: Main vLLM model implementation with `DeepseekOCRForCausalLM` class
- **`process/image_process.py`**: Image preprocessing, tokenization, and dynamic tiling logic
- **`deepencoder/sam_vary_sdpa.py`**: SAM (Segment Anything Model) vision encoder
- **`deepencoder/clip_sdpa.py`**: CLIP vision encoder
- **`deepencoder/build_linear.py`**: MLP projection layers (2048→1280D)
- **`process/ngram_norepeat.py`**: N-gram repetition prevention logits processor

### Dual-Encoder Architecture

The system processes images through two parallel pathways:

1. **Global View**: Full image resized to `base_size` (1024-1280px) → SAM encoder → CLIP encoder → Projector
2. **Local Views**: Dynamic tiles at `image_size` (640px) → SAM encoder → CLIP encoder → Projector

Features are concatenated and formatted with special tokens:
- `image_newline`: Row separator for 2D token organization
- `view_separator`: Separator between local and global views

### Dynamic Resolution Processing

For images larger than 640×640 with `crop_mode=True`:
- `count_tiles()` calculates optimal tile layout (e.g., 2×3, 3×3) within `MIN_CROPS` to `MAX_CROPS`
- Each tile is processed independently as a local view
- Tile arrangement is preserved in the 2D token structure for spatial awareness

## Prompt Templates

```python
# Document to markdown
'<image>\n<|grounding|>Convert the document to markdown.'

# General OCR with layout detection
'<image>\n<|grounding|>OCR this image.'

# OCR without layout preservation
'<image>\nFree OCR.'

# Figure/chart parsing
'<image>\nParse the figure.'

# Detailed image description
'<image>\nDescribe this image in detail.'

# Object localization
'<image>\nLocate <|ref|>xxxx<|/ref|> in the image.'
```

## Token Count Estimation

For a given image size `(width, height)`:

```python
# Global view tokens
h = w = ceil((base_size // 16) / 4)  # patch_size=16, downsample_ratio=4
global_tokens = h * (w + 1)  # +1 for newline token per row

# Local view tokens (if crop_mode=True and image > 640×640)
crop_ratio = count_tiles(width, height, image_size=640)  # e.g., (2, 3)
h2 = w2 = ceil((image_size // 16) / 4)
local_tokens = (crop_ratio[1] * h2) * (crop_ratio[0] * w2 + 1)

# Total
total_tokens = global_tokens + local_tokens + 1  # +1 for view_separator
```

Example for Gundam mode (1024 base + 640×3 tiles):
- Global: 16×17 = 272 tokens
- Local: (3×16)×(2×16+1) = 48×33 = 1584 tokens
- Total: 1857 tokens

## Using Upstream vLLM (As of 2025/10/23)

DeepSeek-OCR is officially supported in upstream vLLM:

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

image = Image.open("path/to/image.png").convert("RGB")
prompt = "<image>\nFree OCR."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td>, </td>
    ),
    skip_special_tokens=False,
)

outputs = llm.generate(model_input, sampling_param)
```

## Key Implementation Details

- **Special tokens**: The model uses learned `image_newline` and `view_seperator` embeddings for 2D spatial token organization (`tile_tag="2D"`)
- **N-gram prevention**: `NGramPerReqLogitsProcessor` prevents repetitive text generation with configurable window sizes
- **Memory efficiency**: Vision encoders run with `torch.no_grad()` and use bfloat16 precision
- **vLLM integration**: Custom `MULTIMODAL_REGISTRY` processor handles `<image>` token replacement with correct number of vision tokens
