# DotsOCR Technical Architecture & Development Guide

## Overview

DotsOCR is a unified vision-language model for multilingual document layout parsing that combines layout detection and content recognition in a single 1.7B parameter model. This document provides technical insights into the architecture, implementation details, and development considerations not covered in the main README.

## Architecture Overview

### Core Components

```
dots_ocr/
├── parser.py              # Main parser with ThreadPool processing
├── parser_streaming.py    # Streaming parser with optimized PDF processing
├── model/
│   └── inference.py       # vLLM inference client using OpenAI API
└── utils/
    ├── consts.py          # Core constants and constraints
    ├── prompts.py         # Task-specific prompt templates
    ├── image_utils.py     # Image preprocessing and smart resizing
    ├── layout_utils.py    # Layout processing and visualization
    ├── format_transformer.py # Output format conversion (JSON→MD)
    └── output_cleaner.py  # Fallback output cleaning
```

### Model Inference Backends

DotsOCR supports two inference backends:

1. **vLLM Backend** (Recommended for production)
   - Uses OpenAI-compatible API via [`inference_with_vllm()`](dots_ocr/model/inference.py:7)
   - Optimized for throughput and memory efficiency
   - Supports tensor parallelism and GPU memory utilization control

2. **HuggingFace Transformers Backend**
   - Direct model loading via [`_load_hf_model()`](dots_ocr/parser.py:60)
   - Uses flash attention and mixed precision (bfloat16)
   - Single-threaded processing for memory safety

## Key Technical Features

### Smart Image Resizing

The [`smart_resize()`](dots_ocr/utils/image_utils.py:29) function implements intelligent image preprocessing:

```python
# Core constraints
MIN_PIXELS = 3136        # Minimum resolution
MAX_PIXELS = 11289600    # Maximum resolution (~3360x3360)
IMAGE_FACTOR = 28        # Divisibility factor for model input
```

**Algorithm:**
1. Maintains aspect ratio while ensuring dimensions are divisible by 28
2. Enforces pixel count within [MIN_PIXELS, MAX_PIXELS] range
3. Rejects images with aspect ratio > 200:1
4. Uses mathematical optimization to find optimal dimensions

### Prompt Engineering System

DotsOCR uses task-specific prompts defined in [`prompts.py`](dots_ocr/utils/prompts.py:1):

| Prompt Mode | Purpose | Output Format |
|-------------|---------|---------------|
| `prompt_layout_all_en` | Full layout parsing | JSON with bbox + text |
| `prompt_layout_only_en` | Layout detection only | JSON with bbox only |
| `prompt_ocr` | Text extraction | Plain text |
| `prompt_grounding_ocr` | Region-specific OCR | Text from bbox |

### Streaming PDF Processing

The [`parser_streaming.py`](dots_ocr/parser_streaming.py:1) implements optimized PDF processing:

```python
# Key optimizations:
- ThreadPoolExecutor with controlled concurrency
- Streaming page processing (max_inflight = num_thread * 2)
- Memory-efficient PyMuPDF integration
- Progressive result collection with tqdm progress tracking
```

**Processing Flow:**
1. PDF → Individual pages via PyMuPDF
2. Concurrent page processing with controlled memory usage
3. Results sorted by page order
4. Output aggregation to JSONL format

## Output Processing Pipeline

### Layout Post-Processing

The [`post_process_output()`](dots_ocr/utils/layout_utils.py:202) function handles model output:

1. **JSON Parsing**: Attempts to parse structured layout JSON
2. **Coordinate Transformation**: Converts model coordinates back to original image space
3. **Fallback Processing**: Uses [`OutputCleaner`](dots_ocr/utils/output_cleaner.py) for malformed outputs
4. **Validation**: Ensures legal bounding boxes and proper formatting

### Format Transformation

The [`layoutjson2md()`](dots_ocr/utils/format_transformer.py:145) converts structured output to Markdown:

- **Formula Handling**: LaTeX formulas → `$$...$$` blocks
- **Table Processing**: HTML tables preserved as-is
- **Image Embedding**: Pictures → base64-encoded inline images
- **Text Cleaning**: Removes extra whitespace and formatting artifacts

## Docker Deployment Architecture

### AMD GPU Support

The [`docker-compose.amd.yml`](docker/docker-compose.amd.yml:1) provides ROCm-based deployment:

```yaml
# Key ROCm configurations:
environment:
  - HIP_VISIBLE_DEVICES=0
  - HSA_OVERRIDE_GFX_VERSION=10.3.0
  - PYTORCH_ROCM_ARCH=gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102

devices:
  - /dev/kfd:/dev/kfd    # ROCm kernel driver
  - /dev/dri:/dev/dri    # Direct rendering infrastructure
```

### Container Optimization

The [`Dockerfile.rocm`](docker/Dockerfile.rocm:1) implements:
- ROCm 6.4.3 with PyTorch 2.6.0 base image
- Automatic model initialization and validation
- HuggingFace-based inference server for AMD GPUs
- Mixed precision and memory optimization

## Performance Characteristics

### Memory Requirements

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Model Weights | ~3.4GB | bfloat16 precision |
| Image Processing | Variable | Depends on input resolution |
| vLLM KV Cache | ~2-4GB | Configurable via `--gpu-memory-utilization` |
| PDF Processing | ~100MB/page | Temporary during processing |

### Throughput Optimization

**Single Image Processing:**
- vLLM: ~2-5 seconds per image (GPU-dependent)
- HuggingFace: ~5-10 seconds per image

**PDF Batch Processing:**
- Configurable thread count via `--num_thread`
- Memory-bounded concurrency in streaming mode
- Optimal thread count: min(total_pages, available_memory/500MB)

## Development Considerations

### Extending Prompt Templates

To add new parsing modes:

1. Define prompt in [`prompts.py`](dots_ocr/utils/prompts.py:1)
2. Update [`post_process_output()`](dots_ocr/utils/layout_utils.py:202) for custom output handling
3. Add corresponding CLI argument in [`parser.py`](dots_ocr/parser.py:323)

### Custom Output Formats

The format transformation pipeline is modular:
- Extend [`format_transformer.py`](dots_ocr/utils/format_transformer.py:1) for new output formats
- Implement custom post-processing in [`layout_utils.py`](dots_ocr/utils/layout_utils.py:1)
- Add format-specific validation logic

### Model Integration

For custom model variants:
1. Update model loading in [`_load_hf_model()`](dots_ocr/parser.py:60)
2. Modify inference parameters in [`inference_with_vllm()`](dots_ocr/model/inference.py:7)
3. Adjust image preprocessing constraints in [`consts.py`](dots_ocr/utils/consts.py:1)

## Debugging and Monitoring

### Common Issues

1. **Memory Overflow**: Reduce `max_pixels` or increase GPU memory
2. **Aspect Ratio Errors**: Check input image dimensions
3. **JSON Parsing Failures**: Enable fallback processing via `OutputCleaner`
4. **ROCm Compatibility**: Verify GPU architecture in `PYTORCH_ROCM_ARCH`

### Performance Monitoring

```python
# Memory usage tracking
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()

# Processing time profiling
import time
start_time = time.time()
result = parser.parse_file(input_path)
processing_time = time.time() - start_time
```

### Logging Configuration

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dots_ocr')
```

## API Integration

### vLLM Server Integration

The inference client uses OpenAI-compatible API:

```python
# Custom inference parameters
response = inference_with_vllm(
    image=pil_image,
    prompt=custom_prompt,
    ip="localhost",
    port=8000,
    temperature=0.1,
    top_p=0.9,
    max_completion_tokens=32768
)
```

### Batch Processing API

For high-throughput scenarios:

```python
parser = DotsOCRParser(
    num_thread=64,           # Concurrent processing
    max_completion_tokens=16384,  # Token limit per request
    temperature=0.1,         # Deterministic output
    use_hf=False            # Use vLLM backend
)
```

## Future Development Areas

### Performance Optimization

1. **Model Quantization**: Implement INT8/INT4 quantization for memory efficiency
2. **Batch Inference**: Add support for multi-image batch processing
3. **Streaming Output**: Implement token-level streaming for large documents

### Feature Extensions

1. **Custom Layout Categories**: Support for domain-specific layout types
2. **Multi-Modal Output**: Integration of text, image, and structured data
3. **Quality Metrics**: Automated confidence scoring for parsed content

### Infrastructure Improvements

1. **Distributed Processing**: Multi-GPU and multi-node support
2. **Caching Layer**: Intelligent result caching for repeated processing
3. **Monitoring Integration**: Prometheus metrics and health checks

## Dependencies and Compatibility

### Core Dependencies

```toml
# From pyproject.toml
torch >= 2.8.0           # PyTorch with CUDA/ROCm support
transformers >= 4.56.1   # HuggingFace model loading
flash-attn >= 2.8.3      # Attention optimization
pymupdf >= 1.26.4        # PDF processing
qwen-vl-utils >= 0.0.11  # Vision-language utilities
```

### Hardware Compatibility

**NVIDIA GPUs:**
- Minimum: RTX 3060 (12GB VRAM)
- Recommended: RTX 4090, A100, H100

**AMD GPUs:**
- Minimum: RX 6700 XT (12GB VRAM)
- Recommended: RX 7900 XTX, MI250X, MI300X

**CPU Fallback:**
- Minimum: 16GB RAM
- Recommended: 32GB+ RAM for large documents

## Contributing Guidelines

### Code Style

- Follow PEP 8 for Python code formatting
- Use type hints for function signatures
- Document complex algorithms with inline comments
- Maintain backward compatibility for public APIs

### Testing Framework

```bash
# Unit tests for core functionality
python -m pytest tests/test_image_utils.py
python -m pytest tests/test_layout_utils.py

# Integration tests with sample documents
python -m pytest tests/test_parser_integration.py

# Performance benchmarks
python tools/benchmark_processing.py
```

### Documentation Standards

- Update this technical README for architectural changes
- Maintain API documentation for public interfaces
- Include performance benchmarks for optimization changes
- Document breaking changes in migration guides

---

This technical documentation provides the foundation for understanding DotsOCR's architecture and extending its capabilities. For implementation details, refer to the inline code documentation and the main README for usage examples.