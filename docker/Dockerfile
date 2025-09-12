FROM vllm/vllm-openai:v0.9.1

# Install required packages
RUN pip3 install flash_attn==2.8.0.post2
RUN pip3 install transformers==4.51.3

# Set working directory
WORKDIR /workspace

# Copy the model weights (will be mounted as volume)
# This creates the expected directory structure
RUN mkdir -p /workspace/weights/DotsOCR

# Create initialization script to ensure proper model setup
COPY <<EOF /workspace/init_model.sh
#!/bin/bash
set -e

# Ensure __init__.py exists with proper imports
cat > /workspace/weights/DotsOCR/__init__.py << 'PYEOF'
# DotsOCR Package
from . import modeling_dots_ocr_vllm
from . import modeling_dots_ocr
from . import modeling_dots_vision
from . import configuration_dots

__all__ = ['modeling_dots_ocr_vllm', 'modeling_dots_ocr', 'modeling_dots_vision', 'configuration_dots']
PYEOF

echo "DotsOCR __init__.py created successfully"

# Verify the model files exist
if [ ! -f "/workspace/weights/DotsOCR/modeling_dots_ocr_vllm.py" ]; then
    echo "ERROR: modeling_dots_ocr_vllm.py not found. Please ensure model weights are properly mounted."
    exit 1
fi

# Patch vLLM entrypoint to import our custom model
echo "Patching vLLM entrypoint..."
sed -i '/^from vllm\.entrypoints\.cli\.main import main/a\\from DotsOCR import modeling_dots_ocr_vllm' $(which vllm)

# Verify the patch was applied
echo "vLLM script after patch:"
grep -A 1 'from vllm.entrypoints.cli.main import main' $(which vllm)

echo "Initialization complete!"
EOF

RUN chmod +x /workspace/init_model.sh

# Set up Python path and optimize PyTorch memory usage
ENV PYTHONPATH=/workspace/weights:$PYTHONPATH
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
ENV VLLM_USE_V1=0

# Create startup script
COPY <<EOF /workspace/start_server.sh
#!/bin/bash
set -e

echo "=== DotsOCR vLLM Server Startup ==="

# Run initialization
/workspace/init_model.sh

echo "Starting vLLM server with legacy engine..."
exec vllm serve /workspace/weights/DotsOCR \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.4 \\
    --chat-template-content-format string \\
    --served-model-name dotsocr-model \\
    --trust-remote-code \\
    --max-model-len 2048 \\
    --enforce-eager
EOF

RUN chmod +x /workspace/start_server.sh

# Expose port
EXPOSE 8000

# Use the startup script as entrypoint
ENTRYPOINT ["/workspace/start_server.sh"]
