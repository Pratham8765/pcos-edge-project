# Stripped-down Linux core for edge container simulation
FROM python:3.10-slim

# Avoid writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Decouple framework dependencies: install ONLY lightweight inference runtimes
RUN pip install --no-cache-dir \
    numpy \
    onnxruntime \
    Pillow \
    psutil

# Copy the benchmarking script
COPY src/benchmark.py /app/benchmark.py

# Expected runtime mount points to avoid copying large validation assets into the container layer:
#   - Model Files: /pcos_resnet18.onnx and /pcos_resnet18_quant.onnx (mounted to /)
#   - Dataset Folder: /data/cleaned_pcos (mounted to /data/cleaned_pcos)
#   - Test File Index: /data/test_files.txt (mounted to /data/test_files.txt)
#
# Workdir paths are resolved inside benchmark.py fallback mechanism to check root and /data.

ENTRYPOINT ["python", "-u", "/app/benchmark.py"]
