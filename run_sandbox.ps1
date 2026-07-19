# run_sandbox.ps1
# Builds the Docker container and runs it with strict hardware resource limits:
#   - CPU: Capped at 0.5 cores (--cpus="0.5")
#   - Memory: Capped at 256MB (--memory="256m")
#   - Mounted volumes: Reads model files and validation images dynamically from the host.

Write-Host "🐳 Building Edge Container Sandbox..." -ForegroundColor Cyan
docker build -t pcos-edge-sandbox .

Write-Host "`n🚀 Running benchmark inside Sandbox container with constraints (0.5 CPU, 256MB RAM)..." -ForegroundColor Cyan
docker run --rm `
  --cpus="0.5" `
  --memory="256m" `
  -v "${PWD}/pcos_resnet18.onnx:/pcos_resnet18.onnx" `
  -v "${PWD}/pcos_resnet18_quant.onnx:/pcos_resnet18_quant.onnx" `
  -v "${PWD}/data/cleaned_pcos:/data/cleaned_pcos" `
  -v "${PWD}/data/test_files.txt:/data/test_files.txt" `
  pcos-edge-sandbox
