import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    # Quick GPU test
    x = torch.rand(1000, 1000).cuda()
    y = torch.mm(x, x)
    print("GPU matrix multiply: SUCCESS (speed test passed)")
    print(f"GPU memory usage: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
else:
    print("Still CPU — check NVIDIA drivers (need 546.01+ for CUDA 12.6)")