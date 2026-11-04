import time
import torch

print("PyTorch 버전 :", torch.__version__)
print("Intel GPU(xpu) :", torch.xpu.is_available())

if torch.xpu.is_available():
    print("GPU 이름 : ", torch.xpu.get_device_name(0))
    p = torch.xpu.get_device_properties(0)
    print("GPU 메모리 : %.1f GB (RAM에서 공유)" % (p.total_memory / 1024**3))

def bench(device, n=4096, reps=10):
    x = torch.randn(n, n, device=device)
    x @ x
    if device == "xpu":
        torch.xpu.synchronize()

    t = time.perf_counter()
    for _ in range(reps):
        y = x @ x

    if device == "xpu":
        torch.xpu.synchronize()
    
    return (time.perf_counter() - t) / reps

print("CPU 행렬곱 1회 : %.4f 초" % bench("cpu", reps=2))
if torch.xpu.is_available():
    print("GPU 행렬곱 1회 : %.4f 초" % bench("xpu"))

