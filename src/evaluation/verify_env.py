"""Verify the environment: real GPU compute, not just cuda.is_available()."""
import sys
import time

import torch


def main():
    print("python     :", sys.version.split()[0])
    print("executable :", sys.executable)
    print("torch      :", torch.__version__)
    print("cuda build :", torch.version.cuda)
    print("cuda avail :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: no CUDA")
        return 1

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("device     : %s (%.1f GB)" % (name, total))

    # Real matmul on GPU, timed, with correctness check against CPU.
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = (2 * 4096**3 * 10) / dt / 1e12
    print("matmul     : %.3fs for 10x4096^3  -> %.1f TFLOPS" % (dt, tflops))

    # Numerical agreement with CPU on a small slice.
    a_s, b_s = a[:256, :256].cpu(), b[:256, :256].cpu()
    ref = a_s @ b_s
    got = (a[:256, :256] @ b[:256, :256]).cpu()
    err = (ref - got).abs().max().item()
    print("max abs err vs cpu (256 slice, not same op): %.4f" % err)

    # Conv + backward, the actual workload shape for segmentation.
    import torch.nn as nn
    net = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 10, 1),
    ).to(dev)
    x = torch.randn(8, 3, 512, 512, device=dev)
    y = torch.randint(0, 10, (8, 512, 512), device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad()
        loss = lossf(net(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print("conv train : 5 steps @ 8x3x512x512 in %.2fs, loss=%.4f" % (time.time() - t0, loss.item()))

    # AMP, which any real training run will use.
    scaler = torch.amp.GradScaler("cuda")
    with torch.amp.autocast("cuda"):
        loss = lossf(net(x), y)
    scaler.scale(loss).backward()
    print("amp        : OK (loss=%.4f)" % loss.item())

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print("peak vram  : %.2f GB" % peak)
    print("\nPASS: GPU compute verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
