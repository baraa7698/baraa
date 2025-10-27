#!/usr/bin/env python3
# visualize_filters_mnist.py
# Standalone: loads conv weights from an .npz checkpoint, samples an MNIST image,
# applies conv -> ReLU -> 2x2 max-pool, and visualizes each stage.
#
# Usage:
#   python visualize_filters_mnist.py --ckpt checkpoints/mnist_cnn.npz --seed 0 --index -1
#
# Notes:
# - Expects keys: convW, convB in the .npz file.
# - If convW has multiple input channels, they are averaged for visualization.
# - Requires: numpy, matplotlib, torch, torchvision

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torchvision
    from torchvision import transforms
except Exception as e:
    raise SystemExit(
        "This script requires torch and torchvision to load MNIST.\n"
        "Install them with:\n"
        "  pip install torch torchvision\n"
        f"Import error was: {e}"
    )

def _minmax(x):
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (max(hi - lo, 1e-9))

def _show_grid(tensor, title="", ncols=8, cmap="gray"):
    """
    tensor: (N, H, W) feature maps or filters
    """
    N, H, W = tensor.shape
    ncols = max(1, min(ncols, N))
    nrows = int(np.ceil(N / ncols))
    plt.figure(figsize=(1.8*ncols, 1.8*nrows))
    for i in range(N):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(_minmax(tensor[i]), cmap=cmap)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def _extract_kernels_from_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Checkpoint not found: {npz_path}")
    data = np.load(npz_path)
    if "convW" not in data or "convB" not in data:
        raise KeyError("Checkpoint must contain 'convW' and 'convB'.")

    W = data["convW"]
    b = data["convB"]

    # Normalize kernel shape to (out_ch, kh, kw) for visualization
    if W.ndim == 4:
        # assume (out_ch, in_ch, kh, kw)
        if W.shape[1] == 1:
            Wk = W[:, 0, :, :]
        else:
            # average across input channels for display
            Wk = W.mean(axis=1)
    elif W.ndim == 3:
        # Already (out_ch, kh, kw)
        Wk = W
    else:
        raise ValueError(f"Unexpected convW shape: {W.shape}")

    # Make sure we return writeable copies
    return Wk.copy(), np.array(b, copy=True)

def conv2d_valid_single(x2d, Wk, b=None):
    """
    x2d: (H, W) single-channel image
    Wk: (out_ch, kh, kw) kernels
    b:  (out_ch,) bias or None
    returns: (out_ch, H-kh+1, W-kw+1)
    """
    out_ch, kh, kw = Wk.shape
    H, W = x2d.shape
    H2, W2 = H - kh + 1, W - kw + 1
    if H2 <= 0 or W2 <= 0:
        raise ValueError(f"Kernel larger than input. Input {x2d.shape}, kernel {(kh, kw)}")
    y = np.zeros((out_ch, H2, W2), dtype=np.float32)
    for oc in range(out_ch):
        k = Wk[oc]
        for i in range(H2):
            ii = i + kh
            for j in range(W2):
                jj = j + kw
                y[oc, i, j] = float(np.sum(x2d[i:ii, j:jj] * k))
        if b is not None:
            y[oc] += float(b[oc])
    return y

def relu(x):
    return np.maximum(x, 0)

def maxpool2x2(x):
    """
    x: (C, H, W) -> (C, H//2, W//2)
    """
    C, H, W = x.shape
    H2, W2 = H // 2, W // 2
    x = x[:, :H2*2, :W2*2]
    x = x.reshape(C, H2, 2, W2, 2).max(axis=(2, 4))
    return x

def load_random_mnist(seed=0, index=-1):
    """
    Returns:
      x2d: (H, W) float32 in [0,1]
      y:   int label
    """
    g = torch.Generator()
    ds = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    if index < 0:
        index = torch.randint(0, len(ds), (1,), generator=g).item()
    x, y = ds[index]
    # x is (1, H, W) tensor in [0,1], convert to (H, W) numpy
    x2d = x[0].numpy().astype(np.float32)
    return x2d, int(y)

def main():
    ap = argparse.ArgumentParser(description="Visualize learned conv filters on a random MNIST image.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .npz checkpoint with convW and convB")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for selecting the sample if index < 0")
    ap.add_argument("--index", type=int, default=-1, help="Fixed sample index; use -1 for random")
    ap.add_argument("--ncols", type=int, default=8, help="Grid columns for filter/feature map display")
    args = ap.parse_args()

    # 1) Load kernels
    Wk, b = _extract_kernels_from_npz(args.ckpt)  # (out_ch, kh, kw), (out_ch,)
    print(f"[INFO] Loaded conv kernels: {Wk.shape}, bias: {b.shape} from {args.ckpt}")

    # 2) Load one MNIST image
    x2d, y = load_random_mnist(seed=args.seed, index=args.index)
    print(f"[INFO] Using MNIST sample index={args.index} (label={y})")

    # 3) Forward: conv -> ReLU -> 2x2 max-pool
    conv_maps = conv2d_valid_single(x2d, Wk, b)
    act_maps  = relu(conv_maps)
    pool_maps = maxpool2x2(act_maps)

    # 4) Visualize
    plt.figure(figsize=(3, 3))
    plt.imshow(_minmax(x2d), cmap="gray")
    plt.axis("off")
    plt.title("Input image")
    plt.tight_layout()
    plt.show()

    _show_grid(Wk,        title=f"Learned conv filters ({Wk.shape[0]})", ncols=args.ncols)
    _show_grid(conv_maps, title="After conv", ncols=args.ncols)
    _show_grid(act_maps,  title="After ReLU", ncols=args.ncols)
    _show_grid(pool_maps, title="After 2x2 max-pool", ncols=args.ncols)

if __name__ == "__main__":
    main()
