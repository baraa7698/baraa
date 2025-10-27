# mnist_conv_viz.py
import numpy as np
import matplotlib.pyplot as plt

# --- tiny NumPy conv/pool ---
def conv2d_valid(x, w, b=0.0):
    # x: (H,W)  w: (kH,kW)
    H, W = x.shape
    kH, kW = w.shape
    outH, outW = H - kH + 1, W - kW + 1
    y = np.empty((outH, outW), dtype=np.float32)
    for i in range(outH):
        for j in range(outW):
            y[i, j] = np.sum(x[i:i+kH, j:j+kW] * w) + b
    return y

def relu(a): return np.maximum(0, a)

def maxpool2x2(x):
    # x: (H,W) with even sides
    H, W = x.shape
    H2, W2 = H//2, W//2
    x = x[:2*H2, :2*W2]  # crop to even
    return x.reshape(H2, 2, W2, 2).max(axis=(1,3))

# --- filters to visualize ---
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]], dtype=np.float32)

sobel_y = np.array([[-1,-2,-1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

blur3 = (1/9.0) * np.ones((3,3), dtype=np.float32)

sharpen = np.array([[ 0,-1, 0],
                    [-1, 5,-1],
                    [ 0,-1, 0]], dtype=np.float32)

FILTERS = {
    "sobel_x": sobel_x,
    "sobel_y": sobel_y,
    "blur3":   blur3,
    "sharpen": sharpen
}

# --- load MNIST (via torchvision), convert to NumPy ---
import torch
from torchvision import datasets, transforms

ds = datasets.MNIST(root="./data", train=False, download=True,
                    transform=transforms.ToTensor())  # returns [1,28,28] in 0..1

def show_grid(dic, titles, suptitle):
    index = 1
    # dic = dic.keys()
    plt.figure(figsize=(8, 8))
    for i,ttl in enumerate( dic.keys()):
        for z,(t,im) in enumerate(dic[ttl].items()):
            
            #print(z)
            ax =plt.subplot(4,4,index)
            plt.imshow(im, cmap="gray")
            # plt.title(t)
            plt.axis("off")
             # Only add column titles on the top row
            if z ==0:
                ax.text(-3, 13, ttl, ha='center', va='center', rotation='vertical')
            if i ==0:
                ax.text(11, -1, t, ha='center', va='center')
           
                
            index+=1
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

# --- pick a few samples and run each filter pipeline ---
indices = [1]  # change if you want other digits
lable = ''

dicf = {}
for idx in indices:
    x_t, lable = ds[idx]                 # x_t: torch tensor [1,28,28]
    x = x_t.squeeze(0).numpy()       # -> (28,28) float32 in [0,1]
    for fname, F in FILTERS.items():
        conv = conv2d_valid(x, F)    # (26,26) for 3x3 kernel
        act  = relu(conv)            # ReLU
        pool = maxpool2x2(act)       # (13,13)
        dicf[fname]={'x':x,'conv':conv,'act':act,'pool':pool}

# print(dicf.keys())

show_grid(
            dicf,
            ["orig 28×28",
             f"conv(test)",
             "after ReLU",
             "maxpool 2×2"],
            suptitle=f"Digit={lable}  |  Filter=test"
        )
