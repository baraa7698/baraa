# numpy_cnn_mnist_split.py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os, json, time
from datetime import datetime
X_sample =[]
#-----------------------------------------------------------------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _append_summary_csv(path, time_s, acc, loss):
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("time_s,accuracy,loss\n")
        f.write(f"{time_s:.6f},{acc:.6f},{loss:.6f}\n")

def _save_config(path, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def _snapshot_weights(feature_extractor, classifier):
    # conv
    convW = feature_extractor.conv.W.copy()
    convB = feature_extractor.conv.b.copy()
    # head
    fc1W = classifier.fc1.W.copy(); fc1B = classifier.fc1.b.copy()
    fc2W = classifier.fc2.W.copy(); fc2B = classifier.fc2.b.copy()
    return dict(convW=convW, convB=convB, fc1W=fc1W, fc1B=fc1B, fc2W=fc2W, fc2B=fc2B)

# ----------------------------------------------------------------------------------------
# Utils (unchanged style)
# ----------------------------------------------------------------------------------------


def sparse_ce_from_logits(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    prob = ez / ez.sum(axis=1, keepdims=True)
    
    N = y.shape[0]
    oh = np.zeros((y.size, 10), dtype=np.float32)
    prob[np.arange(N), y] -= 1.0
    dlogits = prob 
    loss = -np.mean(z[np.arange(N), y] - np.log(ez.sum(axis=1)))
    return loss, dlogits , prob


# ---------------------
# Core layers (your originals)
# ---------------------
class Conv2D:
    def __init__(self, C_in, C_out, k=3):
        limit = np.sqrt(2.0 / (C_in * k * k))
        self.W = np.random.randn(C_out, C_in, k, k).astype(np.float32) * limit
        self.b = np.zeros((C_out, 1), dtype=np.float32)
        self.k = k

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        k = self.k
        outH, outW = H - k + 1, W - k + 1

        # im2col via sliding_window_view
        patches = sliding_window_view(x, (k, k), axis=(2, 3))                     # (N,C,OH,OW,k,k)
        cols = patches.reshape(N, C, outH*outW, k*k).transpose(0,1,3,2)           # (N,C,k*k,L)
        cols = np.ascontiguousarray(cols.reshape(N, C*k*k, outH*outW))            # (N,Ckk,L)
        self.cols = cols
        self.outHW = (outH, outW)

        Wm = np.ascontiguousarray(self.W.reshape(self.W.shape[0], -1))            # (C_out,Ckk)

        # Batched GEMM: (1,C_out,Ckk) @ (N,Ckk,L) -> (N,C_out,L)
        y = np.matmul(Wm[None, :, :], cols)                                       # (N,C_out,L)
        y += self.b.reshape(1, -1, 1)                                             # broadcast bias (1,C_out,1)
        y = y.reshape(N, self.W.shape[0], outH, outW)
        return y


    def backward(self, dy, lr):
        # x: (N, C_in, H, W)
        x = self.x
        N, C_in, H, W = x.shape
        C_out, _, k, _ = self.W.shape
        outH, outW = self.outHW
        L = outH * outW
        Ckk = C_in * k * k

        # reshape once
        dyL = dy.reshape(N, C_out, L)                        # (N, C_out, L)
        Wm  = np.ascontiguousarray(self.W.reshape(C_out, Ckk))  # (C_out, Ckk)

        # ---- dW: sum_n (dy_n @ cols_n^T) ----
        # dyL: (N, C_out, L), cols: (N, Ckk, L) -> transpose to (N, L, Ckk)
        dW_batch = np.matmul(dyL, np.transpose(self.cols, (0, 2, 1)))  # (N, C_out, Ckk)
        dW = dW_batch.sum(axis=0) / N                                  # (C_out, Ckk)

        # ---- db: sum over N,H,W per output channel ----
        db = dy.sum(axis=(0, 2, 3), keepdims=True).reshape(self.b.shape) / N  # (C_out, 1)

        # ---- dcols: W^T @ dy (batched) ----
        # Wm.T: (Ckk, C_out), dyL: (N, C_out, L) -> (N, Ckk, L)
        dcols = np.matmul(Wm.T[None, :, :], dyL)                  # (N, Ckk, L)  <-- no squeeze
        dcols = dcols.reshape(N, C_in, k * k, outH, outW)         # (N, C_in, k*k, OH, OW)

        # ---- col2im scatter (tiny loop over k*k) ----
        dx = np.zeros_like(x, dtype=np.float32)
        t = 0
        for p in range(k):
            for q in range(k):
                dx[:, :, p:p+outH, q:q+outW] += dcols[:, :, t, :, :]
                t += 1

        # ---- SGD update ----
        self.W -= lr * dW.reshape(self.W.shape)
        self.b -= lr * db
        return dx


class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
    def backward(self, dy, lr):
        return dy * self.mask

class MaxPool2x2:
    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        H2, W2 = H // 2, W // 2
        x2 = x[:, :, :2*H2, :2*W2]                      # crop if odd
        # reshape to expose 2x2 windows as a length-4 axis
        self.flat = x2.reshape(N, C, H2, 2, W2, 2).reshape(N, C, H2, W2, 4)
        self.argmax = self.flat.argmax(axis=-1)         # (N,C,H2,W2)
        y = self.flat.max(axis=-1)                      # (N,C,H2,W2)
        return y

    def backward(self, dy, lr):
        N, C, H2, W2 = dy.shape
        # route grads to winning positions only
        dx_flat = np.zeros((N, C, H2, W2, 4), dtype=np.float32)
        r = np.arange(N)[:, None, None, None]
        c = np.arange(C)[None, :, None, None]
        i = np.arange(H2)[None, None, :, None]
        j = np.arange(W2)[None, None, None, :]
        dx_flat[r, c, i, j, self.argmax] = dy
        dx = dx_flat.reshape(N, C, H2, 2, W2, 2).reshape(N, C, H2*2, W2*2)

        # if input had odd H/W we cropped in forward; pad zeros back
        H, W = self.x_shape[2], self.x_shape[3]
        out = np.zeros(self.x_shape, dtype=np.float32)
        out[:, :, :dx.shape[2], :dx.shape[3]] = dx
        return out

class Dense:
    def __init__(self, D_in, D_out):
        limit = np.sqrt(2.0/D_in)
        self.W = np.random.randn(D_in, D_out).astype(np.float32) * limit
        self.b = np.zeros((1, D_out), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dY, lr):
        N = dY.shape[0]
        dW = (self.x.T @ dY) / N
        db = np.sum(dY, axis=0, keepdims=True) / N
        dx = dY @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx

# ---------------------
# Part 1: Feature extractor (conv → relu → pool → flatten)
# ---------------------
class FeatureExtractor:
    """
    Processes the image only: Conv → ReLU → MaxPool → Flatten.
    Keeps its own learning rate for internal layers.
    """
    def __init__(self, lr=0.05):
        self.conv = Conv2D(C_in=1, C_out=8, k=3)  # 28 -> 26
        self.relu = ReLU()
        self.pool = MaxPool2x2()                  # 26 -> 13
        self.lr = lr
        self.flat_dim = 8 * 13 * 13

    def forward(self, x):
        z = self.conv.forward(x)            # (N,8,26,26)
        z = self.relu.forward(z)            # (N,8,26,26)
        z = self.pool.forward(z)            # (N,8,13,13)
        self.after_pool_shape = z.shape     # save for backward reshape
        return z.reshape(z.shape[0], -1)    # (N, flat_dim)

    def backward(self, dfeatures):
        # dfeatures: (N, flat_dim) coming from the classifier
        dz = dfeatures.reshape(self.after_pool_shape)       # (N,8,13,13)
        dz = self.pool.backward(dz, self.lr)                # (N,8,26,26)
        dz = self.relu.backward(dz, self.lr)                # (N,8,26,26)
        dx = self.conv.backward(dz, self.lr)                # (N,1,28,28)
        return dx

# ---------------------
# Part 2: Classifier head (adjustable hidden nodes, epochs, batches)
# ---------------------
class Classifier:
    """
    A small MLP head on top of extracted features.
    hidden_dim: number of nodes in the 'second layer' (the hidden layer)
    """
    def __init__(self, input_dim, num_classes=10, hidden_dim=128, lr=0.05):
        self.lr = lr
        self.num_classes = num_classes
        # Two-layer MLP: input -> hidden -> ReLU -> logits
        self.fc1 = Dense(input_dim, hidden_dim)
        self.act = ReLU()
        self.fc2 = Dense(hidden_dim, num_classes)

    def forward(self, features):
        z = self.fc1.forward(features)
        z = self.act.forward(z)
        logits = self.fc2.forward(z)
        return logits

    def backward(self, dlogits):
        dz = self.fc2.backward(dlogits, self.lr)
        dz = self.act.backward(dz, self.lr)
        dfeatures = self.fc1.backward(dz, self.lr)
        return dfeatures

    def train(
        self, feature_extractor, epochs=5, batch_size=64,
        split="train", verbose=True
    ):
        
        # ---------------- timing + run folder ----------------       # NEW
        t0 = time.perf_counter()
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", run_ts)
        _ensure_dir(run_dir)

        # capture initial weights                                    # NEW
        init_weights = _snapshot_weights(feature_extractor, self)

        # capture config you started with                            # NEW
        run_cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "split": split,
            "feature_extractor": {
                "lr": getattr(feature_extractor, "lr", None),
                "conv_k": feature_extractor.conv.k,
                "conv_cin": feature_extractor.conv.W.shape[1],
                "conv_cout": feature_extractor.conv.W.shape[0],
                "flat_dim": feature_extractor.flat_dim
            },
            "classifier": {
                "lr": self.lr,
                "num_classes": self.num_classes,
                "hidden_dim": self.fc1.W.shape[1],   # D_out of fc1
                "input_dim": self.fc1.W.shape[0]
            }
        }
        _save_config(os.path.join(run_dir, "config.json"), run_cfg)

    # ---------------- training loop (your proper mini-batch loop) ----------------
        acc , loss=0 ,0
        X_all, y_all = load_mnist_batch( split=split)
        X_sample = X_all
        N = X_all.shape[0]
        for ep in range(epochs):
            for steps in range(0, N, batch_size):
                id = steps+batch_size
                
                X, y = X_all[steps:id], y_all[steps:id]
                # forward through extractor and classifier
                feats = feature_extractor.forward(X)
                logits = self.forward(feats)
                loss, dlogits, prob = sparse_ce_from_logits(logits, y)




                # backward through classifier and extractor
                dfeatures = self.backward(dlogits)
                feature_extractor.backward(dfeatures)

                pred = logits.argmax(axis=1)
                acc = np.mean(pred == y)
                if verbose and (steps // batch_size) % 50 == 0:
                    print(f"epoch {ep+1:02d}, batch:{steps:05d}/{N:05d}  loss {loss:.3f}  acc {acc*100:5.1f}% ")

        # ---------------- end-of-run logging ----------------       # NEW
        elapsed = time.perf_counter() - t0
        final_acc  = acc 
        final_loss = loss

        # append to global CSV
        _ensure_dir("runs")
        _append_summary_csv(os.path.join("runs", "summary.csv"), elapsed, final_acc, final_loss)

        # save initial and final weights
        np.savez(os.path.join(run_dir, "weights_initial.npz"), **init_weights)
        final_weights = _snapshot_weights(feature_extractor, self)
        np.savez(os.path.join(run_dir, "weights_final.npz"), **final_weights)

        # also dump a quick human-readable recap
        with open(os.path.join(run_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(f"elapsed_s: {elapsed:.6f}\n")
            f.write(f"final_acc: {final_acc:.6f}\n")
            f.write(f"final_loss: {final_loss:.6f}\n")
            f.write("\nconfig:\n")
            f.write(json.dumps(run_cfg, indent=2))

                
# ---------------------
# Data loader (unchanged logic, grabs a random batch)
# ---------------------
def load_mnist_batch(split="train", cache_path="data/mnist_all.npz"):
    import os
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["X"], d["y"]

    import torch
    from torchvision import datasets, transforms

    ds = datasets.MNIST(root="./data", train=(split=="train"),
                        download=True, transform=transforms.ToTensor())
    # stack all tensors once (C,H,W) then convert once
    xs = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)  # (N,1,28,28)
    ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
    X = xs.numpy().astype(np.float32)
    y = ys.numpy().astype(np.int64)

    # optional shuffle once to match your previous behavior
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    # cache for next runs
    np.savez(cache_path, X=X, y=y)
    return X, y


# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Part 1: image processing pipeline
    extractor = FeatureExtractor(lr=0.05)

    # Part 2: classifier with adjustable hidden nodes
    cls = Classifier(
        input_dim=extractor.flat_dim,
        num_classes=10,
        hidden_dim=10,   # ← adjust this
        lr=0.05
    )

    # Train controls: epochs and batch size are adjustable
    cls.train(
        feature_extractor=extractor,
        epochs=1,        # ← adjust epochs
        batch_size=300,    # ← adjust batch size
        split="train",
        verbose=True
    )

    print("Done.")
