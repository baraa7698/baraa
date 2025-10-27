# numpy_cnn_mnist_split.py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os, json, time
from datetime import datetime

# =========================================================================================================================
# ====== Misclassification review (drop-in) ===================================
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def _softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def review_misclassified(feature_extractor, classifier, split="train",
                         save_dir=None, batch_size=1024, limit=None):
    """
    Scan a split, collect wrong predictions, optionally save them,
    and open an interactive viewer with Prev/Next buttons.

    Args:
        feature_extractor, classifier: your trained modules
        split: "train" or "test" (your loader ignores it but keeping for clarity)
        save_dir: if not None, will save PNGs of wrong samples there
        batch_size: forward-pass chunk to avoid RAM drama
        limit: cap number of wrong samples shown/saved (None = show all)
    """
    
    
    N = pics.shape[0]

    wrong_imgs, wrong_true, wrong_pred, wrong_probs = [], [], [], []

    # forward in chunks
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        Xb = pics[s:e]
        yb = lables[s:e]

        feats = feature_extractor.forward(Xb)
        logits = classifier.forward(feats)
        probs = _softmax_np(logits)
        pred = logits.argmax(axis=1)

        mask = pred != yb
        if np.any(mask):
            Xbad = Xb[mask]
            ybad = yb[mask]
            pbad = pred[mask]
            probs_bad = probs[mask]

            wrong_imgs.append(Xbad)
            wrong_true.append(ybad)
            wrong_pred.append(pbad)
            wrong_probs.append(probs_bad)

    if not wrong_imgs:
        print("[REVIEW] No misclassifications found. Either your model is perfect or broken in a very creative way.")
        return

    Xw = np.concatenate(wrong_imgs, axis=0)
    yw = np.concatenate(wrong_true, axis=0)
    pw = np.concatenate(wrong_pred, axis=0)
    probw = np.concatenate(wrong_probs, axis=0)

    if limit is not None:
        Xw = Xw[:limit]
        yw = yw[:limit]
        pw = pw[:limit]
        probw = probw[:limit]

    # optional save
    if save_dir is not None:
        _ensure_dir(save_dir)
        for i in range(Xw.shape[0]):
            # save as 28x28 grayscale PNG
            try:
                plt.imsave(os.path.join(save_dir, f"idx_{i:05d}_pred{int(pw[i])}_true{int(yw[i])}.png"),
                           Xw[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
            except Exception as ex:
                print(f"[REVIEW] Save failed @ {i}: {ex}")

        # also drop a compact index for programmatic use
        np.savez(os.path.join(save_dir, "misclassified_index.npz"),
                 X=Xw, y_true=yw, y_pred=pw, probs=probw)
        print(f"[REVIEW] Saved {Xw.shape[0]} misclassified samples to {save_dir}")

    # --------- tiny viewer with Prev/Next -----------
        # --------- viewer with Prev/Next and scrollable prob list -----------
    # Optional: custom label names for EMNIST splits
    # If None, it will fall back to numeric indices.
    def _default_label_names(C):
        # Common full set, we'll truncate to C
        full = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        if C <= len(full):
            return full[:C]
        # Fallback to indices if C is unusual
        return [str(i) for i in range(C)]

    C = probw.shape[1]
    label_names = _default_label_names(C)

    state = {"i": 0, "offset": 0}  # offset for scrolling the prob list
    VISIBLE = 14                   # how many probability rows to show at once

    def _fmt_label(cls_idx):
        try:
            return f"{label_names[cls_idx]}"
        except Exception:
            return f"{cls_idx}"

    def _title_for(idx):
        p = probw[idx]
        top3 = np.argsort(-p)[:3]
        t3 = ", ".join([f"{_fmt_label(c)}:{p[c]*100:.1f}%" for c in top3])
        pred_lbl = _fmt_label(int(pw[idx]))
        true_lbl = _fmt_label(int(yw[idx]))
        return f"Wrong #{idx+1}/{Xw.shape[0]} | pred={pred_lbl} vs true={true_lbl}"

    fig = plt.subplots(figsize=(8, 6))[0]
    # GridSpec-ish manual layout: left probs, right image, bottom buttons
    left = plt.axes([0.05, 0.2, 0.35, 0.75])   # probs panel
    right = plt.axes([0.45, 0.2, 0.5, 0.75])   # image panel
    plt.subplots_adjust(bottom=0.2)

    # Image
    img_ax = right
    img = img_ax.imshow(Xw[0, 0], cmap="gray", vmin=0.0, vmax=1.0)
    img_ax.set_title(_title_for(0))
    img_ax.axis("off")

    # Probability panel (text list, scrollable)
    prob_ax = left
    prob_ax.set_xlim(0, 1)
    prob_ax.set_ylim(0, VISIBLE)
    prob_ax.set_xticks([])
    prob_ax.set_yticks([])
    prob_ax.set_title("Class probabilities (scroll)")

    rows_text = [prob_ax.text(0.02, VISIBLE - 0.5 - r, "", fontsize=10, family="monospace")
                 for r in range(VISIBLE)]

    highlight_bar = prob_ax.barh(y=[], width=[], height=0.8)  # placeholder, unused

    def _sorted_indices(idx):
        return np.argsort(-probw[idx])

    def _update_prob_list(idx):
        p = probw[idx]
        order = _sorted_indices(idx)
        start = state["offset"]
        end = min(start + VISIBLE, len(order))
        # Clear old lines
        for t in rows_text:
            t.set_text("")
        # Fill visible slice
        row = 0
        for j in range(start, end):
            c = order[j]
            mark = "←" if c == pw[idx] else "  "
            rows_text[row].set_text(f"{mark} {_fmt_label(c):>3}  {p[c]*100:6.2f}%")
            row += 1
        # Draw a thin guideline bar to hint at probability scale
        prob_ax.clear()
        prob_ax.set_xlim(0, 1)
        prob_ax.set_ylim(0, VISIBLE)
        prob_ax.set_xticks([])
        prob_ax.set_yticks([])
        prob_ax.set_title("Class probabilities (scroll)")
        # Re-add texts after clearing
        for r in range(VISIBLE):
            if r < row:
                prob_ax.text(0.02, VISIBLE - 0.5 - r, rows_text[r].get_text(),
                             fontsize=10, family="monospace")
        fig.canvas.draw_idle()

    # Buttons
    axprev = plt.axes([0.25, 0.05, 0.15, 0.075])
    axnext = plt.axes([0.60, 0.05, 0.15, 0.075])
    bprev = Button(axprev, "Prev")
    bnext = Button(axnext, "Next")

    def show_idx(k):
        # clamp scroll offset so it stays valid for this sample
        state["offset"] = max(0, min(state["offset"], C - VISIBLE))
        img.set_data(Xw[k, 0])
        img_ax.set_title(_title_for(k))
        _update_prob_list(k)
        fig.canvas.draw_idle()

    def on_prev(event):
        state["i"] = (state["i"] - 1) % Xw.shape[0]
        state["offset"] = 0
        show_idx(state["i"])

    def on_next(event):
        state["i"] = (state["i"] + 1) % Xw.shape[0]
        state["offset"] = 0
        show_idx(state["i"])

    def on_scroll(event):
        if event.inaxes != prob_ax:
            return
        # Matplotlib wheel: step is +1 or -1 depending on scroll direction
        step = -1 if event.button == "up" else 1
        state["offset"] = int(np.clip(state["offset"] + step, 0, max(0, C - VISIBLE)))
        _update_prob_list(state["i"])

    bprev.on_clicked(on_prev)
    bnext.on_clicked(on_next)
    cid = fig.canvas.mpl_connect("scroll_event", on_scroll)

    # initial draw
    show_idx(0)
    print(f"[REVIEW] Collected {Xw.shape[0]} misclassified samples. Scroll over the left panel.")
    plt.show()











# ==========================================================================================================================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_wights(feature_extractor, classifier, ckpt_path="weights_saved.npz"):
    """
    Load weights if the file exists; otherwise, keep random initialization.
    Returns True if loaded successfully, False otherwise.
    """
    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path)
        feature_extractor.conv.W = data["convW"]
        feature_extractor.conv.b = data["convB"]
        classifier.fc1.W = data["fc1W"]
        classifier.fc1.b = data["fc1B"]
        classifier.fc2.W = data["fc2W"]
        classifier.fc2.b = data["fc2B"]
        print(f"[INFO] Loaded weights from {ckpt_path}")
        return True
    else:
        print("[INFO] No saved weights found. Initializing randomly.")
        return False


def save_weights(feature_extractor, classifier, ckpt_path="weights_saved.npz"):
    np.savez(ckpt_path,
             convW=feature_extractor.conv.W,
             convB=feature_extractor.conv.b,
             fc1W=classifier.fc1.W,
             fc1B=classifier.fc1.b,
             fc2W=classifier.fc2.W,
             fc2B=classifier.fc2.b)
    print(f"[INFO] Weights saved to {ckpt_path}")


#-----------------------------------------------------------------------------

def _append_summary_csv(path, time_s, acc, loss):
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("time_s,accuracy,loss\n")
        f.write(f"{time_s:.6f},{acc:.6f},{loss:.6f}\n")

# def _save_config(path, cfg: dict):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(cfg, f, indent=2)

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
    # logits: (N, C), y: (N,)
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    prob = ez / ez.sum(axis=1, keepdims=True)

    N = y.shape[0]
    loss = -np.mean(z[np.arange(N), y] - np.log(ez.sum(axis=1)))

    dlogits = prob.copy()
    dlogits[np.arange(N), y] -= 1.0
    return loss, dlogits, prob



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


class Act:
    def __init__(self, kind="relu", alpha=0.01):
        self.kind  = kind
        self.alpha = float(alpha)
        self.z = None
        self.y = None
        self.grad = None  # per-activation derivative cache

    def forward(self, z):
        self.z = z

        if self.kind == "relu":
            self.y = np.maximum(0, z)
            self.grad = (z > 0).astype(z.dtype)

        elif self.kind == "leaky_relu":
            # forward = z if z>0 else alpha*z ; derivative = 1 or alpha
            self.y = np.where(z > 0, z, self.alpha * z)
            self.grad = np.where(z > 0, 1.0, self.alpha).astype(z.dtype)

        elif self.kind == "sigmoid":
            # numerically safer
            zc = np.clip(z, -50, 50)
            s = 1.0 / (1.0 + np.exp(-zc))
            self.y = s
            self.grad = s * (1.0 - s)

        elif self.kind == "tanh":
            t = np.tanh(z)
            self.y = t
            self.grad = 1.0 - t * t

        else:
            raise ValueError(f"Unknown activation kind: {self.kind}")

        return self.y

    def backward(self, dy, lrr=None):  # lrr is useless, kept only to not break your calls
        if self.grad is None:
            raise RuntimeError("Call forward() before backward().")
        return dy * self.grad

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
        self.relu = Act()
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
    def __init__(self, input_dim, num_classes=62, hidden_dim=128, lr=0.05):
        self.lr = lr
        self.num_classes = num_classes
        # Two-layer MLP: input -> hidden -> ReLU -> logits
        self.fc1 = Dense(input_dim, hidden_dim)
        self.act = Act()
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
        split="test", verbose=True
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
        # _save_config(os.path.join(run_dir, "config.json"), run_cfg)

    # ---------------- training loop (your proper mini-batch loop) ----------------
        acc , loss=0 ,0
        X_all, y_all = load_mnist_batch(split=split, emnist_split="balanced")  # or "letters"
        global pics , lables
        pics , lables = X_all , y_all

        N = X_all.shape[0]

        for ep in range(epochs):
            perm = np.random.permutation(N)
            X_all = X_all[perm]
            y_all = y_all[perm]
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
       
        _append_summary_csv(os.path.join("runs", "summary.csv"), elapsed, final_acc, final_loss)

        

                
# ---------------------
# Data loader (unchanged logic, grabs a random batch)
# ---------------------
def load_mnist_batch(split="train", emnist_split="balanced", cache_root="data"):
    """
    split: "train" or "test"
    emnist_split: "digits", "letters", "balanced", "byclass", "bymerge", or "mnist"
    """
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f"emnist_{emnist_split}_{split}.npz")
    # if os.path.exists(cache_path):
    #     d = np.load(cache_path)
    #     return d["X"], d["y"]

    import torch
    from torchvision import datasets, transforms

    # EMNIST comes rotated/transposed. Fix orientation to normal 28x28.
    def _fix(x):
        # x: [1,28,28] tensor; rotate 90° and flip to upright
        # return x.transpose(1, 2).flip(2)
        return torch.rot90(x, 3, [1, 2]).flip(2)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_fix)])

    ds = datasets.EMNIST(
        root=cache_root,
        split=emnist_split,
        train=(split == "train"),
        download=True,
        transform=tfm
        
    )
    def label_to_name(ds, y):
        # Try to get a human-friendly class name if the dataset provides it.
        name = str(y)
        try:
            # Some torchvision versions expose textual classes; fallback to id.
            if hasattr(ds, "classes") and len(ds.classes) > int(y):
                name = str(ds.classes[int(y)])
        except Exception:
            pass
        return name
    # import random
    # random.seed(1)
    # idxs = random.sample(range(len(ds)), 4 * 4)
    # fig, axes = plt.subplots(4, 4, figsize=(4 * 1.6, 4 * 1.6))
    # axes = axes.ravel()
    # for ax, i in zip(axes, idxs):
    #     x, y = ds[i]                 # x: [1,28,28] tensor in [0,1]
    #     ax.imshow(x[0].numpy(), cmap="gray", vmin=0, vmax=1)
    #     ax.set_title(label_to_name(ds, y), fontsize=9)
    #     ax.axis("off")

    # fig.suptitle(f"EMNIST split={"balanced"}, train={True}, orient={"ROT90_CW_FLIP_H"}", fontsize=12)
    # plt.tight_layout()
    # plt.show()
    

    xs = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)  # (N,1,28,28)
    ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)

    X = xs.numpy().astype(np.float32)
    y = ys.numpy().astype(np.int64)

    # One shuffle before caching so first use isn’t sorted by class id
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    np.savez(cache_path, X=X, y=y)
    return X, y


# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    np.random.seed(2)

    
    # Part 1: image processing pipeline
    extractor = FeatureExtractor(lr=0.05)

    # Part 2: classifier with adjustable hidden nodes
    cls = Classifier(
        input_dim=extractor.flat_dim,
        num_classes=62,
        hidden_dim=628,   # ← adjust this
        lr=0.01
    )

    # check and load weights
    weights_loaded = load_wights(extractor, cls)

    # Train controls: epochs and batch size are adjustable
    cls.train(
        feature_extractor=extractor,
        epochs=1,        # ← adjust epochs
        batch_size=300,    # ← adjust batch size
        split="train",
        verbose=True
    )
    save_weights(extractor, cls)
    print("Done.")
    # ====== Easy toggle after training ===========================================
    # Set to True when you want the viewer. False when you don't want to plot.
    ENABLE_MISCLASS_REVIEW = True
    if ENABLE_MISCLASS_REVIEW:
        # keep runs tidy if you want files saved too
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("runs", f"miscls_{ts}")
        review_misclassified(extractor, cls, split="train", save_dir=None, limit=None)
        # and inside review_misclassified:
        # X_all, y_all = load_mnist_batch(split="train", emnist_split="balanced")

