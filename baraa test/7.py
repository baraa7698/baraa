# numpy_cnn_mnist_split.py
import numpy as np

# ---------------------
# Utils (unchanged style)
# ---------------------
def one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

def softmax_logits(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(prob, y_onehot):
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(prob + eps), axis=1))

# ---------------------
# Core layers (your originals)
# ---------------------
class Conv2D:
    def __init__(self, C_in, C_out, k=3):
        limit = np.sqrt(2.0/(C_in*k*k))
        self.W = np.random.randn(C_out, C_in, k, k).astype(np.float32) * limit
        self.b = np.zeros((C_out, 1), dtype=np.float32)
        self.k = k

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        k = self.k
        outH, outW = H - k + 1, W - k + 1
        y = np.zeros((N, self.W.shape[0], outH, outW), dtype=np.float32)
        for n in range(N):
            for co in range(self.W.shape[0]):
                for i in range(outH):
                    for j in range(outW):
                        patch = x[n, :, i:i+k, j:j+k]
                        y[n, co, i, j] = np.sum(patch * self.W[co]) + self.b[co]
        self.y = y
        return y

    def backward(self, dy, lr):
        x = self.x
        N, C_in, H, W = x.shape
        C_out, _, k, _ = self.W.shape
        outH, outW = H - k + 1, W - k + 1

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(x)

        for n in range(N):
            for co in range(C_out):
                for i in range(outH):
                    for j in range(outW):
                        grad = dy[n, co, i, j]
                        db[co] += grad
                        patch = x[n, :, i:i+k, j:j+k]
                        dW[co] += grad * patch
                        dx[n, :, i:i+k, j:j+k] += grad * self.W[co]

        dW /= N
        db /= N
        self.W -= lr * dW
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
        self.x = x
        N, C, H, W = x.shape
        H2, W2 = H//2, W//2                              # // is integer division 5//2 = 2
        self.mask = np.zeros_like(x, dtype=np.float32)
        y = np.zeros((N,C,H2,W2), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(H2):
                    for j in range(W2):
                        window = x[n,c, 2*i:2*i+2, 2*j:2*j+2]
                        m = np.max(window)
                        y[n,c,i,j] = m
                        a, b = divmod(np.argmax(window), 2)
                        self.mask[n,c,2*i+a,2*j+b] = 1.0
        return y

    def backward(self, dy, lr):
        N,C,H,W = self.x.shape
        H2, W2 = H//2, W//2
        dx = np.zeros_like(self.x, dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(H2):
                    for j in range(W2):
                        a, b = np.where(self.mask[n,c, 2*i:2*i+2, 2*j:2*j+2] == 1.0)
                        if a.size:
                            dx[n,c, 2*i+a[0], 2*j+b[0]] = dy[n,c,i,j]
        self.mask[...] = 0.0
        return dx

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
        X_all, y_all = load_mnist_batch( split=split)
        N = X_all.shape[0]
        for ep in range(epochs):
            for steps in range(0, N, batch_size):
                id = steps+batch_size+1
                
                X, y = X_all[steps:id], y_all[steps:id]
                # forward through extractor and classifier
                feats = feature_extractor.forward(X)
                logits = self.forward(feats)
                prob = softmax_logits(logits)
                yoh = one_hot(y, self.num_classes)
                loss = cross_entropy(prob, yoh)

                # grad wrt logits (no 1/N here; Dense averages internally)
                dlogits = (prob - yoh)

                # backward through classifier and extractor
                dfeatures = self.backward(dlogits)
                feature_extractor.backward(dfeatures)

                pred = prob.argmax(axis=1)
                acc = np.mean(pred == y)
                if verbose:
                    print(f"epoch {ep+1:02d}, batch:{steps:05d}/{N:05d}  loss {loss:.3f}  acc {acc*100:5.1f}% ")

# ---------------------
# Data loader (unchanged logic, grabs a random batch)
# ---------------------
def load_mnist_batch( split="train"):
    import torch
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root="./data", train=(split=="train"), download=True,
                        transform=transforms.ToTensor())
    idx = torch.randperm(len(ds))
    xs = []
    ys = []
    for i in idx:
        x, y = ds[i]
        xs.append(x.numpy())        # (1,28,28)
        ys.append(int(y))
    X = np.stack(xs, axis=0).astype(np.float32)  # (N,1,28,28)
    y = np.array(ys, dtype=np.int64)
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
        hidden_dim=128,   # ← adjust this
        lr=0.05
    )

    # Train controls: epochs and batch size are adjustable
    cls.train(
        feature_extractor=extractor,
        epochs=10,        # ← adjust epochs
        batch_size=64,    # ← adjust batch size
        split="train",
        verbose=True
    )

    print("Done.")
