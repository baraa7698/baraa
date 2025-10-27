import numpy as np
import matplotlib.pyplot as plt

# data
np.random.seed(0)
X = np.linspace(-3, 3, 200).astype(np.float32).reshape(-1,1)
Y = (X[:,0]**2 + 0.3*np.random.randn(len(X))).astype(np.float32).reshape(-1,1)

# normalize inputs (helps training)
Xn = (X - X.mean())/X.std()

# model: 1-hidden-layer MLP: ŷ = W2 * act(W1 X + b1) + b2
H = 64  # hidden width; try 8–32
W1 = 0.1*np.random.randn(1, H).astype(np.float32)   # (in, H)
b1 = np.zeros((1, H), np.float32)
W2 = 0.1*np.random.randn(H, 1).astype(np.float32)   # (H, out)
b2 = np.zeros((1, 1), np.float32)

def relu(z): return np.maximum(0, z)
def d_relu(a): return (a > 0).astype(np.float32)

def forward(X):
    Z1 = X @ W1 + b1          # (N,H)
    A1 = relu(Z1)             # (N,H)
    Yh = A1 @ W2 + b2         # (N,1) linear output for regression
    return Z1, A1, Yh

def mse(y, yhat): return np.mean((yhat - y)**2)

lr = 0.01
epochs = 5000
losses = []

for ep in range(epochs):
    Z1, A1, Yh = forward(Xn)
    L = mse(Y, Yh); losses.append(L)

    # backprop
    N = len(Xn)
    dYh = (2.0/N) * (Yh - Y)          # dL/dYh
    dW2 = A1.T @ dYh                  # (H,1)
    db2 = np.sum(dYh, axis=0, keepdims=True)
    dA1 = dYh @ W2.T                  # (N,H)
    dZ1 = dA1 * d_relu(A1)            # (N,H)
    dW1 = Xn.T @ dZ1                  # (1,H)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # update
    W2 -= lr*dW2; b2 -= lr*db2
    W1 -= lr*dW1; b1 -= lr*db1

    if ep % 500 == 0:
        print(f"{ep:4d}  loss={L:.4f}")

print(f"final loss: {losses[-1]:.4f}")

# plots
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], Y[:,0], s=10, label="y=x^2 + noise")
_, _, Yfit = forward(Xn)
plt.plot(X[:,0], Yfit[:,0], lw=2, label="MLP fit (ReLU, H=16)")
plt.legend(); plt.title("MLP fits parabola"); plt.show()

plt.figure(figsize=(5,3))
plt.plot(losses); plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.show()
