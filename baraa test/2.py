import numpy as np

# Fake data with 3 features
X = np.array([[1,2,3],
              [2,0,1],
              [3,1,2],
              [4,2,0]], dtype=np.float32)    # shape (N=4, m=3)
y = np.array([14, 7, 12, 10], dtype=np.float32)  # any targets

N, m = X.shape
W = np.zeros(m, dtype=np.float32)
b = 0.0
# print(W)
def forward(X):
    return X @ W + b  # shape (N,)

def loss(y, y_pred):
    return np.mean((y_pred - y)**2)

lr = 0.01
epochs = 2000
for ep in range(epochs):
    y_pred = forward(X)
    r = y_pred - y
    dW = (2.0/N) * (X.T @ r)     # vectorized gradient
    db = 2.0 * np.mean(r)
    W -= lr * dW
    b -= lr * db
    if ep % 200 == 0:
        print(f"{ep:4d}  loss={loss(y,y_pred):.6f}  W={W}  b={b}")

print("final:", W, b)
