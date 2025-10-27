import numpy as np
import matplotlib.pyplot as plt

# nonlinear data
X = np.linspace(-3, 3, 100, dtype=np.float32)
Y = X**2 + np.random.randn(*X.shape) * 0.3  # add noise

# convert to matrix form (N,1)
X = X.reshape(-1, 1)

# initialize parameters
W = np.zeros(1, dtype=np.float32)
b = 0.0
lr = 0.01
epochs = 300

def forward(X):
    return X @ W + b  # linear model

def loss(y, y_pred):
    return np.mean((y_pred - y)**2)

losses = []
for ep in range(epochs):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    grad_w = (2/len(X)) * (X.T @ (y_pred - Y))
    grad_b = 2 * np.mean(y_pred - Y)
    W -= lr * grad_w
    b -= lr * grad_b
    losses.append(l)

print(f"Final loss: {losses[-1]:.4f}")

# Plot results
plt.figure(figsize=(6,4))
plt.scatter(X, Y, color='blue', label='True (y = x^2)')
plt.plot(X, forward(X), color='red', label='Linear fit')
plt.legend()
plt.title("Linear model vs quadratic data")
plt.show()
