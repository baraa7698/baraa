import numpy as np 
import matplotlib.pyplot as plt
# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.int32)
Y = np.array([2, 4, 6, 8], dtype=np.int32)

w = 0
b = 0
# model output
def forward(x):
    w_hat = w*x +b
    return w_hat

# loss = MSE
def loss(y, y_pred):
    
    return ((y_pred - y)**2).mean()

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    dw = np.mean(2*x*(y_pred - y))
    db = np.mean(2*(y_pred - y))
    return dw,db

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.1
n_iters = 20
losses = []
for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    losses.append(l)  
    dw,db = gradient(X, Y, y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, dw:{dw:.3f} ,b = {b:.3f}, db:{db:.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}')

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over time")
plt.show()