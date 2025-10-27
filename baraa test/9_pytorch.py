# pytorch_vs_numpy_style_cnn.py
import time, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class TinyCNN(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)  # 28→26
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                # 26→13
        self.fc   = nn.Linear(8*13*13, 10)

        act = activation.lower()
        if act == "relu":
            self.act = F.relu
            self._act_name = "ReLU"
        elif act == "leakyrelu":
            self.leaky = nn.LeakyReLU(0.01, inplace=True)
            self.act = lambda x: self.leaky(x)
            self._act_name = "LeakyReLU(0.01)"
        elif act == "tanh":
            self.act = torch.tanh
            self._act_name = "tanh"
        elif act == "sigmoid":
            self.act = torch.sigmoid
            self._act_name = "sigmoid"
        else:
            raise ValueError("activation must be one of: relu, leakyrelu, tanh, sigmoid")

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # raw logits

def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total

def train_steps(model, loader, device, steps=30, lr=0.05, log_every=5):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    seen = 0
    t0 = time.time()
    per_step_times = []

    it = iter(loader)
    for step in range(1, steps+1):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader)
            X, y = next(it)

        X, y = X.to(device), y.to(device)

        t_step0 = time.time()
        opt.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        t_step1 = time.time()

        seen += X.size(0)
        per_step_times.append(t_step1 - t_step0)

        if step % log_every == 50:
            with torch.no_grad():
                pred = logits.argmax(1)
                acc = (pred == y).float().mean().item()
            print(f"step {step*60:02d}/{steps*60} | batch_loss {loss.item():.4f} | batch_acc {acc*100:5.1f}% | step_time {per_step_times[-1]*1000:6.1f} ms")

    total_time = time.time() - t0
    imgs_per_sec = seen / total_time if total_time > 0 else float('nan')
    print(f"\nSeen {seen} images in {total_time:.2f}s  →  {imgs_per_sec:.1f} img/s (mean step {np.mean(per_step_times)*1000:.1f} ms)")
    return total_time, imgs_per_sec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=60)
    ap.add_argument("--steps", type=int, default=1000, help="number of training minibatches (to mirror your NumPy run)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--activation", type=str, default="relu", choices=["relu","leakyrelu","tanh","sigmoid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) or args.device=="cuda" else "cpu")
    print(f"Device: {device}")

    # Data
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST("./data", train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type=="cuda")
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST("./data", train=False, transform=transform),
        batch_size=1000, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda")
    )

    # Model
    model = TinyCNN(activation=args.activation).to(device)
    print(model)
    print(f"Activation: {model._act_name}")
    print(f"Total parameters: {count_params(model):,}")

    # Train for a fixed number of steps to line up with your NumPy loop
    total_time, ips = train_steps(model, train_loader, device, steps=args.steps, lr=args.lr)

    # Quick eval
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest loss {test_loss:.4f} | Test acc {test_acc*100:5.2f}%")

if __name__ == "__main__":
    # Lazy import inside main to avoid errors at top-level
    from torchvision import transforms
    main()
