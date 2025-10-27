import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def viz_mlp_triple(
    X=None, Y=None,
    H=4, lr=0.01, batch_size=64, epochs=1500, seed=0,
    activation="sigmoid",
    # --- blink controls (event-driven) ---
    base_alpha=0.01,   # minimum (faded) alpha
    max_alpha=1,    # cap the flash
    blink_gain=1,    # how much a weight change bumps alpha
    blink_decay=0.6,  # per-frame decay of alpha (0.8–0.95 works well)
    ema_beta=0,      # smooth |Δw| before feeding blink
    node_radius=0.16, left_x=-1.0, right_x=3.0
):
    np.random.seed(seed)

    # ---------- data ----------
    if X is None or Y is None:
        X = np.linspace(-3, 3, 200).astype(np.float32).reshape(-1,1)
        Y = (X[:,0]**2).astype(np.float32).reshape(-1,1)
    else:
        X = np.asarray(X, dtype=np.float32).reshape(-1, 1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1, 1)
    Xn = (X - X.mean())/X.std()
    Xgrid = np.linspace(X.min(), X.max(), 400, dtype=np.float32).reshape(-1,1)
    Xgrid_n = (Xgrid - X.mean())/X.std()

    # ---------- model ----------
    def he(f_in, f_out): return np.random.randn(f_in, f_out).astype(np.float32) * np.sqrt(2.0/f_in)
    W1 = he(1, H); b1 = np.zeros((1, H), np.float32)
    W2 = he(H, 1); b2 = np.zeros((1, 1), np.float32)

    W1_prev, W2_prev = W1.copy(), W2.copy()
    dW1_ema = np.zeros_like(W1); dW2_ema = np.zeros_like(W2)

    # activations
    def relu(z): return np.maximum(0, z)
    def leaky(z, a=0.01): return np.where(z>0, z, a*z)
    def sigmoid(z): return 1/(1+np.exp(-z))
    def tanh(z): return np.tanh(z)

    def d_relu(a): return (a > 0).astype(np.float32)
    def d_leaky(z, a=0.01): g = np.ones_like(z); g[z<=0] = a; return g
    def d_sigmoid(a): return a*(1-a)
    def d_tanh(a): return 1 - a**2

    acts = {
        "relu": (relu, d_relu),
        "sigmoid": (sigmoid, d_sigmoid),
        "tanh": (tanh, d_tanh),
        "leakyrelu": (lambda z: leaky(z,0.01), lambda z: d_leaky(z,0.01)),
    }
    act, d_act = acts.get(activation.lower(), (relu, d_relu))

    def forward(Xin):
        Z1 = Xin @ W1 + b1
        A1 = act(Z1)
        Yh = A1 @ W2 + b2
        return Z1, A1, Yh

    # ---------- figure ----------
    fig = plt.figure(figsize=(14,4))
    ax_graph  = plt.subplot(1,3,1); ax_graph.set_title(f"Network (1-{H}-1, {activation})")
    ax_loss   = plt.subplot(1,3,2); ax_loss.set_title("Loss"); ax_loss.set_xlabel("step"); ax_loss.set_ylabel("MSE")
    ax_output = plt.subplot(1,3,3); ax_output.set_title("Model output"); ax_output.set_xlabel("x"); ax_output.set_ylabel("y")

    ax_graph.set_aspect("equal", adjustable="box"); ax_graph.axis("off")
    pos = {"x": (left_x, 0.0), **{f"h{i}": (1.0, 1.5 - i) for i in range(H)}, "y": (right_x, 0.0)}
    ax_graph.set_xlim(left_x - 0.6, right_x + 0.6); ax_graph.set_ylim(-2.0, 2.0)

    nodes, center_txt, left_txt, right_txt = {}, {}, {}, {}
    for name,(px,py) in pos.items():
        face = "black" if name.startswith("h") else "white"
        c = plt.Circle((px,py), node_radius, facecolor=face, edgecolor="black", lw=2.0)
        ax_graph.add_patch(c); nodes[name] = c
        center_txt[name] = ax_graph.text(px, py, "", ha="center", va="center",
                                         fontsize=10, color=("white" if name.startswith("h") else "black"))
        if name.startswith("h"):
            left_txt[name]  = ax_graph.text(px - node_radius - 0.12, py, "", ha="right", va="center", fontsize=9)
            right_txt[name] = ax_graph.text(px + node_radius - 0.12 + 0.24, py, "", ha="left",  va="center", fontsize=9)

    # edges (constant thin black)
    edges_W1 = [ax_graph.plot([pos["x"][0], pos[f"h{i}"][0]],
                              [pos["x"][1], pos[f"h{i}"][1]],
                              lw=1.5, alpha=base_alpha, color="black")[0] for i in range(H)]
    edges_W2 = [ax_graph.plot([pos[f"h{i}"][0], pos["y"][0]],
                              [pos[f"h{i}"][1], pos["y"][1]],
                              lw=1.5, alpha=base_alpha, color="black")[0] for i in range(H)]

    # per-edge alpha state (event-driven blink)
    alpha_W1 = np.full((H,), base_alpha, dtype=np.float32)
    alpha_W2 = np.full((H,), base_alpha, dtype=np.float32)

    # loss + output
    loss_vals = []; (loss_line,) = ax_loss.plot([], [], color="black")
    ax_output.scatter(X[:,0], Y[:,0], s=12, c="black", alpha=0.5, label="data")
    (pred_line,) = ax_output.plot([], [], lw=2, color="black", label="prediction")
    ax_output.legend(loc="best")

    # ---------- minibatches ----------
    def batches(X,Y,b):
        n = len(X)
        while True:
            idx = np.random.permutation(n)
            for i in range(0, n, b):
                j = idx[i:i+b]
                yield X[j], Y[j]
    batch_iter = batches(Xn, Y, batch_size)

    step = 0
    def train_step():
        nonlocal W1, b1, W2, b2, step, W1_prev, W2_prev, dW1_ema, dW2_ema
        Xb, Yb = next(batch_iter)
        Z1, A1, Yh = forward(Xb)
        L = np.mean((Yh - Yb)**2)

        N = len(Xb)
        dZ2 = (2.0/N) * (Yh - Yb)
        dW2 = A1.T @ dZ2; db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * d_act(A1)
        dW1 = Xb.T @ dZ1; db1 = np.sum(dZ1, axis=0, keepdims=True)

        # update
        W2 -= lr*dW2; b2 -= lr*db2
        W1 -= lr*dW1; b1 -= lr*db1

        # smooth |Δw| for blinking
        dW1_ema = ema_beta*dW1_ema + (1-ema_beta)*np.abs(W1 - W1_prev)
        dW2_ema = ema_beta*dW2_ema + (1-ema_beta)*np.abs(W2 - W2_prev)
        W1_prev, W2_prev = W1.copy(), W2.copy()

        # full curve for right pane
        _, A1g, Yg = forward(Xgrid_n)

        step += 1
        return float(L), A1, Yh, dW1_ema, dW2_ema, Yg

    def init_anim():
        ax_loss.set_xlim(0, 100); ax_loss.set_ylim(0, 10)
        loss_line.set_data([], []); pred_line.set_data([], [])
        artists = edges_W1 + edges_W2 + [loss_line, pred_line] + list(center_txt.values())
        artists += list(left_txt.values()) + list(right_txt.values())
        return artists

    def normalize(x, eps=1e-8):
        m = np.max(x);  return x/(m+eps) if m>0 else x

    def update_anim(fi):
        nonlocal alpha_W1, alpha_W2
        L, A1_b, Yh_b, dW1e, dW2e, Yg = train_step()
        loss_vals.append(L)
        # loss
        if len(loss_vals) > 5:
            ax_loss.set_xlim(0, max(100, len(loss_vals)))
            window = loss_vals[-200:] if len(loss_vals)>200 else loss_vals
            ymin, ymax = min(window), max(window)
            if ymax == ymin: ymax += 1e-3
            ax_loss.set_ylim(ymin*0.9, ymax*1.1)
        loss_line.set_data(range(len(loss_vals)), loss_vals)
        # output
        pred_line.set_data(Xgrid[:,0], Yg[:,0])

        # --- event-driven blink: bump on change, then decay ---
        # normalize per-edge update magnitudes
        n1 = normalize(dW1e[0])      # shape (H,)
        n2 = normalize(dW2e[:,0])    # shape (H,)

        # bump
        alpha_W1 = np.minimum(max_alpha, alpha_W1 + blink_gain * n1)
        alpha_W2 = np.minimum(max_alpha, alpha_W2 + blink_gain * n2)
        # decay
        alpha_W1 = np.maximum(base_alpha, alpha_W1 * blink_decay)
        alpha_W2 = np.maximum(base_alpha, alpha_W2 * blink_decay)

        # apply to edges; update labels
        act_mean = np.mean(A1_b, axis=0).ravel()
        for i in range(H):
            edges_W1[i].set_alpha(float(alpha_W1[i]))
            edges_W2[i].set_alpha(float(alpha_W2[i]))
            center_txt[f"h{i}"].set_text(f"{act_mean[i]:.2f}")
            left_txt[f"h{i}"].set_text(f"{W1[0,i]:+.2f}")
            right_txt[f"h{i}"].set_text(f"{W2[i,0]:+.2f}")

        center_txt["y"].set_text(f"{float(np.mean(Yh_b)):.2f}")
        center_txt["x"].set_text("")
        return edges_W1 + edges_W2 + [loss_line, pred_line] + list(center_txt.values()) + list(left_txt.values()) + list(right_txt.values())

    ani = FuncAnimation(fig, update_anim, init_func=init_anim, frames=epochs, interval=50, blit=False)
    plt.tight_layout(); plt.show()
    return ani

# Example:
ani = viz_mlp_triple(H=7, lr=0.1, batch_size=64, epochs=1200)
