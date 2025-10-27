import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

# Layers
input_nodes = ["x"]
hidden_nodes = [f"h{i}" for i in range(1, 17)]
output_node = ["ŷ"]

# Edges
for i in input_nodes:
    for h in hidden_nodes:
        G.add_edge(i, h)
for h in hidden_nodes:
    G.add_edge(h, "ŷ")

# Plot
plt.figure(figsize=(8,4))
pos = {}
pos["x"] = (0,0)
for i,h in enumerate(hidden_nodes):
    pos[h] = (1, i-8)
pos["ŷ"] = (2,0)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color="#add8e6", arrows=False)
plt.title("MLP Architecture: 1 Input → 16 Hidden (ReLU) → 1 Output")
plt.axis("off")
plt.show()
