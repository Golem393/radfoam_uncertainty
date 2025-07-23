import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15
})


N = 100
log_dir = "output_final/andunc03"
output_file = "downscaled4_weighted_loss.png"

files = [
    ("uncertainty_loss_log56.txt", "Uncertainty (×0.001)", "blue", 0.001),
    ("color_loss_log56.txt", "Color", "green", 1.0),
    ("opacity_loss_log56.txt", "Opacity", "red", 1.0),
    ("quant_loss_log56.txt", "Quant", "purple", 1.0)
]

plt.figure(figsize=(10, 6))

for filename, label, color, scale in files:
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        continue

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    try:
        data = np.array([list(map(float, line.split())) for line in lines])
    except ValueError as e:
        print(f"Error parsing {filename}: {e}")
        continue

    if data.shape[1] < 2:
        print(f"Skipping {filename}, not enough columns.")
        continue

    epochs = data[:, 0]
    losses = data[:, 1] * scale

    n_chunks = len(losses) // N
    if n_chunks == 0:
        print(f"Skipping {filename}, not enough data to average.")
        continue

    epochs_avg = np.mean(epochs[:n_chunks * N].reshape(-1, N), axis=1)
    losses_avg = np.mean(losses[:n_chunks * N].reshape(-1, N), axis=1)

    plt.plot(
        epochs_avg, losses_avg,
        label=label,
        color=color,
        marker='o', markersize=3,
        linewidth=1.5, alpha=0.8
    )

plt.ylim(0.0000, 0.0010)
plt.xlim(0, 20000)
plt.title("Training Loss Components Over Epochs (Uncertainty Scaled ×0.001)")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()