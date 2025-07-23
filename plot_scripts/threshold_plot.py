import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

metrics_dir = "output_final/normal_setting2"
metrics_files = [f for f in os.listdir(metrics_dir) if f.startswith("metrics_model_pruned_") and f.endswith(".pt.txt")]
metrics_files.sort(key=lambda x: float(re.search(r"pruned_(\d*\.?\d+)", x).group(1)))  # sort by pruning value

point_counts = {}
with open(os.path.join(metrics_dir, "point_counts.txt")) as f:
    for line in f:
        if not line.strip():
            continue
        name, count = line.strip().split(":")
        point_counts[name.strip()] = int(count.strip().split()[0])

def parse_metrics(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    values = {}
    for line in lines:
        if line.startswith("Average"):
            key, val = line.split(":")
            values[key.strip()] = float(val.strip())
    return values

original_metrics = parse_metrics(os.path.join(metrics_dir, "metrics.txt"))
original_points = point_counts["model.pt"]

thresholds = []
points = []
metrics_dict = {
    "PSNR": [],
    "LPIPS": [],
    "SSIM": []
}

for fname in metrics_files:
    thresh = float(re.search(r"pruned_(\d*\.?\d+)", fname).group(1))
    model_name = f"model_pruned_{thresh}.pt"
    if model_name not in point_counts:
        continue
    metrics = parse_metrics(os.path.join(metrics_dir, fname))
    thresholds.append(thresh)
    points.append(point_counts[model_name])
    for key in metrics_dict:
        metrics_dict[key].append(metrics[f"Average {key}"])

sorted_data = sorted(zip(thresholds, points, *metrics_dict.values()))
thresholds, points, *metric_lists = zip(*sorted_data)

for i, (metric_name, values) in enumerate(zip(metrics_dict.keys(), metric_lists)):
    plt.figure(figsize=(10, 6))
    sizes = [p / max(points) * 800 for p in points]

    plt.scatter(thresholds, values, s=sizes, alpha=0.7, edgecolors='k', label='Pruned Models')

    for x, y, p in zip(thresholds, values, points):
        plt.annotate(f"{p // 1000}k", (x, y), textcoords="offset points", xytext=(0, 18),
                     ha='center', fontsize=20)

    x_smooth = np.linspace(min(thresholds), max(thresholds), 200)
    y_smooth = make_interp_spline(thresholds, values, k=3)(x_smooth)
    plt.plot(x_smooth, y_smooth, color='C0', linewidth=4, label='Trend')

    baseline_value = original_metrics[f"Average {metric_name}"]
    plt.axhline(y=baseline_value, color='gray', linestyle='--', linewidth=2, label='Original Model')
    plt.annotate(f"{original_points // 1000}k points", xy=(0, baseline_value),
                 xytext=(0.02, baseline_value + 0.01), textcoords='data',
                 fontsize=20, color='gray')

    plt.title(f"{metric_name} vs Pruning Threshold", fontsize=26)
    plt.xlabel("Pruning Threshold", fontsize=22)
    plt.ylabel(metric_name, fontsize=22)

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()