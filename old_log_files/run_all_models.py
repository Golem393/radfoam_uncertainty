
"""import subprocess

model_dir = "./output_final/normal_setting2"
test_script = "test.py"

for fname in sorted(os.listdir(model_dir)):
    if fname.startswith("model_pruned_") and fname.endswith(".pt"):
        print(f"Testing model: {fname}")
        subprocess.run([
            "python", test_script,
            f"--modelfile={fname}"
        ])"""

import os
import torch
import configargparse
from configs import *
from radfoam_model.scene import RadFoamScene

def parse_arguments():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )
    return parser

model_dir = "./output_final/normal_setting2"
output_file = "point_counts.txt"

model_files = sorted([
    f for f in os.listdir(model_dir)
    if f.startswith("model") and f.endswith(".pt")
])

results = []

print(f"Found {len(model_files)} models")

for fname in model_files:
    full_path = os.path.join(model_dir, fname)

    parser = parse_arguments()
    model_params = ModelParams(parser)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = RadFoamScene(args=model_params.extract(args), device=device)
    model.load_pt(full_path)
    
    num_points = model.primal_points.shape[0]
    print(f"{fname}: {num_points} points")
    results.append(f"{fname}: {num_points} points")
    

with open(output_file, "w") as f:
    f.write("\n".join(results))

print(f"\nResults saved to {output_file}")

