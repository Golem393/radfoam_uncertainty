import subprocess
import random
import argparse
import json
import os
import itertools
import random


def run_variant(variant_id, params, target_script):
    param_str = json.dumps(params)
    print(f"Variant {variant_id} with params {param_str} executed successfully.")
    subprocess.run([
        "python", target_script,
        f"--variant_id={variant_id}",
        f"--additional_params={param_str}"
    ])
    return param_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_script', type=str, required=True)
    args = parser.parse_args()


    thresholds = [round(i * 0.05, 2) for i in range(21)]


    variant_index = 0
    for threshold in thresholds:
        print(f"Running variant with threshold: {threshold}")
        params = {}
        params["threshold"] = threshold
        param_str = run_variant(
            variant_id=f"{variant_index}_t{threshold:.1f}",
            params=params,
            target_script=args.target_script
        )
        variant_index += 1

if __name__ == "__main__":
    main()
