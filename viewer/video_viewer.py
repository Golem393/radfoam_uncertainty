import subprocess
import random
import argparse
import json
import os
import itertools
import random

BASE_CONFIGS = [
    {
        "radius": 4.2700000000000005, "elevation": -5.01, 
        "target_x_offset": 0.47, "target_y_offset": 0.067, "target_z_offset": -0.609, 
        "pitch_deg": -28.0, "yaw_deg": 0, "roll_deg": -7.7, "start_angle_deg": 10
    },
    {
        "radius": 2.636, "elevation": -6.204, 
        "target_x_offset": 0.143, "target_y_offset": 0.266, "target_z_offset": -0.343, 
        "pitch_deg": -14.74, "yaw_deg": 0, "roll_deg": -19.32, "start_angle_deg": 38
    },
    {
        "radius": 2.166, "elevation": -5.232, 
        "target_x_offset": 0.156, "target_y_offset": -0.038, "target_z_offset": -0.883, 
        "pitch_deg": -10.08, "yaw_deg": 0, "roll_deg": -19.2, "start_angle_deg": 55
    },
    {
        "radius": 3.583, "elevation": -5.369, 
        "target_x_offset": 0.43, "target_y_offset": -0.237, "target_z_offset": -0.887, 
        "pitch_deg": -16.59, "yaw_deg": 0, "roll_deg": -18.62, "start_angle_deg": 51
    },
    {
        "radius": 3.087, "elevation": -8, 
        "target_x_offset": 0.051, "target_y_offset": -0.151, "target_z_offset": -0.587, 
        "pitch_deg": -5.09, "yaw_deg": 0, "roll_deg": -15.32, "start_angle_deg": 52
    },
    {
        "radius": 2.87, "elevation": -7.03, 
        "target_x_offset": 0.731, "target_y_offset": 0.088, "target_z_offset": -0.643, 
        "pitch_deg": -5.5, "yaw_deg": 0, "roll_deg": -8.6, "start_angle_deg": 31
    }
]


PRESET_VARIANTS = BASE_CONFIGS.copy()


def random_param_combo():
    return PRESET_VARIANTS.pop(0)


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
    parser.add_argument('--log_file', type=str, default="generated_params.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    thresholds = [round(i * 0.1, 1) for i in range(11)]
    """def frange(start, stop, step):
        while start < stop + 1e-9:  # small epsilon to avoid float issues
            yield round(start, 10)
            start += step

    thresholds = [round(i, 2) for i in 
              list(frange(0.1, 0.45, 0.05)) + 
              list(frange(0.6, 1.01, 0.2))]"""


    with open(args.log_file, "w") as f:
        variant_index = 0
        for threshold in thresholds:
            global PRESET_VARIANTS
            PRESET_VARIANTS = BASE_CONFIGS.copy()

            for i in range(len(BASE_CONFIGS)):
                params = random_param_combo()
                params["threshold"] = threshold
                param_str = run_variant(
                    variant_id=f"{variant_index}_t{threshold:.1f}",
                    params=params,
                    target_script=args.target_script
                )
                f.write(f"video360_{variant_index}_t{threshold:.1f}.mp4: {param_str}\n")
                variant_index += 1

if __name__ == "__main__":
    main()
