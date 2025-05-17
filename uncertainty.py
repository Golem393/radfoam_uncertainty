from pathlib import Path
import time
import torch
from configs import *
from radfoam_model.scene import RadFoamScene
from nerfstudio.field_components.encodings import HashEncoding
from data_loader import DataHandler
from utils.utils import (find_grid_indices, normalize_point_coords)
import tqdm
from radfoam_model.render import TraceRays


parser = configargparse.ArgParser(
    default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
)
dataset_params = DatasetParams(parser)
args = parser.parse_args()
device = torch.device(args.device)

train_data_handler = DataHandler(
    dataset_params.extract(args), rays_per_batch=250_000, device=device
)

train_data_handler.reload(split="train")

len_train = len(train_data_handler)
print(len_train)
for i in range(len_train):
    print("step", i)
    ray_batch, rgb_batch, alpha_batch = train_data_handler.get_iter(random=True, index=i)
    print(ray_batch)
