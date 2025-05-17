import torch
from configs import *
from data_loader import DataHandler


parser = configargparse.ArgParser(
    default_config_files=["arguments/mipnerf360_outdoor_config.yaml"])

model_params = ModelParams(parser)
dataset_params = DatasetParams(parser)
args = parser.parse_args()
print(args)
print(args.device)
device = torch.device(args.device)

train_data_handler = DataHandler(
    dataset_params.extract(args), rays_per_batch=250_000, device=device
)
iter2downsample = dict(
            zip(
                dataset_params.extract(args).downsample_iterations,
                dataset_params.extract(args).downsample,
            )
        )
downsample = iter2downsample[0]
train_data_handler.reload(split="train", downsample=downsample)

data_iterator = train_data_handler.get_iter()

len_train = len(train_data_handler)
print(len_train)

for i in range(len_train):
    print("step", i)
    ray_batch, rgb_batch, alpha_batch = next(data_iterator)
    print(ray_batch)
