import torch
import configargparse
from configs import *
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
import radfoam
import numpy as np
from data_loader import DataHandler

from utils.utils import find_grid_indices
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components import renderers

from PIL import Image

class UncertaintyViewer:
    def __init__(self, args, model_args, dataset_args):
        self.device = torch.device(args.device)
        self.model = RadFoamScene(args=model_args, device=self.device)
        self.model.load_pt(args.checkpoint_path)
        self.model.eval()

        self.hessian = torch.tensor(np.load(args.hessian_path)).to(self.device)
        self.lod = np.log2(round(self.hessian.shape[0] ** (1 / 3)) - 1)

        self.data_handler = DataHandler(dataset_args, rays_per_batch=0, device=self.device)
        self.data_handler.reload(split="test", downsample=min(dataset_args.downsample))

        self.ray_batch_fetcher = radfoam.BatchFetcher(self.data_handler.rays, batch_size=1, shuffle=False)
        self.rgb_batch_fetcher = radfoam.BatchFetcher(self.data_handler.rgbs, batch_size=1, shuffle=False)
        width, height = 3118, 2078
        downscale_factor = 2.0
        self.N = 1000 * ((width * height) / downscale_factor)

    def get_uncertainty(self, points, un):
        inds, coeffs = find_grid_indices(points, self.lod, points.device)
        cfs_2 = (coeffs ** 2) / torch.sum((coeffs ** 2), dim=0, keepdim=True)
        uns = un[inds.long()].squeeze()
        un_points = torch.sqrt(torch.sum((uns * cfs_2), dim=0)).unsqueeze(1)
        return torch.log10(un_points + 1e-12)

    def get_outputs(self, ray_batch):
        reg_lambda = 1e-4 / ((2 ** self.lod) ** 3)
        H = self.hessian / self.N + reg_lambda
        un = 1 / H
        primal_points = self.model.primal_points.clone().detach()
        un_points = self.get_uncertainty(primal_points, un).view(-1)
        un_points_cp = (un_points - un_points.min()) / (un_points.max() - un_points.min())
        depth_quantiles = torch.rand(*ray_batch.shape[:-1], 2, device=self.device).sort(dim=-1, descending=True).values
        rgba_output, depth, ray_samples, _, _ = self.model(
            ray_batch,
            depth_quantiles=depth_quantiles,
            uncertainty=un_points_cp
        )
        return rgba_output

    def render_uncertainty_images(self, output_dir="output"):
        rays = self.data_handler.rays
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = self.ray_batch_fetcher.next()[0]
                rgb_batch = self.rgb_batch_fetcher.next()[0]
                alpha_batch = torch.ones_like(rgb_batch[..., :1])
                output = self.get_outputs(ray_batch)
                uncertainty = output[..., -1:].detach().cpu().numpy().squeeze(-1)
                uncertainty_img = np.uint8(np.clip(uncertainty, 0, 1) * 255)
                Image.fromarray(uncertainty_img, mode="L").save(f"{output_dir}/uncertainty{i}.png")


def parse_arguments():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )
    parser.add_argument("--checkpoint_path", type=str, default="output/bonsai@44a5858d/model.pt")
    parser.add_argument("--hessian_path", type=str, default="output/bonsai@44a5858d/unc.npy")
    return parser


def main():
    parser = parse_arguments()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)
    args = parser.parse_args()
    viewer = UncertaintyViewer(args, model_params.extract(args), dataset_params.extract(args))
    viewer.render_uncertainty_images()


if __name__ == "__main__":
    main()