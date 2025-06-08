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
from viewer.uncertainty_360 import Uncertainty360Viewer

from PIL import Image

class UncertaintyViewer:
    def __init__(self, args, model_args, dataset_args, optimization_args, pipeline_args):
        self.pipeline_args = pipeline_args
        self.optimization_args = optimization_args
        self.model_args = model_args
        self.device = torch.device(args.device)
        self.model = RadFoamScene(args=model_args, device=self.device)
        self.model.load_pt(args.checkpoint_path)
        self.model.eval()

        self.model.declare_optimizer(
            args=optimization_args,
            warmup=pipeline_args.densify_from,
            max_iterations=pipeline_args.iterations,
        )

        self.hessian = torch.tensor(np.load(args.hessian_path)).to(self.device)
        self.lod = np.log2(round(self.hessian.shape[0] ** (1 / 3)) - 1)

        self.data_handler = DataHandler(dataset_args, rays_per_batch=0, device=self.device)
        self.data_handler.reload(split="test", downsample=min(dataset_args.downsample))

        self.ray_batch_fetcher = radfoam.BatchFetcher(self.data_handler.rays, batch_size=1, shuffle=False)
        self.rgb_batch_fetcher = radfoam.BatchFetcher(self.data_handler.rgbs, batch_size=1, shuffle=False)
        self.filter_out_in_image = False
        self.filter_out_cells = False
        self.show_uncertainty = True
        self.create_360_video = True
        self.filter_thresh = 0.8
        width, height = 3118, 2078
        downscale_factor = 2.0
        self.N = 1000 * ((width * height) / downscale_factor)

    def get_uncertainty(self, points, un):
        inds, coeffs = find_grid_indices(points, self.lod, points.device)
        cfs_2 = (coeffs ** 2) / torch.sum((coeffs ** 2), dim=0, keepdim=True)
        uns = un[inds.long()].squeeze()
        un_points = torch.sqrt(torch.sum((uns * cfs_2), dim=0)).unsqueeze(1)
        return torch.log10(un_points + 1e-12)

    def get_uncertainty_per_cell(self):
        print("Calculating uncertainty per cell...")
        reg_lambda = 1e-4 / ((2 ** self.lod) ** 3)
        H = self.hessian / self.N + reg_lambda
        un = 1 / H
        primal_points = self.model.primal_points.clone().detach()
        un_points = self.get_uncertainty(primal_points, un).view(-1)
        self.un_points_cp = (un_points - un_points.min()) / (un_points.max() - un_points.min())
        primal_points = self.model.primal_points.clone().detach()
        if self.filter_out_cells:
            valid_indices = self.un_points_cp <= self.filter_thresh
            bins = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            hist = torch.histc(self.un_points_cp, bins=len(bins)-1, min=0.0, max=1.0)

            # Print frequency for each bin range
            print(f"Min: {self.un_points_cp.min()}, Max: {self.un_points_cp.max()}")

            for i in range(len(hist)):
                print(f"{bins[i]:.2f} – {bins[i+1]:.2f}: {int(hist[i].item())} values")
            print(f"Leaving {valid_indices.sum().item()} points out of {primal_points.shape[0]} based on uncertainty threshold {self.filter_thresh}")
            prune_mask = torch.zeros(primal_points.shape[0], dtype=torch.bool, device=self.device)
            prune_mask[valid_indices] = True
            #self.model.prune_points(prune_mask)
            self.un_points_cp = self.un_points_cp[valid_indices]
            self.model.prune_from_mask(prune_mask)
            print(f"Pruned points shape: {self.model.primal_points.shape}")
            #primal_points = primal_points[valid_indices]

    def get_outputs(self, ray_batch):
        depth_quantiles = torch.rand(*ray_batch.shape[:-1], 2, device=self.device).sort(dim=-1, descending=True).values
        rgba_output, depth, ray_samples, _, _ = self.model(
            ray_batch,
            #depth_quantiles=depth_quantiles,
            uncertainty=self.un_points_cp
        )
        return rgba_output

    def render_360_video(self, output_dir): 
        uncertainty_viewer_360 = Uncertainty360Viewer(self.device)
        uncertainty_viewer_360.render_frames(self.get_outputs, output_dir)

    def render_test_images(self, output_dir="output"):
        rays = self.data_handler.rays
        print(rays.shape)
        for i in range(rays.shape[0]):
            ray_batch = self.ray_batch_fetcher.next()[0]
            print(ray_batch.shape)
            #rgb_batch = self.rgb_batch_fetcher.next()[0]
            #alpha_batch = torch.ones_like(rgb_batch[..., :1])
            output = self.get_outputs(ray_batch)
            uncertainty = output[..., -1:].detach().cpu().numpy().squeeze(-1)
            if self.filter_out_in_image:
                mask = uncertainty > self.filter_thresh
            rgb_img = np.uint8(np.clip(output[..., :3].detach().cpu().numpy(), 0, 1) * 255)
            if self.filter_out_in_image:
                rgb_img[mask] = 255  # set high-uncertainty pixels to white
            if self.show_uncertainty:
                uncertainty_img = np.uint8(np.clip(uncertainty, 0, 1) * 255)
                Image.fromarray(uncertainty_img, mode="L").save(f"{output_dir}/bare_uncertainty/uncertainty{i}.png")
            else:
                if self.filter_out_cells:
                    Image.fromarray(rgb_img, mode="RGB").save(f"{output_dir}/rgb_cells_threshold/rgb{i}_{self.filter_thresh}.png")
                else: 
                    Image.fromarray(rgb_img, mode="RGB").save(f"{output_dir}/rgb_threshold/rgb{i}_{self.filter_thresh}.png")

    def render_uncertainty_images(self, output_dir="output"):
        print("Rendering uncertainty images...")
        self.get_uncertainty_per_cell()
        with torch.no_grad():
            if self.create_360_video:
                self.render_360_video(output_dir)
            else:
                self.render_test_images(output_dir)
                


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
    viewer = UncertaintyViewer(args, model_params.extract(args), dataset_params.extract(args),
        optimization_params.extract(args), pipeline_params.extract(args))
    viewer.render_uncertainty_images()


if __name__ == "__main__":
    main()