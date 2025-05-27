
from pathlib import Path
import time
import torch
from configs import *
from radfoam_model.scene import RadFoamScene
from nerfstudio.field_components.encodings import HashEncoding
from data_loader import DataHandler
from utils.utils import (find_grid_indices, normalize_point_coords)
from radfoam_model.render import TraceRays
import math
import numpy as np

class ComputeUncertainty:

    def __init__(self, args, pipeline_params, model_params, optimization_params, dataset_params):
        self.pipeline_params = pipeline_params
        self.model_params = model_params
        self.optimization_params = optimization_params
        self.dataset_params = dataset_params
        self.args = args
        self.device = torch.device(args.device)
        self.checkpoint_path = Path("output/bonsai@44a5858d")
        self.output_path = self.checkpoint_path / "unc.npy"
        self.lod = 8


    def find_uncertainty(self, points, deform_points_1, rgb):
        inds, coeffs = find_grid_indices(points, self.lod, self.device)
        # because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        colors = torch.sum(rgb, dim=0)
        TraceRays.RETAIN_GRAPH = True
        colors[0].backward(retain_graph=True)
        r = deform_points_1.grad.clone().detach().view(-1, 3)

        deform_points_1.grad.zero_()
        colors[1].backward(retain_graph=True)
        g = deform_points_1.grad.clone().detach().view(-1, 3)

        deform_points_1.grad.zero_()
        TraceRays.RETAIN_GRAPH = False
        colors[2].backward()
        b = deform_points_1.grad.clone().detach().view(-1, 3)

        deform_points_1.grad.zero_()

        dmy = torch.arange(inds.shape[1], device=self.device)
        first = True
        for corner in range(8):
            if first:
                all_ind = torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)
                all_r = coeffs[corner].unsqueeze(-1) * r
                all_g = coeffs[corner].unsqueeze(-1) * g
                all_b = coeffs[corner].unsqueeze(-1) * b
                first = False
            else:
                all_ind = torch.cat((all_ind, torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)),
                                    dim=0)
                all_r = torch.cat((all_r, coeffs[corner].unsqueeze(-1) * r), dim=0)
                all_g = torch.cat((all_g, coeffs[corner].unsqueeze(-1) * g), dim=0)
                all_b = torch.cat((all_b, coeffs[corner].unsqueeze(-1) * b), dim=0)

        keys_all, inds_all = torch.unique(all_ind, dim=0, return_inverse=True)
        grad_r_1 = torch.bincount(inds_all, weights=all_r[..., 0])  # for first element of deformation field
        grad_g_1 = torch.bincount(inds_all, weights=all_g[..., 0])
        grad_b_1 = torch.bincount(inds_all, weights=all_b[..., 0])
        grad_r_2 = torch.bincount(inds_all, weights=all_r[..., 1])  # for second element of deformation field
        grad_g_2 = torch.bincount(inds_all, weights=all_g[..., 1])
        grad_b_2 = torch.bincount(inds_all, weights=all_b[..., 1])
        grad_r_3 = torch.bincount(inds_all, weights=all_r[..., 2])  # for third element of deformation field
        grad_g_3 = torch.bincount(inds_all, weights=all_g[..., 2])
        grad_b_3 = torch.bincount(inds_all, weights=all_b[..., 2])
        grad_1 = grad_r_1 ** 2 + grad_g_1 ** 2 + grad_b_1 ** 2
        grad_2 = grad_r_2 ** 2 + grad_g_2 ** 2 + grad_b_2 ** 2
        grad_3 = grad_r_3 ** 2 + grad_g_3 ** 2 + grad_b_3 ** 2  # will consider the trace of each submatrix for each deformation
        # vector as indicator of hessian wrt the whole vector

        grads_all = torch.cat((keys_all[:, 1].unsqueeze(-1), (grad_1 + grad_2 + grad_3).unsqueeze(-1)), dim=-1)

        hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        hessian = hessian.put((grads_all[:, 0]).long(), grads_all[:, 1], True)

        return hessian


    def get_outputs(self, model, ray_batch, rgb_batch, alpha_batch):
        # get the offsets from the deform field
        primal_points = model.primal_points.clone().detach()
        normalized_points = normalize_point_coords(primal_points)
        offsets_1 = self.deform_field(normalized_points).clone().detach()
        offsets_1.requires_grad = True
        deformed_points = primal_points + offsets_1

        # Sample depths along the rays (optional for diversity)
        depth_quantiles = (
            torch.rand(*ray_batch.shape[:-1], 2, device=self.device)
            .sort(dim=-1, descending=True)
            .values
        )

        # Forward pass through the model to get RGBA
        rgba_output, depth, ray_samples, _, _ = model(
            ray_batch,
            depth_quantiles=depth_quantiles,
            primal_points = deformed_points
        )

        # Extract opacity and apply white background if needed
        opacity = rgba_output[..., -1:]
        #if self.pipeline_args.white_background:
        rgb_output = rgba_output[..., :3] + (1 - opacity)
        #else:
            #rgb_output = rgba_output[..., :3]

        outputs = {
            "rgb": rgb_output,
            #"accumulation": accumulation,
            "depth": depth,
        }
        return outputs, deformed_points, offsets_1


    def determine_hessian(self):

        #initialize model
        model = RadFoamScene(args=self.model_params, device=self.device)
        model.load_pt(f"{self.checkpoint_path}/model.pt")
        model.eval()

        #initialize hessian and deformation field
        self.hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        self.deform_field = HashEncoding(num_levels=1,
                                    min_res=2 ** self.lod,
                                    max_res=2 ** self.lod,
                                    log2_hashmap_size=self.lod * 3 + 1,
                                    # simple regular grid (hash table size > grid size)
                                    features_per_level=3,
                                    hash_init_scale=0.,
                                    implementation="torch",
                                    interpolation="Linear")
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2 ** self.lod]).to(self.device)


        print("Computing Hessian")
        start_time = time.time()


        # Load the dataset
        train_data_handler = DataHandler(
            self.dataset_params, rays_per_batch=250_000, device=self.device
        )
        iter2downsample = dict(
            zip(
                self.dataset_params.downsample_iterations,
                self.dataset_params.downsample
            )
        )
        downsample = iter2downsample[0]
        train_data_handler.reload(split="train", downsample=downsample)


        len_train = len(train_data_handler)
        data_iterator = train_data_handler.get_iter()
        print("Length of training data:", len_train)
        for i in range(len_train):
            #print("step", i)
            ray_batch, rgb_batch, alpha_batch = next(data_iterator)
            outputs, points, offsets_1 = self.get_outputs(model, ray_batch, rgb_batch, alpha_batch)
            hessian = self.find_uncertainty(points, offsets_1, outputs['rgb'].view(-1, 3))
            self.hessian += hessian.clone().detach()

        with open(str(self.output_path), 'wb') as f:
            np.save(f, self.hessian.cpu().numpy())


        print("Saving Hessian")
        print("Hessian", self.hessian)
        print("Hessian shape sqrt:", math.sqrt(self.hessian.numel()))
        print("Hessian unique", torch.unique(self.hessian))
        print("Very small (<1e-10):", (hessian < 1e-10).sum().item())
        print("Small (1e-10 to 1e-3):", ((hessian >= 1e-10) & (hessian < 1e-3)).sum().item())
        print("Medium (1e-3 to 1e3):", ((hessian >= 1e-3) & (hessian < 1e3)).sum().item())
        print("Large (>1e3):", (hessian >= 1e3).sum().item())

        end_time = time.time()
        print("Done")
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")

def main():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)
    args = parser.parse_args()

    compute_uncertainty = ComputeUncertainty(args, pipeline_params.extract(args),
                                             model_params.extract(args), optimization_params.extract(args),
                                             dataset_params.extract(args))
    compute_uncertainty.determine_hessian()


if __name__ == "__main__":
    main()