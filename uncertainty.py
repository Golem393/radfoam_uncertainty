
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
import copy
import gc

"""def hook_print_grad_norm(grad, name=""):
    grad_norm = grad.norm().item()
    print(f"Gradient norm at {name}: {grad_norm:.6f}")
    return grad  # Must return the gradient"""

def log_uncertainty_metrics(
    point_uncertainty: torch.Tensor,
    label: str = "",
    log_file: str = None
    ):
        
        def safe_mean(tensor):
            return tensor.mean().item() if tensor.numel() > 0 else 0.0

        def safe_entropy(tensor):
            safe_tensor = tensor[tensor > 0]
            return - (safe_tensor * safe_tensor.log()).mean().item() if safe_tensor.numel() > 0 else 0.0

        masked_uncert = point_uncertainty
        num_active_points = masked_uncert.numel()
        mean = safe_mean(masked_uncert)
        std = masked_uncert.std().item() if masked_uncert.numel() > 1 else 0.0
        entropy = safe_entropy(masked_uncert)
        quantiles = torch.quantile(masked_uncert, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=masked_uncert.device)).tolist()

        low_cert_mask = point_uncertainty < 0.2
        high_cert_mask = point_uncertainty > 0.8
        low_avg_all = safe_mean(point_uncertainty[low_cert_mask])
        high_avg_all = safe_mean(point_uncertainty[high_cert_mask])

        uncertainty_bins = {}
        bin_edges = [i * 0.1 for i in range(11)]

        for i in range(len(bin_edges) - 1):
            lower_bound = bin_edges[i]
            upper_bound = bin_edges[i+1]
            
            if i == len(bin_edges) - 2:
                bin_mask = (masked_uncert >= lower_bound) & (masked_uncert <= upper_bound)
            else:
                bin_mask = (masked_uncert >= lower_bound) & (masked_uncert < upper_bound)
            
            count_in_bin = bin_mask.sum().item()
            percentage = (count_in_bin / num_active_points) * 100
            uncertainty_bins[f'{lower_bound:.1f}-{upper_bound:.1f}'] = percentage

        bin_log_str = "\n".join([f"            {k}: {v:.2f}%" for k, v in uncertainty_bins.items()])


        log = f"""
            --- Uncertainty Metrics [{label}] ---
            Active Points: {num_active_points}
            Mean: {mean:.4f}, Std: {std:.4f}, Entropy: {entropy:.4f}
            Quantiles: Q10: {quantiles[0]:.4f}, Q50: {quantiles[1]:.4f}, Q90: {quantiles[2]:.4f}
            Low-certainty region (<0.2): {low_avg_all:.4f}
            High-certainty region (>0.8): {high_avg_all:.4f}
            --- Uncertainty Bin Distribution ---
            {bin_log_str}
            ----------------------------------------
            """
        print(log)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(log)

class ComputeUncertainty:

    def __init__(self, args, pipeline_params, model_params, optimization_params, dataset_params, 
                 save_hessian = False):
        self.pipeline_params = pipeline_params
        self.model_params = model_params
        self.optimization_params = optimization_params
        self.dataset_params = dataset_params
        self.args = args
        self.device = torch.device(args.device)
        self.checkpoint_path = Path("output_final/normal_setting")
        self.output_path = self.checkpoint_path / "unc.npy"
        self.lod = 8
        width, height = 3118, 2078
        downscale_factor = 2.0
        self.N = 1000 * ((width * height) / downscale_factor)
        self.save_hessian = save_hessian

        self.train_data_handler = DataHandler(
            self.dataset_params, rays_per_batch=250_000, device=self.device
        )
        iter2downsample = dict(
            zip(
                self.dataset_params.downsample_iterations,
                self.dataset_params.downsample
            )
        )
        downsample = iter2downsample[0]
        self.train_data_handler.reload(split="train", downsample=downsample)

        self.deform_field = HashEncoding(num_levels=1,
                                    min_res=2 ** self.lod,
                                    max_res=2 ** self.lod,
                                    log2_hashmap_size=self.lod * 3 + 1,
                                    features_per_level=3,
                                    hash_init_scale=0.,
                                    implementation="torch",
                                    interpolation="Linear")
        
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2 ** self.lod]).to(self.device)
        self.hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)

    def get_ray_uncertainty(self, ray_batch, model):
        #self.hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        outputs, points, offsets_1 = self.get_outputs(model, ray_batch)
        self.hessian = self.find_uncertainty(points, offsets_1, outputs['rgb'].view(-1, 3))
        self.hessian = self.hessian.clamp(-1e4, 1e4)
        unc_per_cell = self.get_uncertainty_per_cell(model)
        #unc_per_pixel = self.get_unc_outputs_pixel(model, ray_batch, unc_per_cell)
        #unc_per_pixel.register_hook(lambda g: hook_print_grad_norm(g, "unc_per_pixel"))
        torch.cuda.empty_cache()
        gc.collect()
        return unc_per_cell

    def get_prune_uncertainty(self, model):
        self.hessian.zero_() 
        len_train = len(self.train_data_handler)
        data_iterator = self.train_data_handler.get_iter()
        print("Length of training data:", len_train)
        for i in range(len_train):
            ray_batch, rgb_batch, alpha_batch = next(data_iterator)
            data_iterator = self.train_data_handler.get_iter()
            outputs, points, offsets_1 = self.get_outputs(model, ray_batch)
            self.hessian += self.find_uncertainty(points, offsets_1, outputs['rgb'].view(-1, 3)).clamp(-1e4, 1e4).clone().detach()
        unc_per_cell = self.get_uncertainty_per_cell(model)
        #unc_per_pixel = self.get_unc_outputs_pixel(model, ray_batch, unc_per_cell)
        #unc_per_pixel.register_hook(lambda g: hook_print_grad_norm(g, "unc_per_pixel"))
        torch.cuda.empty_cache()
        gc.collect()
        return unc_per_cell

    def get_uncertainty(self, points, un):
        lod2 = np.log2(round(self.hessian.shape[0] ** (1 / 3)) - 1)
        inds, coeffs = find_grid_indices(points, lod2, points.device)
        cfs_2 = (coeffs ** 2) / torch.sum((coeffs ** 2), dim=0, keepdim=True)
        uns = un[inds.long()].squeeze()
        un_points = torch.sqrt(torch.sum((uns * cfs_2), dim=0)).unsqueeze(1)
        return torch.log10(un_points + 1e-12)

    def get_unc_outputs_pixel(self, model, ray_batch, unc_per_cell):
        rgba_output, _, _, _, _ = model(
            ray_batch,
            uncertainty=unc_per_cell
        )
        uncertainty = rgba_output[..., -1:].squeeze(-1)
        return uncertainty

    def get_uncertainty_per_cell(self, model):
        #print("Calculating uncertainty per cell...")
        reg_lambda = 1e-4 / ((2 ** self.lod) ** 3)
        H = self.hessian / self.N + reg_lambda
        un = 1 / H
        primal_points = model.primal_points
        un_points = self.get_uncertainty(primal_points, un).view(-1)
        self.un_points_cp = (un_points - un_points.min()) / (un_points.max() - un_points.min()+ 1e-12)
        return self.un_points_cp

    """def find_uncertainty(self, points, deform_points_1, rgb):
        #rgb.register_hook(lambda g: hook_print_grad_norm(g, "rgb in find_uncertainty"))
        inds, coeffs = find_grid_indices(points, self.lod, self.device)
        inds = [corner_ind.to(torch.int64) for corner_ind in inds]
        colors = torch.sum(rgb, dim=0)
        gradients = []
        for i in range(3):
            TraceRays.RETAIN_GRAPH = True #if i < 2 else False
            grad_output = torch.zeros_like(colors)
            grad_output[i] = 1.0
            grad = torch.autograd.grad(colors, deform_points_1, grad_outputs=grad_output, 
                                    retain_graph=True, create_graph=True)[0]#.clone().detach()
            gradients.append(grad.view(-1, 3))
        
        r, g, b = gradients
       
        all_indices = []
        all_values_r = []
        all_values_g = []
        all_values_b = []
        
        for corner in range(8):
            indices = inds[corner]
            weights = coeffs[corner].unsqueeze(-1)
            
            all_indices.append(indices)
            all_values_r.append(weights * r)
            all_values_g.append(weights * g)
            all_values_b.append(weights * b)
        
        all_indices = torch.cat(all_indices)
        all_values_r = torch.cat(all_values_r)
        all_values_g = torch.cat(all_values_g)
        all_values_b = torch.cat(all_values_b)

        num_bins = ((2 ** self.lod) + 1) ** 3
        grad_r = torch.zeros(num_bins, 3, device=self.device)
        grad_g = torch.zeros(num_bins, 3, device=self.device)
        gc.collect()
        torch.cuda.empty_cache()
        grad_b = torch.zeros(num_bins, 3, device=self.device)

        expanded_indices = all_indices.unsqueeze(1).expand(-1, 3)
        
        grad_r.scatter_add_(0, expanded_indices, all_values_r)
        grad_g.scatter_add_(0, expanded_indices, all_values_g)
        grad_b.scatter_add_(0, expanded_indices, all_values_b)
        
        grad_1 = (grad_r[..., 0] ** 2 + grad_g[..., 0] ** 2 + grad_b[..., 0] ** 2)
        grad_2 = (grad_r[..., 1] ** 2 + grad_g[..., 1] ** 2 + grad_b[..., 1] ** 2)
        grad_3 = (grad_r[..., 2] ** 2 + grad_g[..., 2] ** 2 + grad_b[..., 2] ** 2)
        
        hessian = grad_1 + grad_2 + grad_3
        #hessian.register_hook(lambda g: hook_print_grad_norm(g, "hessian"))
        return hessian"""

    def find_uncertainty(self, points, deform_points_1, rgb):
        inds, coeffs = find_grid_indices(points, self.lod, self.device)
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
        grad_r_1 = torch.bincount(inds_all, weights=all_r[..., 0])
        grad_g_1 = torch.bincount(inds_all, weights=all_g[..., 0])
        grad_b_1 = torch.bincount(inds_all, weights=all_b[..., 0])
        grad_r_2 = torch.bincount(inds_all, weights=all_r[..., 1])
        grad_g_2 = torch.bincount(inds_all, weights=all_g[..., 1])
        grad_b_2 = torch.bincount(inds_all, weights=all_b[..., 1])
        grad_r_3 = torch.bincount(inds_all, weights=all_r[..., 2])
        grad_g_3 = torch.bincount(inds_all, weights=all_g[..., 2])
        grad_b_3 = torch.bincount(inds_all, weights=all_b[..., 2])
        grad_1 = grad_r_1 ** 2 + grad_g_1 ** 2 + grad_b_1 ** 2
        grad_2 = grad_r_2 ** 2 + grad_g_2 ** 2 + grad_b_2 ** 2
        grad_3 = grad_r_3 ** 2 + grad_g_3 ** 2 + grad_b_3 ** 2

        grads_all = torch.cat((keys_all[:, 1].unsqueeze(-1), (grad_1 + grad_2 + grad_3).unsqueeze(-1)), dim=-1)

        hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        hessian = hessian.put((grads_all[:, 0]).long(), grads_all[:, 1], True)

        return hessian


    def get_outputs(self, model, ray_batch):
        primal_points = model.primal_points.clone().detach()
        normalized_points = normalize_point_coords(primal_points)
        offsets_1 = self.deform_field(normalized_points).clone().detach()
        offsets_1.requires_grad = True
        deformed_points = primal_points + offsets_1

        depth_quantiles = (
                    torch.rand(*ray_batch.shape[:-1], 2, device=self.device)
                    .sort(dim=-1, descending=True)
                    .values
                )
        
        rgba_output, depth, ray_samples, _, _ = model(
            ray_batch,
            depth_quantiles=depth_quantiles,
            primal_points=deformed_points
        )

        opacity = rgba_output[..., -1:]
        rgb_output = rgba_output[..., :3] + (1 - opacity)

        outputs = {
            "rgb": rgb_output,
            "depth": depth,
        }
        return outputs, deformed_points, offsets_1


    def determine_hessian(self, model = None):
        torch.cuda.empty_cache()
        gc.collect()
        if not model:
            model = RadFoamScene(args=self.model_params, device=self.device)
            model.load_pt(f"{self.checkpoint_path}/model.pt")
            model.declare_optimizer(
                args=self.optimization_params,
                warmup=self.pipeline_params.densify_from,
                max_iterations=self.pipeline_params.iterations,
            )
        model.eval()

        self.hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        self.deform_field = HashEncoding(num_levels=1,
                                    min_res=2 ** self.lod,
                                    max_res=2 ** self.lod,
                                    log2_hashmap_size=self.lod * 3 + 1,
                                    features_per_level=3,
                                    hash_init_scale=0.,
                                    implementation="torch",
                                    interpolation="Linear")
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2 ** self.lod]).to(self.device)


        print("Computing Hessian")
        start_time = time.time()

        sample_fraction = 1.0
        len_train = int(len(self.train_data_handler) * sample_fraction)
        data_iterator = self.train_data_handler.get_iter()
        print("Length of training data:", len_train)
        for i in range(len_train):
            ray_batch, rgb_batch, alpha_batch = next(data_iterator)
            outputs, points, offsets_1 = self.get_outputs(model, ray_batch)
            hessian = self.find_uncertainty(points, offsets_1, outputs['rgb'].view(-1, 3))         
            self.hessian += hessian.clone().detach()


        if self.save_hessian:
            print("Saving Hessian")
            with open(str(self.output_path), 'wb') as f:
                np.save(f, self.hessian.cpu().numpy())

        model.train()
        print("Hessian calc done")


        
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
    gc.collect()
    torch.cuda.empty_cache()
    parser = configargparse.ArgParser(
        default_config_files=["arguments/configs/LF.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)
    args = parser.parse_args()

    compute_uncertainty = ComputeUncertainty(args, pipeline_params.extract(args),
                                             model_params.extract(args), optimization_params.extract(args),
                                             dataset_params.extract(args),
                                             save_hessian = True)
    
    compute_uncertainty.determine_hessian()


if __name__ == "__main__":
    main()