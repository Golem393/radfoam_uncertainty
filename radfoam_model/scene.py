import os
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
import tqdm

import radfoam
from radfoam_model.render import TraceRays
from radfoam_model.utils import *
import gc


class RadFoamScene(torch.nn.Module):

    def __init__(
        self,
        args,
        points=None,
        points_colors=None,
        cameras=None,
        device=torch.device("cuda"),
        attr_dtype=torch.float32,
    ):
        super().__init__()

        self.device = device
        self.attr_dtype = attr_dtype
        if cameras is not None:
            self.cameras = cameras.to(device)
        else:
            self.cameras = None
        self.sh_degree = args.sh_degree
        self.num_init_points = args.init_points
        self.num_final_points = args.final_points
        self.activation_scale = args.activation_scale

        if points is not None:
            self.initialize_from_pcd(points, points_colors)
        else:
            self.random_initialize()

        self.att_dc = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3,
                device=self.device,
                dtype=self.attr_dtype,
            )
        )
        self.att_sh = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3 * ((1 + self.sh_degree) * (1 + self.sh_degree) - 1),
                device=device,
                dtype=self.attr_dtype,
            )
        )

        self.pipeline = radfoam.create_pipeline(self.sh_degree, self.attr_dtype)

    def random_initialize(self):
        primal_points = (
            torch.randn(self.num_init_points, 3, device=self.device) * 25
        )
        self.triangulation = radfoam.Triangulation(primal_points)
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)

        self.att_dc = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3,
                device=self.device,
                dtype=self.attr_dtype,
            )
        )

        density = torch.zeros(
            self.num_init_points, 1, device=self.device, dtype=self.attr_dtype
        )
        self.density = nn.Parameter(density[perm])

    def initialize_from_pcd(self, points, points_colors):
        points = points.to(self.device)
        points_colors = points_colors.to(self.device)

        num_random = 5_000
        random = torch.randn([num_random, 3], device=self.device) * 10

        num_samples = int(0.9 * points.shape[0])
        print(
            f"Starting with {num_samples} points from {points.shape[0]} COLMAP points"
        )
        points_idx = torch.randint(0, points.shape[0], (num_samples,))
        samp_points = points[points_idx]
        samp_points += torch.randn_like(samp_points) * 1e-2
        samp_colors = points_colors[points_idx]

        primal_points = torch.cat([samp_points, random], dim=0)
        primal_density = torch.cat(
            [
                torch.rand(samp_colors.shape[0], 1, dtype=self.attr_dtype),
                -0.5 * torch.ones(num_random, 1, dtype=self.attr_dtype),
            ],
            dim=0,
        ).to(self.device)

        torch.cuda.empty_cache()

        self.triangulation = radfoam.Triangulation(primal_points)
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)

        self.density = nn.Parameter(primal_density)
        self.num_init_points = self.primal_points.shape[0]

    def permute_points(self, permutation):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "env" not in group["name"]:
                stored_state = self.optimizer.state.get(
                    group["params"][0], None
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][
                        permutation
                    ]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][
                        permutation
                    ]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][permutation].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][permutation].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        self.primal_points = optimizable_tensors["primal_points"]
        self.density = optimizable_tensors["density"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]

    def update_triangulation(self, rebuild=True, incremental=False, points = None):
        torch.cuda.empty_cache()
        gc.collect()
        if points == None:
            points = self.primal_points
        if not points.isfinite().all():
            raise RuntimeError("NaN in points")

        needs_permute = False
        perturbation = 1e-6
        del_points = points
        failures = 0
        while rebuild:
            if failures > 25:
                raise RuntimeError("aborted triangulation after 25 attempts")
            try:
                needs_permute = self.triangulation.rebuild(
                    del_points, incremental=incremental
                )
                break
            except radfoam.TriangulationFailedError as e:
                print("caught: ", e)
                perturbation *= 2
                failures += 1
                incremental = False
                with torch.no_grad():
                    del_points = (
                        points
                        + perturbation * torch.randn_like(points)
                    )

        if failures > 5:
            with torch.no_grad():
                self.primal_points.copy_(del_points)

        if needs_permute:
            perm = self.triangulation.permutation().to(torch.long)
            self.permute_points(perm)

        self.aabb_tree = radfoam.build_aabb_tree(points)

        self.point_adjacency = self.triangulation.point_adjacency()
        self.point_adjacency_offsets = (
            self.triangulation.point_adjacency_offsets()
        )

    def get_primal_density(self):
        return self.activation_scale * F.softplus(self.density, beta=10)

    def get_primal_attributes(self):
        return torch.cat([self.att_dc, self.att_sh], dim=-1)

    def get_trace_data(self, points = None):
        if points == None:
            points = self.primal_points
        attributes = torch.cat(
            [self.get_primal_attributes(), self.get_primal_density()],
            dim=-1,
        ).to(self.attr_dtype)
        point_adjacency = self.point_adjacency
        point_adjacency_offsets = self.point_adjacency_offsets

        return points, attributes, point_adjacency, point_adjacency_offsets

    def show(self, loop_fn=lambda v: None, iterations=None, **viewer_kwargs):
        radfoam.run_with_viewer(
            self.pipeline, loop_fn, total_iterations=iterations, **viewer_kwargs
        )

    def get_starting_point(self, rays, points, aabb_tree):
        with torch.no_grad():
            camera_origins = rays[..., :3]
            unique_cameras, inverse_indices = torch.unique(
                camera_origins, dim=0, return_inverse=True
            )

            nn_inds = radfoam.nn(points, aabb_tree, unique_cameras).long()

            start_point = nn_inds[inverse_indices]
            return start_point.type(torch.uint32)

    def forward(
        self,
        rays,
        start_point=None,
        depth_quantiles=None,
        return_contribution=False,
        primal_points=None,
        uncertainty = None,
        update_triangulation = False
    ):
        points = None
        if primal_points is not None:
            points = primal_points
            if update_triangulation:
                self.update_triangulation(rebuild=True, points = points)

        points, attributes, point_adjacency, point_adjacency_offsets = (
            self.get_trace_data(points)
        )
        
        if start_point is None:
            start_point = self.get_starting_point(rays, points, self.aabb_tree)
        else:
            start_point = torch.broadcast_to(start_point, rays.shape[:-1])
        return TraceRays.apply(
            self.pipeline,
            points,
            attributes,
            point_adjacency,
            point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles,
            return_contribution,
            uncertainty
        )

    def update_viewer(self, viewer):
        points, attributes, point_adjacency, point_adjacency_offsets = (
            self.get_trace_data()
        )
        viewer.update_scene(
            points,
            attributes,
            point_adjacency,
            point_adjacency_offsets,
            self.aabb_tree,
        )

    def declare_optimizer(self, args, warmup, max_iterations):
        params = [
            {
                "params": self.primal_points,
                "lr": args.points_lr_init,
                "name": "primal_points",
            },
            {
                "params": self.density,
                "lr": args.density_lr_init,
                "name": "density",
            },
            {
                "params": self.att_dc,
                "lr": args.attributes_lr_init,
                "name": "att_dc",
            },
            {
                "params": self.att_sh,
                "lr": args.attributes_lr_init,
                "name": "att_sh",
            },
        ]

        self.optimizer = torch.optim.Adam(params, eps=1e-15)
        self.xyz_scheduler_args = get_cosine_lr_func(
            lr_init=args.points_lr_init,
            lr_final=args.points_lr_final,
            max_steps=args.freeze_points,
        )
        self.den_scheduler_args = get_cosine_lr_func(
            lr_init=args.density_lr_init,
            lr_final=args.density_lr_final,
            warmup_steps=warmup,
            max_steps=max_iterations,
        )
        self.attr_dc_scheduler_args = get_cosine_lr_func(
            lr_init=args.attributes_lr_init,
            lr_final=args.attributes_lr_final,
            max_steps=max_iterations,
        )
        self.attr_rest_scheduler_args = get_cosine_lr_func(
            lr_init=args.sh_factor * args.attributes_lr_init,
            lr_final=args.sh_factor * args.attributes_lr_final,
            warmup_steps=max_iterations // 5,
            max_steps=max_iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "primal_points":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "density":
                lr = self.den_scheduler_args(iteration)
                param_group["lr"] = lr
                param_group["lr"] = lr
            elif param_group["name"] == "att_dc":
                lr = self.attr_dc_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "att_sh":
                lr = self.attr_rest_scheduler_args(iteration)
                param_group["lr"] = lr

    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, prune_mask):
        valid_points_mask = ~prune_mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        self.primal_points = optimizable_tensors["primal_points"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]
        self.density = optimizable_tensors["density"]

    def cat_tensors_to_optimizer(self, new_params):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in new_params.keys():
                assert len(group["params"]) == 1
                stored_tensor = group["params"][0]
                extension_tensor = new_params[group["name"]]
                stored_state = self.optimizer.state.get(
                    group["params"][0], None
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (
                            stored_state["exp_avg"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_params):
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        self.primal_points = optimizable_tensors["primal_points"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]
        self.density = optimizable_tensors["density"]

    def prune_from_mask(self, keep_mask, upsample_factor=1.2):
        with torch.no_grad():
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            points, _, point_adjacency, point_adjacency_offsets = (
                self.get_trace_data()
            )
            ################### Farthest neighbor ###################
            farthest_neighbor, cell_radius = radfoam.farthest_neighbor(
                points,
                point_adjacency,
                point_adjacency_offsets,
            )
            farthest_neighbor = farthest_neighbor.long()

            ######################## Pruning ########################
            # Directly use the provided keep_mask to determine which points to prune
            prune_mask = ~keep_mask
            
            # Additional pruning based on cell size (kept from original)
            cell_size_mask = cell_radius < 1e-1
            prune_mask = prune_mask & cell_size_mask

            ######################## Random sampling ########################
            perturbation = 0.25 * (points[farthest_neighbor] - points)
            delta = torch.randn_like(perturbation)
            delta /= delta.norm(dim=-1, keepdim=True)
            perturbation += (
                0.1 * perturbation.norm(dim=-1, keepdim=True) * delta
            )

            # Sample new points from all current points (not just error-based)
            # Using uniform probability since we don't have error information
            num_sample_points = num_new_points
            sampled_inds = torch.multinomial(
                torch.ones(num_curr_points, device=points.device),
                num_sample_points,
                replacement=False,
            )
            sampled_points = (points + perturbation)[sampled_inds]

            new_params = {
                "primal_points": sampled_points,
                "att_dc": self.att_dc[sampled_inds],
                "att_sh": self.att_sh[sampled_inds],
                "density": self.density[sampled_inds],
            }

            # Don't prune the newly added points
            prune_mask = torch.cat(
                (
                    prune_mask,
                    torch.zeros(
                        sampled_points.shape[0],
                        device=prune_mask.device,
                        dtype=bool,
                    ),
                )
            )

            self.densification_postfix(new_params)
            self.prune_points(prune_mask)
            self.update_triangulation(rebuild=True)
            gc.collect()


        """
        assert keep_mask.dim() == 1 and keep_mask.dtype == torch.bool, "keep_mask must be a 1D boolean tensor"

        # Apply mask to all relevant attributes
        model.primal_points = nn.Parameter(model.primal_points[keep_mask])
        model.att_dc = nn.Parameter(model.att_dc[keep_mask])
        model.att_sh = nn.Parameter(model.att_sh[keep_mask])
        model.density = nn.Parameter(model.density[keep_mask])
        #model.point_adjacency = model.point_adjacency[keep_mask]
        #model.point_adjacency_offsets = model.point_adjacency_offsets[keep_mask]
        model.update_triangulation(rebuild=True)"""


    def prune_and_densify(
        self, point_error, point_contribution, point_uncertainty, upsample_factor=1.2, uncertainty_influence_factor = 0.3,
        iter = 0
    ):
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():

            print(f"Pruning and densifying at iteration {iter} with upsample factor {upsample_factor}")
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            primal_error_accum = point_error.clip(min=0).squeeze()
            points, _, point_adjacency, point_adjacency_offsets = (
                self.get_trace_data()
            )
            ################### Farthest neighbor ###################
            farthest_neighbor, cell_radius = radfoam.farthest_neighbor(
                points,
                point_adjacency,
                point_adjacency_offsets,
            )
            farthest_neighbor = farthest_neighbor.long()

            """if point_contribution.dim() == 2 and point_contribution.shape[1] == 1:
                point_contribution = point_contribution.squeeze(1) # Squeeze specifically dimension 1"""

            """contrib_based_mask = point_contribution > 1e-2
            target_num_points = contrib_based_mask.sum().item()
            uncert_sorted, _ = torcoutput/pruneinterp03fact3000contribnormh.sort(point_uncertainty.view(-1)) 
            uncertainty_threshold = uncert_sorted[target_num_points - 1].item()

            ######################## Pruning ########################
            """contrib_norm = (point_contribution / point_contribution.max()).clamp(0, 1).squeeze()
            uncert_norm = (point_uncertainty / point_uncertainty.max()).clamp(0, 1).squeeze()
            cert_norm = 1 - uncert_norm
            contrib_norm_fixed = contrib_norm * 1000 * 3
            prune_score = (1 - uncertainty_influence_factor) * contrib_norm_fixed + uncertainty_influence_factor * cert_norm 
            self_mask = prune_score > 0.3"""
            #self_mask = (point_contribution > 1e-2) & (cert_norm > 0.2)

            #contrib_threshold = 0#5e-2 #* point_contribution.max()
            #uncert_threshold = 0.3
            gc.collect()
            torch.cuda.empty_cache()
            #print(f"Shape of point_contribution (after squeeze): {point_contribution.shape}")
            #print(f"Shape of point_uncertainty (after squeeze): {point_uncertainty.shape}")

            """chunk_size = 10000
            self_mask_chunks = []
            for i in range(0, num_curr_points, chunk_size):
                #chunk_contrib = point_contribution[i:i+chunk_size]
                chunk_uncert = point_uncertainty[i:i + chunk_size]
                chunk_mask = chunk_uncert < uncert_threshold
                #chunk_uncert = uncert_norm [i:i+chunk_size]

                # These shapes should now both be (X,)
                # print(f"Chunk contrib shape: {chunk_contrib.shape}, Chunk uncert shape: {chunk_uncert.shape}")

                # Now, (X,) & (X,) will result in (X,), which is what you want for a 1D mask
                #chunk_mask = (chunk_contrib > contrib_threshold) & (chunk_uncert < uncert_threshold)
                self_mask_chunks.append(chunk_mask)"""
            """

            # These shapes should now be (10000,) or (951,)
            #for k, chunk in enumerate(self_mask_chunks):
                #print(f"Final chunk {k} shape (before cat): {chunk.shape}")

            self_mask = torch.cat(self_mask_chunks, dim=0)

            #self_mask = (point_contribution > contrib_threshold) & (point_uncertainty < uncert_threshold)
            #print("Contr", point_contribution)
            #print("Contr norm", contrib_norm)
            #print("Uncert norm", uncert_norm)
            #print("prune score", prune_score)"""
            #self_mask = torch.cat(self_mask_chunks, dim=0)
            """with open("uncertainty_thresholds.log", "a") as f:
                f.write(f"Iter {iter}: {uncertainty_threshold:.6f}\n")
            """
            """def print_distribution(tensor, name, bins=[0, 1e-5, 1e-3, 1e-2, 1e-1, 0.25, 0.3, 0.5, 0.75, 0.9, 1.0, float('inf')], filename="distribution_log.txt"):
                with open(filename, 'a') as f:  # 'a' mode for append
                    f.write(f"\n{name} distribution:\n")
                    hist = torch.histc(tensor, bins=len(bins)-1, min=bins[0], max=bins[-2])
                    for i in range(len(bins) - 1):
                        low = bins[i]
                        high = bins[i+1]
                        # Get mask and count
                        count = ((tensor >= low) & (tensor < high)).sum().item()
                        f.write(f"[{low:.1e}, {high:.1e}): {count} values\n")

            # Example usage:
            with open("distribution_log.txt", 'a') as f:
                f.write(f"\niter {iter}\n")
            print_distribution(point_contribution.squeeze(), "Contribution")
            print_distribution(contrib_norm, "Contrib norm")
            print_distribution(uncert_norm, "Uncertainty norm")
            print_distribution(prune_score, "Prune score")"""

            
            #mask = point_contribution > 1e-2
            #num_pruned_uncertainty = (~self_mask).sum()
            #num_pruned_contrib_only = (~mask).sum()
            #print("Uncertainty-aware pruned:", num_pruned_uncertainty.item())
           # print("Contribution-only pruned:", num_pruned_contrib_only.item())

            self_mask = point_contribution > 1e-2
            neighbor_mask = self_mask.long()[point_adjacency.long()]
            neighbor_mask = torch.cat(
                [neighbor_mask, torch.zeros_like(neighbor_mask[:1])], dim=0
            )
            nsum = torch.cumsum(neighbor_mask, dim=0)

            offsets = point_adjacency_offsets.long()
            n_masked_adj = nsum[offsets[1:]] - nsum[offsets[:-1]]

            contrib_mask = ((n_masked_adj == 0) & ~self_mask).squeeze()
            cell_size_mask = cell_radius < 1e-1
            prune_mask = contrib_mask * cell_size_mask

            ######################## Random sampling ########################
            primal_contribution_accum = point_contribution.squeeze()
            mask = primal_contribution_accum < 1e-3
            self.density[mask] = -1

            perturbation = 0.25 * (points[farthest_neighbor] - points)
            delta = torch.randn_like(perturbation)
            delta /= delta.norm(dim=-1, keepdim=True)
            perturbation += (
                0.1 * perturbation.norm(dim=-1, keepdim=True) * delta
            )

            num_sample_points = num_new_points
            sampled_inds = torch.multinomial(
                primal_error_accum * cell_radius,
                num_sample_points,
                replacement=False,
            )
            sampled_points = (points + perturbation)[sampled_inds]

            new_params = {
                "primal_points": sampled_points,
                "att_dc": self.att_dc[sampled_inds],
                "att_sh": self.att_sh[sampled_inds],
                "density": self.density[sampled_inds],
            }

            prune_mask = torch.cat(
                (
                    prune_mask,
                    torch.zeros(
                        sampled_points.shape[0],
                        device=prune_mask.device,
                        dtype=bool,
                    ),
                )
            )

            self.densification_postfix(new_params)
            self.prune_points(prune_mask)
            torch.cuda.empty_cache()
            gc.collect()

    def collect_error_map(self, data_handler, white_bkg=True, downsample=2):
        rays, rgbs = data_handler.rays, data_handler.rgbs

        points, _, _, _ = self.get_trace_data()
        start_points = self.get_starting_point(
            rays[:, 0, 0].cuda(), points, self.aabb_tree
        )

        ray_batch_fetcher = radfoam.BatchFetcher(
            rays, batch_size=1, shuffle=False
        )
        rgb_batch_fetcher = radfoam.BatchFetcher(
            rgbs, batch_size=1, shuffle=False
        )

        point_error_accum = torch.zeros_like(self.primal_points[..., 0:1])
        point_contribution_accum = torch.zeros_like(
            self.primal_points[..., 0:1]
        )
        rgb_loss = nn.L1Loss(reduction="none")

        for i in range(rays.shape[0]):
            ray_batch = ray_batch_fetcher.next()
            rgb_batch = rgb_batch_fetcher.next()

            d = torch.randint(0, downsample, (2,))
            ray_batch = ray_batch[:, d[0] :: downsample, d[1] :: downsample, :]
            rgb_batch = rgb_batch[:, d[0] :: downsample, d[1] :: downsample, :]

            rgba_output, _, contribution, _, errbox = self.forward(
                ray_batch, start_points[i], return_contribution=True
            )
            opacity = rgba_output[..., -1:]
            if white_bkg:
                rgb_output = rgba_output[..., :3] + (1 - opacity)
            else:
                rgb_output = rgba_output[..., :3]

            color_loss = rgb_loss(rgb_batch, rgb_output).mean(dim=-1)

            color_loss.sum().backward()
            point_error_accum += self.primal_points.grad.norm(
                dim=-1, keepdim=True
            ).detach()
            point_contribution_accum = torch.maximum(
                point_contribution_accum, contribution.detach()
            )
            torch.cuda.synchronize()

            self.optimizer.zero_grad(set_to_none=True)

        return point_error_accum, point_contribution_accum

    def save_ply(self, ply_path):
        points = self.primal_points.detach().float().cpu().numpy()
        density = self.get_primal_density().detach().float().cpu().numpy()
        color_attributes = (
            self.get_primal_attributes().detach().float().cpu().numpy()
        )
        adjacency = self.point_adjacency.cpu().numpy()
        adjacency_offsets = self.point_adjacency_offsets.cpu().numpy()

        C0 = 0.28209479177387814
        r = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 0]), 0, 255),
            dtype=np.uint8,
        )
        g = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 1]), 0, 255),
            dtype=np.uint8,
        )
        b = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 2]), 0, 255),
            dtype=np.uint8,
        )

        vertex_data = []
        for i in tqdm.trange(points.shape[0]):
            vertex_data.append(
                (
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    r[i],
                    g[i],
                    b[i],
                    density[i, 0],
                    adjacency_offsets[i + 1],
                    *[
                        color_attributes[i, 3 + j]
                        for j in range(color_attributes.shape[1] - 3)
                    ],
                )
            )

        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
            ("density", np.float32),
            ("adjacency_offset", np.uint32),
        ]

        for i in range(self.att_sh.shape[1]):
            dtype.append(("color_sh_{}".format(i), np.float32))

        vertex_data = np.array(vertex_data, dtype=dtype)
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        adjacency_data = np.array(adjacency, dtype=[("adjacency", np.uint32)])
        adjacency_element = PlyElement.describe(adjacency_data, "adjacency")

        PlyData([vertex_element, adjacency_element]).write(ply_path)

    def save_pt(self, pt_path):
        points = self.primal_points.detach().float().cpu()
        density = self.density.detach().float().cpu()
        color_dc = self.att_dc.detach().float().cpu()
        color_sh = self.att_sh.detach().float().cpu()
        adjacency = self.point_adjacency.cpu()
        adjacency_offsets = self.point_adjacency_offsets.cpu()

        scene_data = {
            "xyz": points,
            "density": density,
            "color_dc": color_dc,
            "color_sh": color_sh,
            "adjacency": adjacency.long(),
            "adjacency_offsets": adjacency_offsets.long(),
        }
        torch.save(scene_data, pt_path)

    def load_pt(self, pt_path):
        scene_data = torch.load(pt_path)

        self.primal_points = nn.Parameter(scene_data["xyz"].to(self.device))
        self.density = nn.Parameter(scene_data["density"].to(self.device))
        self.att_dc = nn.Parameter(
            scene_data["color_dc"].to(self.attr_dtype).to(self.device)
        )

        exp_sh_coeffs = 3 * ((1 + self.sh_degree) * (1 + self.sh_degree) - 1)
        got_sh_coeffs = scene_data["color_sh"].shape[-1]
        assert (
            exp_sh_coeffs == got_sh_coeffs
        ), f"Expected {exp_sh_coeffs} SH coeffs per-point, got {got_sh_coeffs}"
        self.att_sh = nn.Parameter(
            scene_data["color_sh"].to(self.attr_dtype).to(self.device)
        )

        self.point_adjacency = scene_data["adjacency"].to(self.device).to(
            torch.uint32)
        self.point_adjacency_offsets = scene_data["adjacency_offsets"].to(
            self.device
        ).to(torch.uint32)

        self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)
