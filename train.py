import os
import uuid
import yaml
import gc
import numpy as np
from PIL import Image
import configargparse
import tqdm
import warnings
from radfoam_model.render import TraceRays

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
from uncertainty import ComputeUncertainty, log_uncertainty_metrics
import radfoam
from hessian_approximater import HessianAutograd
from hessian_app import HessianApp


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)
    # Setting up output directory
    if not pipeline_args.debug:
        if len(pipeline_args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{dataset_args.scene}@{unique_str}"
        else:
            experiment_name = pipeline_args.experiment_name
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataset
    iter2downsample = dict(
        zip(
            dataset_args.downsample_iterations,
            dataset_args.downsample,
        )
    )
    train_data_handler = DataHandler(
        dataset_args, rays_per_batch=750_000, device=device
    )
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )
    test_ray_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rays, batch_size=1, shuffle=False
    )
    test_rgb_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rgbs, batch_size=1, shuffle=False
    )

    # Define viewer settings
    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
    }

    # Setting up pipeline
    rgb_loss = nn.SmoothL1Loss(reduction="none")

    #Setting up Uncertainty computer
    uncertainty_computer = ComputeUncertainty(args, pipeline_args,
                                             model_args, optimizer_args,
                                             dataset_args)

    # Setting up model
    model = RadFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors
    )

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    def test_render(
        test_data_handler, ray_batch_fetcher, rgb_batch_fetcher, debug=False
    ):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        psnr_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]
                output, _, _, _, _ = model(ray_batch, start_points[i])

                # White background
                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean()
                psnr_list.append(img_psnr)
                torch.cuda.synchronize()

                if not debug:
                    error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                    rgb_output = np.uint8(rgb_output.cpu() * 255)
                    rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                    im = Image.fromarray(
                        np.concatenate([rgb_output, rgb_batch, error], axis=1)
                    )
                    im.save(
                        f"{out_dir}/test/rgb_{i:03d}_psnr_{img_psnr:.3f}.png"
                    )

        average_psnr = sum(psnr_list) / len(psnr_list)
        if not debug:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average PSNR: {average_psnr}")
            f.close()

        return average_psnr

    def train_loop(viewer):
        print("Training")

        torch.cuda.synchronize()

        data_iterator = train_data_handler.get_iter()
        ray_batch, rgb_batch, alpha_batch = next(data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        with tqdm.trange(pipeline_args.iterations) as train:
            for i in train:
                if viewer is not None:
                    model.update_viewer(viewer)
                    viewer.step(i)

                if i in iter2downsample and i:
                    downsample = iter2downsample[i]
                    train_data_handler.reload(
                        split="train", downsample=downsample
                    )
                    data_iterator = train_data_handler.get_iter()
                    ray_batch, rgb_batch, alpha_batch = next(data_iterator)

                depth_quantiles = (
                    torch.rand(*ray_batch.shape[:-1], 2, device=device)
                    .sort(dim=-1, descending=True)
                    .values
                )

                rgba_output, depth, _, _, _ = model(
                    ray_batch,
                    depth_quantiles=depth_quantiles,
                )

                # White background
                opacity = rgba_output[..., -1:]
                if pipeline_args.white_background:
                    rgb_output = rgba_output[..., :3] + (1 - opacity)
                else:
                    rgb_output = rgba_output[..., :3]

                color_loss = rgb_loss(rgb_batch, rgb_output)
                opacity_loss = ((alpha_batch - opacity) ** 2).mean()

                valid_depth_mask = (depth > 0).all(dim=-1)
                quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                quant_loss = (quant_loss * valid_depth_mask).mean()
                w_depth = pipeline_args.quantile_weight * min(
                    2 * i / pipeline_args.iterations, 1
                )

                uncertainty = uncertainty_computer.get_ray_uncertainty(ray_batch, model)
                #confidence = 1 - uncertainty
                #weighted_color_loss = confidence.unsqueeze(1) * color_loss
                #w_uncertainty = 0.01
                #uncertainty_penalty = w_uncertainty * uncertainty
                #if i > 10000 and i % 2 == 0:
                    #loss = weighted_color_loss.mean() + opacity_loss + w_depth * quant_loss + uncertainty_penalty.mean()
                #else:
                #loss = color_loss.mean()
                loss = color_loss.mean() + opacity_loss + w_depth * quant_loss #+ uncertainty_penalty.mean()"""

                #jtj_loss = JTJLoss(model)
                #print(jtj_loss)
                #jtj_loss.backward()


                #primal_points = model.primal_points.clone().detach()

                # Make sure inputs require grad
                
                

                """hessian_fn = HessianAutograd.apply
                gradients = hessian_fn(ray_batch, model, 0)
                print(gradients.shape)
                gradients.backward()"""
                #g = HessianApp(model)
                
                #h = g.forward(ray_batch, primal_points)
                #print("h:", h.shape)
                #print(h)
                #exit(1)
                """primal_points = model.primal_points.clone().detach()
                offsets = torch.ones_like(primal_points) * 0.01 * torch.randn_like(primal_points)
                #offsets_1 = self.deform_field(normalized_points).clone().detach()#offsets#.clone().detach()
                offsets.requires_grad = True
                deformed_points = primal_points + offsets

                rgba_output2, depth2, ray_samples2, _, _ = model(
                    ray_batch,
                    primal_points=deformed_points
                )
                print(f"rgba_output2.requires_grad: {rgba_output2.requires_grad}")
                

                gradients = []
                TraceRays.RETAIN_GRAPH = True #if i < 2 else False
                rgb = rgba_output2.view(-1, 3)
                colors = torch.sum(rgb, dim=0)
                grad_output = torch.zeros_like(colors)  # No requires_grad yet
                grad_output[i] = 1.0
                grad_output.requires_grad_()
                #grad_output.requires_grad = True

                print(f"colors.requires_grad: {colors.requires_grad}")
                print(f"offsets.requires_grad: {offsets.requires_grad}")

                #grad = torch.autograd.grad(colors, offsets, grad_outputs=grad_output, 
                                        #retain_graph=True, create_graph=True)[0]#.clone().detach()
                grad_from_autograd = torch.autograd.grad(
                    colors, 
                    offsets, 
                    grad_outputs=grad_output, 
                    retain_graph=True, 
                    create_graph=True,
                    allow_unused=True  # Add this to be safe
                )[0]
                #print(f"grad.requires_grad: {grad.requires_grad}")
                #print(f"grad.grad_fn: {grad.grad_fn}")
                #gradients.append(grad.view(-1, 3))

                loss = grad_from_autograd.mean()
                print(f"loss: {loss.item()}")"""
                #exit(1)

                inde = 59

                with open(f"uncertainty_loss_log{inde}.txt", "a") as f:
                    f.write(f"{i} {uncertainty.mean()}\n")
                #with open(f"uncertainty_penalty_loss_log{inde}.txt", "a") as f:
                    #f.write(f"{i} {uncertainty_penalty.mean()}\n")
                with open(f"color_loss_log{inde}.txt", "a") as f:
                    f.write(f"{i} {color_loss.mean().item()}\n")
                #with open(f"weighted_color_loss_log{inde}.txt", "a") as f:
                    #f.write(f"{i} {weighted_color_loss.mean().item()}\n")
                with open(f"opacity_loss_log{inde}.txt", "a") as f:
                    f.write(f"{i} {opacity_loss.item()}\n")
                with open(f"quant_loss_log{inde}.txt", "a") as f:
                    f.write(f"{i} {w_depth * quant_loss.item()}\n")

                

                model.optimizer.zero_grad(set_to_none=True)

                # Hide latency of data loading behind the backward pass
                event = torch.cuda.Event()
                event.record()
                loss.backward()
                """for name, param in uncertainty_computer.deform_field.named_parameters():
                    if param.grad is not None:
                        print(f"Deform field param {name} grad norm: {param.grad.norm().item()}")"""

                # For the model (optional, since this is a secondary path)
                """for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Model param {name} grad norm: {param.grad.norm().item()}")"""

                event.synchronize()
                ray_batch, rgb_batch, alpha_batch = next(data_iterator)

                model.optimizer.step()
                model.update_learning_rate(i)

                train.set_postfix(color_loss=f"{color_loss.mean().item():.5f}")

                if i % 500 == 0:
                    point_uncertainty = uncertainty_computer.get_prune_uncertainty(model)
                    log_uncertainty_metrics(
                        point_uncertainty=point_uncertainty.squeeze(),
                        label=f"Training epoch{i}",
                        log_file="uncertainty_log.txt"
                    )

                if i % 100 == 99 and not pipeline_args.debug:
                    writer.add_scalar("train/rgb_loss", color_loss.mean(), i)
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)

                    test_psnr = test_render(
                        test_data_handler,
                        test_ray_batch_fetcher,
                        test_rgb_batch_fetcher,
                        True,
                    )
                    writer.add_scalar("test/psnr", test_psnr, i)

                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/attr_lr", model.attr_dc_scheduler_args(i), i
                    )

                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0

                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (
                    iters_since_densification == next_densification_after
                    and model.primal_points.shape[0]
                    < 0.9 * model.num_final_points
                ):
                    point_error, point_contribution = model.collect_error_map(
                        train_data_handler, pipeline_args.white_background
                    )
                    #uncertainty_computer.determine_hessian(model)
                    point_uncertainty = uncertainty_computer.get_prune_uncertainty(model)
                    log_uncertainty_metrics(
                        point_uncertainty=point_uncertainty.squeeze(),
                        label=f"Before Pruning (iter {i})",
                        log_file="uncertainty_log.txt"
                    )
                    model.prune_and_densify(
                        point_error,
                        point_contribution,
                        point_uncertainty,
                        pipeline_args.densify_factor,
                        iter = i
                    )

                    model.update_triangulation(incremental=False)
                    triangulation_update_period = 1

                    point_uncertainty = uncertainty_computer.get_prune_uncertainty(model)
                    log_uncertainty_metrics(
                        point_uncertainty=point_uncertainty.squeeze(), 
                        label=f"After Pruning (iter {i})",
                        log_file="uncertainty_log.txt"
                    )
                    gc.collect()

                    # Linear growth
                    iters_since_densification = 0
                    next_densification_after = int(
                        (
                            (pipeline_args.densify_factor - 1)
                            * model.primal_points.shape[0]
                            * (
                                pipeline_args.densify_until
                                - pipeline_args.densify_from
                            )
                        )
                        / (model.num_final_points - model.num_init_points)
                    )
                    next_densification_after = max(
                        next_densification_after, 100
                    )

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

        model.save_ply(f"{out_dir}/scene.ply")
        model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    if pipeline_args.viewer:
        model.show(
            train_loop, iterations=pipeline_args.iterations, **viewer_options
        )
    else:
        train_loop(viewer=None)
    if not pipeline_args.debug:
        writer.close()

    test_render(
        test_data_handler,
        test_ray_batch_fetcher,
        test_rgb_batch_fetcher,
        pipeline_args.debug,
    )
    point_uncertainty = uncertainty_computer.get_prune_uncertainty(model)
    log_uncertainty_metrics(
        point_uncertainty=point_uncertainty.squeeze(),
        label="End of Training",
        log_file="uncertainty_log.txt"
    )



def main():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
