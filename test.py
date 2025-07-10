import numpy as np
from PIL import Image
import configargparse
import warnings

warnings.filterwarnings("ignore")

import torch

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
import radfoam

from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure

from metrics.ause import ause


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def test(args, pipeline_args, model_args, optimizer_args, dataset_args, eval_depth=True, save_images=False):
    checkpoint = args.config.replace("/config.yaml", "")
    os.makedirs(os.path.join(checkpoint, "test"), exist_ok=True)
    device = torch.device(args.device)

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

    # Setting up model
    model = RadFoamScene(args=model_args, device=device)

    model.load_pt(f"{checkpoint}/model.pt")

    def test_render(
        test_data_handler, ray_batch_fetcher, rgb_batch_fetcher
    ):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        psnr_list = []
        lpips_list = []
        ssim_list = []
        
        ause_mse_list = []
        ause_mae_list = []
        ause_rmse_list = []
        depth_break = False

        lpips_model = LPIPS(net='alex').to(device)  # Initialize LPIPS model
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # Initialize SSIM metric

        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]

                H, W = ray_batch.shape[:2]
                num_rays = H * W
                depth_quantiles = torch.full((num_rays, 1), 0.5, device=device)

                output, depth, _, _, _ = model(ray_batch, start_points[i], depth_quantiles=depth_quantiles)

                uncertainty = output[..., 3]

                # White background
                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                # Add batch dimension if missing
                if rgb_output.dim() == 3:
                    rgb_output = rgb_output.unsqueeze(0)  # Add batch dimension
                if rgb_batch.dim() == 3:
                    rgb_batch = rgb_batch.unsqueeze(0)  # Add batch dimension
                if uncertainty.dim() == 2:
                    uncertainty = uncertainty.unsqueeze(0)

                # PSNR Calculation
                img_psnr = psnr(rgb_output, rgb_batch).mean()
                psnr_list.append(img_psnr)

                # LPIPS Calculation
                lpips_score = lpips_model(
                    rgb_output.permute(0, 3, 1, 2), rgb_batch.permute(0, 3, 1, 2)
                ).mean()
                lpips_list.append(lpips_score.item())

                # SSIM Calculation
                ssim_score = ssim_metric(
                    rgb_output.permute(0, 3, 1, 2), rgb_batch.permute(0, 3, 1, 2)
                )
                ssim_list.append(ssim_score.item())

                #uncertainty
                if eval_depth and not depth_break:
                    # Load scale factor
                    dataset_path = os.path.join(test_data_handler.args.data_path, test_data_handler.args.scene)
                    a = float(np.loadtxt(str(dataset_path) + '/scale_parameters.txt', delimiter=','))
                    depth = a * depth

                    # Load GT depth
                    try:
                        depth_gt_dir = os.path.join(dataset_path, f'depth_gt_{i:02d}.npy')
                        depth_gt = np.load(depth_gt_dir)
                        depth_gt = torch.tensor(depth_gt, device=depth.device)
                    except FileNotFoundError as e:
                        print(f"[red]File Not FOUND for view_no: {i}: {e}")
                        depth_break = True

                    if not depth_break:
                        # Resize GT depth to match predicted depth shape
                        if depth_gt.shape != depth.shape:
                            if depth_gt.ndim == 2:
                                depth_gt = depth_gt.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            elif depth_gt.ndim == 3:
                                depth_gt = depth_gt.unsqueeze(0)  # [1, H, W]
                            if depth.ndim == 2:
                                depth = depth.unsqueeze(0).unsqueeze(0)
                            elif depth.ndim == 3:
                                depth = depth.unsqueeze(0)
                            depth_gt = torch.nn.functional.interpolate(
                                depth_gt.float(), size=depth.shape[-2:], mode="bilinear", align_corners=False
                            ).squeeze()
                            if depth_gt.shape != depth.shape:
                                depth_gt = depth_gt.squeeze()

                        # Normalize
                        depth = depth / depth_gt.max()
                        depth_gt = depth_gt / depth_gt.max()

                        depth = depth.squeeze().cpu()
                        depth_gt = depth_gt.squeeze().cpu()
                        squared_error = (depth_gt - depth) ** 2
                        absolute_error = (depth_gt - depth).abs()
                        unc_flat = uncertainty.flatten().detach().cpu()
                        absolute_error_flat = absolute_error.flatten().detach().cpu()
                        squared_error_flat = squared_error.flatten().detach().cpu()

                        ratio, err_mse, err_var_mse, ause_mse = ause(unc_flat, squared_error_flat, err_type='mse')
                        ause_mse_list.append(ause_mse)

                        ratio, err_mae, err_var_mae, ause_mae = ause(unc_flat, absolute_error_flat, err_type='mae')
                        ause_mae_list.append(ause_mae)

                        ratio, err_rmse, err_var_rmse, ause_rmse = ause(unc_flat, squared_error_flat, err_type='rmse')
                        ause_rmse_list.append(ause_rmse)

                        depth_img = torch.clip(depth, min=0., max=1.)
                        absolute_error_img = torch.clip(absolute_error, min=0., max=1.)

                        if save_images:
                            path = self.output_path.parent / "plots" 
                            path.mkdir(parents=True, exist_ok=True)
                            im = Image.fromarray((depth_gt.cpu().numpy()* 255).astype('uint8'))
                            im.save(path / Path(str(no)+"_depth_gt.jpeg"))
                            im = Image.fromarray((depth_img.squeeze().cpu().numpy()* 255).astype('uint8'))
                            im.save(path / Path(str(no)+"_depth.jpeg"))
                            im = Image.fromarray(np.uint8(inferno(absolute_error_img.cpu().numpy()) * 255).squeeze() )
                            im.save(path / Path(str(no)+"_error.png"))

                            im = Image.fromarray((image.cpu().numpy()* 255).astype('uint8'))
                            im.save(path / Path(str(no)+"_gt_image.jpeg"))
                            uu, errr = visualize_ranks(unc.squeeze(-1).cpu().numpy(), absolute_error.cpu().numpy())
                            im = Image.fromarray(np.uint8(uu * 255))
                            im.save(path / Path(str(no)+"_unc_colored.png"))

                            im = Image.fromarray(np.uint8(errr * 255).squeeze())
                            im.save(path / Path(str(no)+"_error_colored.png"))


                torch.cuda.synchronize()

                # Save Images
                error = np.uint8((rgb_output.squeeze(0) - rgb_batch.squeeze(0)).cpu().abs() * 255)
                rgb_output = np.uint8(rgb_output.squeeze(0).cpu() * 255)
                rgb_batch = np.uint8(rgb_batch.squeeze(0).cpu() * 255)

                im = Image.fromarray(
                    np.concatenate([rgb_output, rgb_batch, error], axis=1)
                )
                im.save(
                    f"{checkpoint}/test/rgb_{i:03d}_psnr_{img_psnr:.3f}.png"
                )


        # Average Metrics
        average_psnr = sum(psnr_list) / len(psnr_list)
        average_lpips = sum(lpips_list) / len(lpips_list)
        average_ssim = sum(ssim_list) / len(ssim_list)
        
        # Average scalar AUSE values
        average_ause_mse = sum(ause_mse_list) / len(ause_mse_list)
        average_ause_mae = sum(ause_mae_list) / len(ause_mae_list)
        average_ause_rmse = sum(ause_rmse_list) / len(ause_rmse_list)


        # Save Metrics
        with open(f"{checkpoint}/metrics.txt", "w") as f:
            f.write(f"Average PSNR: {average_psnr}\n")
            f.write(f"Average LPIPS: {average_lpips}\n")
            f.write(f"Average SSIM: {average_ssim}\n")
            f.write(f"Average AUSE (MSE): {average_ause_mse}\n")
            f.write(f"Average AUSE (MAE): {average_ause_mae}\n")
            f.write(f"Average AUSE (RMSE): {average_ause_rmse}\n")

        return average_psnr, average_lpips, average_ssim

    test_render(
        test_data_handler, test_ray_batch_fetcher, test_rgb_batch_fetcher
    )


def main():
    parser = configargparse.ArgParser()

    model_params = ModelParams(parser)
    dataset_params = DatasetParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--checkpoint", help="Path to checkpoint file", default="output/bonsai@44a5858d"
    )

    # Parse arguments
    args = parser.parse_args()

    test(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
