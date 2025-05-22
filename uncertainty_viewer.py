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


def get_uncertainty(points, lod, un):
    inds, coeffs = find_grid_indices(points, lod, points.device)
    cfs_2 = (coeffs ** 2) / torch.sum((coeffs ** 2), dim=0, keepdim=True)
    uns = un[inds.long()].squeeze() # [8,N]
    un_points = torch.sqrt(torch.sum((uns * cfs_2), dim=0)).unsqueeze(1)

    # for stability in volume rendering we use log uncertainty
    un_points = torch.log10(un_points + 1e-12)
    return un_points


def get_outputs(model, ray_batch, rgb_batch, alpha_batch, hessian, device, filter_out: bool = False):


    # get the background color  TODO

    downscale_factor: float = 2.0
    width: int = 1280
    # width of the image
    height: int = 720
    # height of the image
    N = 1000 * (
            (width * height) / downscale_factor
        ) 
    lod = np.log2(round(hessian.shape[0] ** (1 / 3)) - 1)
    reg_lambda = 1e-4 / ((2 ** lod) ** 3)
    H = hessian / N + reg_lambda
    un = 1 / H

    max_uncertainty = 6  # approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3  # approximate lower bound of that function (cutting off at hessian = 1000)


    primal_points = model.primal_points.clone().detach()
    un_points = get_uncertainty(primal_points, lod, un).view(-1)

    # Normalize the uncertainty values between 0 and 1
    un_points_min = un_points.min()
    un_points_max = un_points.max()

    un_points_cp = (un_points - un_points_min) / (un_points_max - un_points_min)

    # TODO Filter out Gaussians with uncertainty greater than the threshold
    

    # breakpoint()

    #this is for radfoam

    depth_quantiles = (
            torch.rand(*ray_batch.shape[:-1], 2, device=device)
            .sort(dim=-1, descending=True)
            .values
        )

    # Forward pass through the model to get RGBA
    rgba_output, depth, ray_samples, _, _ = model(
        ray_batch,
        depth_quantiles=depth_quantiles,
        # primal_points = deformed_points,
        uncertainty = un_points_cp
    )
    return rgba_output
    '''
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
'''
    
    # return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "uncertainty": uncertainty_im,
    #         "background": background}  # type: ignore


def main():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)
    args = parser.parse_args()

    device = torch.device(args.device)

    model = RadFoamScene(args=model_params, device=device)
    model.load_pt(f"output/bonsai@44a5858d/model.pt")
    # model.eval()
    # open unc.npy for hessian
    hessian = torch.tensor(np.load(str("unc.npy"))).to(device)

    print("Computing Hessian")
    # start_time = time.time()

    # Load the dataset
    test_data_handler = DataHandler(dataset_params, rays_per_batch=0, device=device)
    test_data_handler.reload(split="test", downsample=min(dataset_params.downsample))
    test_ray_batch_fetcher = radfoam.BatchFetcher(test_data_handler.rays, batch_size=1, shuffle=False)
    test_rgb_batch_fetcher = radfoam.BatchFetcher(test_data_handler.rgbs, batch_size=1, shuffle=False)


    ray_batch = test_ray_batch_fetcher.next()[0]  # [H*W, 6]
    rgb_batch = test_rgb_batch_fetcher.next()[0]  # [H*W, 3]
    alpha_batch = torch.ones_like(rgb_batch[..., :1])  # dummy alpha
    #print("Length of training data:", len_train)
    #for i in range(len_train):
        #ray_batch, rgb_batch, alpha_batch = next(data_iterator)
        # outputs, points, offsets_1 = get_outputs(model, ray_batch, rgb_batch, alpha_batch, hessian, device)
    output = get_outputs(model, ray_batch, rgb_batch, alpha_batch, hessian, device)
        
        # hessian = self.find_uncertainty(points, offsets_1, outputs['rgb'].view(-1, 3))
        # self.hessian += hessian.clone().detach()
        #break

    with torch.no_grad():
        # White background
        print(output.shape)
        uncertainty = output[..., -1:].detach().cpu().numpy().squeeze(-1)
        print(uncertainty.shape)
        uncertainty = np.clip(uncertainty, 0, 1)  # ensure it's between 0-1
        print(uncertainty.shape)
        uncertainty_img = np.uint8(uncertainty * 255)
        Image.fromarray(uncertainty_img, mode="L").save("output/uncertainty.png")


if __name__ == "__main__":
    main()