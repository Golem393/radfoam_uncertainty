import torch
from gsplat import spherical_harmonics
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components import renderers

from bayessplatting.utils.utils import find_grid_indices


def get_uncertainty(self, points, lod, un):
    inds, coeffs = find_grid_indices(points, lod, points.device)
    cfs_2 = (coeffs ** 2) / torch.sum((coeffs ** 2), dim=0, keepdim=True)
    uns = un[inds.long()]  # [8,N]
    un_points = torch.sqrt(torch.sum((uns * cfs_2), dim=0)).unsqueeze(1)

    # for stability in volume rendering we use log uncertainty
    un_points = torch.log10(un_points + 1e-12)
    return un_points


def get_outputs(self, model, ray_batch, rgb_batch, alpha_batch, filter_out: bool = False):


    # get the background color  TODO
    

    N = self.N
    reg_lambda = 1e-4 / ((2 ** self.lod) ** 3)
    H = self.hessian / N + reg_lambda
    self.un = 1 / H

    max_uncertainty = 6  # approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3  # approximate lower bound of that function (cutting off at hessian = 1000)


    primal_points = model.primal_points.clone().detach()
    un_points = self.get_uncertainty(primal_points).view(-1)

    # Normalize the uncertainty values between 0 and 1
    un_points_min = un_points.min()
    un_points_max = un_points.max()

    un_points_cp = (un_points - un_points_min) / (un_points_max - un_points_min)

    # TODO Filter out Gaussians with uncertainty greater than the threshold
    

    # breakpoint()

    #this is for radfoam

    depth_quantiles = (
            torch.rand(*ray_batch.shape[:-1], 2, device=self.device)
            .sort(dim=-1, descending=True)
            .values
        )

    # Forward pass through the model to get RGBA
    rgba_output, depth, ray_samples, _, _ = model(
        ray_batch,
        depth_quantiles=depth_quantiles,
        primal_points = deformed_points,
        uncertainty = un_points_cp
    )
    print(rgba_output)
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
    
    #return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "uncertainty": uncertainty_im,
            #"background": background}  # type: ignore


def main():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)
    args = parser.parse_args()

    model = RadFoamScene(args=self.model_params, device=self.device)
    model.load_pt(f"{self.checkpoint_path}")
    model.eval()
    hessian = 

    


if __name__ == "__main__":
    main()