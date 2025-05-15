from pathlib import Path
import time
import torch
from configs import *
from radfoam_model.scene import RadFoamScene
from nerfstudio.field_components.encodings import HashEncoding
from data_loader import DataHandler

class ComputeUncertainty:
    """Load a checkpoint, compute uncertainty, and save it to a npy file."""

    # location of model params
    checkpoint_path = f"output/bonsai@44a5858d/model.pt"
    # Name of the output file.
    output_path: Path = Path("unc.npy")
    # Uncertainty level of detail (log2 of it)
    lod: int = 8
    # number of iterations on the trainset
    iters: int = 1000
    # deform covariance matrix
    deform_cov: bool = False


    def calc_hessian(self, model):
        print("Computing Hessian")
        start_time = time.time()

        '''iter2downsample = dict(
            zip(
                dataset_args.downsample_iterations,
                dataset_args.downsample,
            )
        )
        train_data_handler = DataHandler(
            dataset_args, rays_per_batch=250_000, device=device
        )
        downsample = iter2downsample[0]
        train_data_handler.reload(split="train", downsample=downsample)

        torch.cuda.synchronize()

        data_iterator = train_data_handler.get_iter()
        ray_batch, rgb_batch, alpha_batch = next(data_iterator)'''

        # Initialize your deformation field and Hessian storage
        hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        deform_field = HashEncoding(num_levels=1,
                                         min_res=2 ** self.lod,
                                         max_res=2 ** self.lod,
                                         log2_hashmap_size=self.lod * 3 + 1,
                                         # simple regular grid (hash table size > grid size)
                                         features_per_level=3,
                                         hash_init_scale=0.,
                                         implementation="torch",
                                         interpolation="Linear")
        deform_field.to(self.device)
        deform_field.scalings = torch.tensor([2 ** self.lod]).to(self.device)

        # Disable gradient computation for model parameters
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        print(model.primal_points)


        '''if not self.deform_cov:
            # get the offsets from the deform field
            normalized_points = normalize_point_coords(means_crop)
            offsets_1 = self.deform_field_pos(normalized_points).clone().detach()
            offsets_1.requires_grad = True

            means_crop = means_crop + offsets_1
        else:
            normalized_points_quats = normalize_point_coords(means_crop)
            offsets_1 = self.deform_field_quats(normalized_points_quats).clone().detach()
            offsets_1.requires_grad = True

            quats_crop = quats_crop + offsets_1



        with tqdm.trange(pipeline_args.iterations) as compute_loop:
            for i in compute_loop:
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
                # Forward pass
                with torch.no_grad():
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


                # Compute your Hessian here
                # You'll need to:
                # 1. Get the points from the ray samples (similar to your get_unc_mipnerf method)
                # 2. Compute offsets using your deform_field
                # 3. Compute the Hessian using your find_uncertainty method

                # For example:
                # points = ... (extract from ray samples)
                # pos, _ = normalize_point_coords(points, aabb, model.field.spatial_distortion)
                # offsets = deform_field(pos).clone().detach()
                # offsets.requires_grad = True
                # current_hessian = find_uncertainty(points, offsets, rgb_output, model.field.spatial_distortion)
                # hessian += current_hessian.clone().detach()

                # Load next batch
                ray_batch, rgb_batch, alpha_batch = next(data_iterator)
        '''
        end_time = time.time()
        print("Done")
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")




    def main(self):
            parser = configargparse.ArgParser()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            model_params = ModelParams(parser)
            dataset_params = DatasetParams(parser)
            args = parser.parse_args()
            self.device = torch.device(args.device)

            model = RadFoamScene(args=model_params.extract(args), device=self.device)

            model.load_pt(f"{self.checkpoint_path}")

            #self.aabb = model.aabb_tree.to(self.device)


            self.calc_hessian(model)

if name == "__main__":
    compute_uncertainty = ComputeUncertainty()
    compute_uncertainty.main()



