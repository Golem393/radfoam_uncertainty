import os

import numpy as np
import einops
import torch

import radfoam

from .colmap import COLMAPDataset
from .blender import BlenderDataset
from .nerfbuster import NerfbusterDataset, visualize_vol_rend_plt

from xvfbwrapper import Xvfb


dataset_dict = {
    "colmap": COLMAPDataset,
    "blender": BlenderDataset,
    "nerfbuster": NerfbusterDataset,
}


def get_up(c2ws):
    right = c2ws[:, :3, 0]
    down = c2ws[:, :3, 1]
    forward = c2ws[:, :3, 2]

    A = torch.einsum("bi,bj->bij", right, right).sum(dim=0)
    A += torch.einsum("bi,bj->bij", forward, forward).sum(dim=0) * 0.02

    l, V = torch.linalg.eig(A)

    min_idx = torch.argmin(l.real)
    global_up = V[:, min_idx].real
    global_up *= torch.einsum("bi,i->b", -down, global_up).sum().sign()

    return global_up


class DataHandler:
    def __init__(self, dataset_args, rays_per_batch, device="cuda"):
        self.args = dataset_args
        self.rays_per_batch = rays_per_batch
        self.device = torch.device(device)
        self.img_wh = None
        self.patch_size = 8

    def reload(self, split, downsample=None):
        data_dir = os.path.join(self.args.data_path, self.args.scene)
        dataset = dataset_dict[self.args.dataset]
        if downsample is not None:
            split_dataset = dataset(
                data_dir, split=split, downsample=downsample
            )
        else:
            split_dataset = dataset(data_dir, split=split)

                
        # Example: use the first image (idx = 0)
        # idx = 0

        # pose = split_dataset.poses[idx].cpu().numpy()
        # K = split_dataset.intrinsics.cpu().numpy()
        # rays = split_dataset.all_rays[idx].reshape(-1, 6).cpu().numpy()
        # centers = rays[:, :3]      # (H*W, 3)
        # directions = rays[:, 3:]   # (H*W, 3)

        # # Near/far for each ray (set as needed)
        # t_min = np.zeros((centers.shape[0],), dtype=np.float32)
        # t_max = np.ones((centers.shape[0],), dtype=np.float32) * 5.0

        # # Image size
        # latent_hw = (split_dataset.img_wh[1], split_dataset.img_wh[0])  # (H, W)

        # # Bounding box (example, set as needed)
        # bbox = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)

        # # Grid points (example, set as needed)
        # grid_points = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)

        # # Optionally subsample rays for visualization
        # n_vis = 500
        # idxs = np.random.choice(centers.shape[0], n_vis, replace=False)
        # vis_rays = directions[idxs]
        # vis_centers = centers[idxs]
        # vis_t_min = t_min[idxs]
        # vis_t_max = t_max[idxs]

        # # Now you can call your visualizer
        # # with Xvfb(width=1400, height=900, colordepth=24):
        # visualize_vol_rend_plt(
        #     rays=vis_rays,
        #     centers=vis_centers,
        #     t_min=vis_t_min,
        #     t_max=vis_t_max,
        #     grid_points=grid_points,
        #     pose=[pose],         # List of poses for each camera to visualize
        #     K=[K],               # List of intrinsics for each camera
        #     latent_hw=latent_hw,
        #     bbox=bbox
        # )

        self.img_wh = split_dataset.img_wh
        self.fx = split_dataset.fx
        self.fy = split_dataset.fy
        self.c2ws = split_dataset.poses
        self.rays, self.rgbs = split_dataset.all_rays, split_dataset.all_rgbs
        self.alphas = getattr(
            split_dataset, "all_alphas", torch.ones_like(self.rgbs[..., 0:1])
        )

        self.viewer_up = get_up(self.c2ws)
        self.viewer_pos = self.c2ws[0, :3, 3]
        self.viewer_forward = self.c2ws[0, :3, 2]

        try:
            self.points3D = split_dataset.points3D
            self.points3D_colors = split_dataset.points3D_color
        except:
            self.points3D = None
            self.points3D_colors = None

        if split == "train":
            if self.args.patch_based:
                dw = self.img_wh[0] - (self.img_wh[0] % self.patch_size)
                dh = self.img_wh[1] - (self.img_wh[1] % self.patch_size)
                w_inds = np.linspace(0, self.img_wh[0] - 1, dw, dtype=int)
                h_inds = np.linspace(0, self.img_wh[1] - 1, dh, dtype=int)

                self.train_rays = self.rays[:, h_inds, :, :]
                self.train_rays = self.train_rays[:, :, w_inds, :]
                self.train_rgbs = self.rgbs[:, h_inds, :, :]
                self.train_rgbs = self.train_rgbs[:, :, w_inds, :]

                self.train_rays = einops.rearrange(
                    self.train_rays,
                    "n (x ph) (y pw) r -> (n x y) ph pw r",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )
                self.train_rgbs = einops.rearrange(
                    self.train_rgbs,
                    "n (x ph) (y pw) c -> (n x y) ph pw c",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )

                self.batch_size = self.rays_per_batch // (self.patch_size**2)
                self.num_batches = self.train_rays.shape[0]
            else:
                self.train_rays = einops.rearrange(
                    self.rays, "n h w r -> (n h w) r"
                )
                self.train_rgbs = einops.rearrange(
                    self.rgbs, "n h w c -> (n h w) c"
                )
                self.train_alphas = einops.rearrange(
                    self.alphas, "n h w 1 -> (n h w) 1"
                )

                self.batch_size = self.rays_per_batch
                num_elements = self.train_rays.shape[0]
                self.num_batches = num_elements // self.batch_size

    def get_iter(self, random=True):
        shuffle = random

        ray_batch_fetcher = radfoam.BatchFetcher(
            self.train_rays, self.batch_size, shuffle=shuffle
        )
        rgb_batch_fetcher = radfoam.BatchFetcher(
            self.train_rgbs, self.batch_size, shuffle=shuffle
        )
        alpha_batch_fetcher = radfoam.BatchFetcher(
            self.train_alphas, self.batch_size, shuffle=shuffle
        )

        while True:
            ray_batch = ray_batch_fetcher.next()
            rgb_batch = rgb_batch_fetcher.next()
            alpha_batch = alpha_batch_fetcher.next()

            yield ray_batch, rgb_batch, alpha_batch

    def __len__(self):
        return self.num_batches
