import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D plotting


def get_ray_directions(H, W, focal, center=None):
    """
    Generate normalized ray directions in camera coordinates for each pixel.

    Args:
        H (int): Image height.
        W (int): Image width.
        focal (list or tuple): Focal lengths [fx, fy].
        center (list or tuple, optional): Principal point [cx, cy]. Defaults to image center.

    Returns:
        torch.Tensor: (H*W, 3) array of normalized ray directions.
    """
    x = np.arange(W, dtype=np.float32) + 0.5
    y = np.arange(H, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    i, j = pix_coords[..., 0:1], pix_coords[..., 1:]

    cent = center if center is not None else [W / 2, H / 2]
    directions = np.concatenate(
        [
            (i - cent[0]) / focal[0],  # x direction
            (j - cent[1]) / focal[1],  # y direction
            np.ones_like(i),           # z direction (forward)
        ],
        axis=-1,
    )
    ray_dirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)


def auto_orient_and_center_poses(poses, up_vector=torch.tensor([0, 0, 1], dtype=torch.float32)):
    """
    Centers and orients camera poses so that the mean camera position is at the origin
    and the up vector is aligned with the specified up_vector (default: [0,0,1]).

    Args:
        poses: (N, 4, 4) torch tensor of camera-to-world matrices.
        up_vector: (3,) torch tensor, desired up direction.

    Returns:
        poses_centered: (N, 4, 4) torch tensor, centered and oriented poses.
    """
    device = poses.device
    # Center: subtract mean translation
    mean_position = poses[:, :3, 3].mean(dim=0)
    poses_centered = poses.clone()
    poses_centered[:, :3, 3] -= mean_position

    # Orient: align average up vector to desired up_vector
    avg_up = poses_centered[:, :3, 1].mean(dim=0)
    avg_up = avg_up / torch.norm(avg_up)
    target_up = up_vector.to(device) / torch.norm(up_vector)
    v = torch.cross(avg_up, target_up)
    s = torch.norm(v)
    c = torch.dot(avg_up, target_up)
    if s < 1e-6:
        R = torch.eye(3, device=device)
    else:
        vx = torch.tensor([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]], device=device)
        R = torch.eye(3, device=device) + vx + vx @ vx * ((1 - c) / (s ** 2))
    poses_centered[:, :3, :3] = torch.matmul(R, poses_centered[:, :3, :3])

    return poses_centered


def visualize_vol_rend(rays, centers, t_min, t_max, grid_points, pose, K, latent_hw, bbox):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=512, height=512, visible=False)
    meshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    visualizer.add_geometry(meshFrame)

    for cam, intr in zip(pose, K):
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=latent_hw[1],
            view_height_px=latent_hw[0],
            intrinsic=intr[:3, :3],
            extrinsic=cam,
            scale=1.0,
        )
        visualizer.add_geometry(cameraLines)

    grid_pcl = o3d.geometry.PointCloud()
    grid_pcl.points = o3d.utility.Vector3dVector(grid_points)
    grid_colors = np.zeros_like(grid_points)
    grid_colors[:, 1] = 1  # green
    grid_pcl.colors = o3d.utility.Vector3dVector(grid_colors)
    visualizer.add_geometry(grid_pcl)

    n_steps = 64
    ray_steps = torch.arange(n_steps)
    ray_steps = ray_steps.unsqueeze(0).numpy()
    frac_step = ray_steps / (n_steps - 1)

    centers = centers[:, None, :]
    rays = rays[:, None, :]
    depth = (t_max[:, None] - t_min[:, None]) * frac_step + t_min[:, None]
    ray_points = centers + depth[..., None] * rays
    ray_points = ray_points.reshape((-1, 3))

    ray_pcl = o3d.geometry.PointCloud()
    ray_pcl.points = o3d.utility.Vector3dVector(ray_points)
    ray_colors = np.zeros_like(ray_points)
    ray_colors[:, 0] = 1  # red
    ray_pcl.colors = o3d.utility.Vector3dVector(ray_colors)
    visualizer.add_geometry(ray_pcl)

    lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox[0],
            max_bound=bbox[1],
        )
    )
    lineset_colors = np.array(lineset.colors)
    lineset_colors[:, 0] = 0.0
    lineset_colors[:, 1] = 0.0
    lineset_colors[:, 2] = 1.0
    lineset.colors = o3d.utility.Vector3dVector(lineset_colors)
    visualizer.add_geometry(lineset)

    # visualizer.run()
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image("output.png")
    visualizer.destroy_window()


def visualize_vol_rend_plt(rays, centers, t_min, t_max, grid_points, pose, K, latent_hw, bbox):
    # Compute ray points as before
    n_steps = 64
    ray_steps = torch.arange(n_steps)
    ray_steps = ray_steps.unsqueeze(0).numpy()
    frac_step = ray_steps / (n_steps - 1)

    centers = centers[:, None, :]
    rays = rays[:, None, :]
    depth = (t_max[:, None] - t_min[:, None]) * frac_step + t_min[:, None]
    ray_points = centers + depth[..., None] * rays
    ray_points = ray_points.reshape((-1, 3))

    # Start plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot grid points (green)
    ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], c='g', s=2, label='Grid Points')

    # Plot ray points (red)
    ax.scatter(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], c='r', s=1, label='Ray Points')

    # Plot bounding box (blue)
    bbox_min, bbox_max = bbox
    for s, e in [
        # 12 edges of a box
        ([0,0,0],[1,0,0]), ([0,0,0],[0,1,0]), ([0,0,0],[0,0,1]),
        ([1,0,0],[1,1,0]), ([1,0,0],[1,0,1]),
        ([0,1,0],[1,1,0]), ([0,1,0],[0,1,1]),
        ([0,0,1],[1,0,1]), ([0,0,1],[0,1,1]),
        ([1,1,0],[1,1,1]), ([1,0,1],[1,1,1]), ([0,1,1],[1,1,1])
    ]:
        s = [bbox_min[i] if v == 0 else bbox_max[i] for i, v in enumerate(s)]
        e = [bbox_min[i] if v == 0 else bbox_max[i] for i, v in enumerate(e)]
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close(fig)

class NerfbusterDataset(Dataset):
    """
    Custom dataparser for loading Nerfbuster-style NeRF datasets for Radfoam.

    Attributes:
        self.poses: (N, 4, 4) camera-to-world matrices for each image.
        self.all_rays: (N, H, W, 6) rays [origin, direction] for each pixel in each image.
        self.all_rgbs: (N, H, W, 3) RGB values for each pixel in each image.
        self.all_alphas: (N, H, W, 1) Alpha channel for each pixel in each image.
        self.intrinsics: (3, 3) camera intrinsic matrix.
    """
    def __init__(self, datadir, split="train", downsample=1):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample

        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        # Load metadata
        with open(os.path.join(self.root_dir, f"transforms.json"), "r") as f:
            meta = json.load(f)

        # Get image size and intrinsics
        if "w" in meta and "h" in meta:
            W, H = int(meta["w"]), int(meta["h"])
        else:
            W, H = 800, 800

        self.img_wh = (int(W / self.downsample), int(H / self.downsample))
        w, h = self.img_wh

        # Focal length and principal point
        fx = float(meta["fl_x"]) if "fl_x" in meta else 0.5 * w / np.tan(0.5 * meta["camera_angle_x"])
        fy = float(meta["fl_y"]) if "fl_y" in meta else fx
        cx = float(meta["cx"]) / self.downsample if "cx" in meta else w / 2
        cy = float(meta["cy"]) / self.downsample if "cy" in meta else h / 2

        if self.downsample != 1.0:
            fx /= self.downsample
            fy /= self.downsample

        self.fx, self.fy = fx, fy

        self.intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Precompute ray directions in camera coordinates
        cam_ray_dirs = get_ray_directions(h, w, [fx, fy], center=[cx, cy])

        applied_transform = None
        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=torch.float32)
            if applied_transform.shape == (3, 4):
                applied_transform = torch.cat(
                    [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], 0
                )
        else:
            # fallback as in the comment
            applied_transform = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        applied_scale = 1.0
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_alphas = []
        frames = meta["frames"]
        for frame in frames:
            # Load pose (camera-to-world transformation)
            pose = np.array(frame["transform_matrix"], dtype=np.float32)
            pose = torch.from_numpy(pose)
            # Apply applied_transform
            pose = pose @ applied_transform
            # Apply scale if present
            pose[:3, 3] *= applied_scale
            # Convert from Blender to OpenCV convention
            pose = pose @ torch.from_numpy(self.blender2opencv).float()
            c2w = pose
            self.poses.append(c2w)

            # Transform ray directions from camera to world coordinates
            world_ray_dirs = torch.einsum("ij,kj->ik", cam_ray_dirs, c2w[:3, :3])
            # Ray origins: camera center for each pixel
            world_ray_origins = c2w[:3, 3] + torch.zeros_like(cam_ray_dirs)
            # Concatenate origins and directions
            world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
            # Reshape to (H, W, 6)
            world_rays = world_rays.reshape(self.img_wh[1], self.img_wh[0], 6)

            # Load image and process RGBA channels
            img_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            img = Image.open(img_path)
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = img.convert("RGBA")
            rgbas = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            # Composite onto white background using alpha
            rgbs = rgbas[..., :3] * rgbas[..., 3:4] + (1 - rgbas[..., 3:4])
            img.close()

            self.all_rays.append(world_rays)
            self.all_rgbs.append(rgbs)
            self.all_alphas.append(rgbas[..., -1:])

        self.poses = torch.stack(self.poses)
        self.poses = auto_orient_and_center_poses(self.poses)
        self.all_rays = torch.stack(self.all_rays)
        self.all_rgbs = torch.stack(self.all_rgbs)
        self.all_alphas = torch.stack(self.all_alphas)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        sample = {
            "rays": self.all_rays[idx],
            "rgbs": self.all_rgbs[idx],
            "alphas": self.all_alphas[idx],
        }
        return sample