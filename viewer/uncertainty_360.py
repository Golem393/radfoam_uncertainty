import math
from scipy.spatial.transform import Rotation as R

import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import pycolmap
import cv2


def generate_circular_camera_poses(num_frames=120, radius=2.0, elevation=-0.8, 
                                 translate_x=0, translate_y=0, translate_z=0,
                                 pitch_deg=0, yaw_deg=0, roll_deg=0):
    """
    Generate camera poses in a circular path around the origin.
    
    Args:
        num_frames: Number of frames in the video
        radius: Radius of the circular path
        elevation: Height of the camera above the origin
        
    Returns:
        List of 3x4 camera-to-world matrices (rotation + translation)
    """
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)
    poses = []
    for i in range(num_frames):
        angle = math.pi + math.pi * i / num_frames
        
        # Calculate camera position
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = elevation

        position = np.array([
            x + translate_x,
            y + translate_y, 
            z + translate_z
        ])
        
        # Calculate camera orientation (still looks at original origin)
        look_at_point = np.array([translate_x, translate_y, translate_z])
        
        # Calculate camera orientation (looking at origin)
        forward = look_at_point - position
        forward = forward / np.linalg.norm(forward)

        # Apply rotation offsets
        if yaw_deg != 0 or pitch_deg != 0 or roll_deg != 0:
            # Create rotation matrices
            Ry = np.array([  # Yaw (Y-axis)
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)]
            ])
            
            Rx = np.array([  # Pitch (X-axis)
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)]
            ])
            
            Rz = np.array([  # Roll (Z-axis)
                [math.cos(roll), -math.sin(roll), 0],
                [math.sin(roll), math.cos(roll), 0],
                [0, 0, 1]
            ])
            
            # Combined rotation
            R = Rz @ Rx @ Ry
            forward = R @ forward


        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        # Build the 3x4 matrix (rotation + translation)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, up, forward], axis=-1)
        c2w[:3, 3] = position
        
        # Convert to 3x4 and return
        poses.append(c2w[:3])
    
    return torch.tensor(np.stack(poses), dtype=torch.float32)

def get_cam_ray_dirs(camera):
    x = np.arange(camera.width, dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate(
        [ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1
    )
    ray_dirs = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)

class Uncertainty360Viewer:
    def __init__(self, device):
        self.device = device
        data_dir = "/mnt/hdd/team4/data/mipnerf360/bonsai"
        self.colmap_dir = os.path.join(data_dir, "sparse/0/")
        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read(self.colmap_dir)

        self.camera = list(self.reconstruction.cameras.values())[0]
        self.img_wh = [3118, 2078]
        self.camera.rescale(self.img_wh[0], self.img_wh[1])

        self.fx = self.camera.focal_length_x
        self.fy = self.camera.focal_length_y

        cam_ray_dirs = get_cam_ray_dirs(self.camera)


        self.poses = generate_circular_camera_poses(translate_y=0.5, translate_z = -2.0,  pitch_deg=0, yaw_deg=30, roll_deg=0)
        self.all_rays = []
        for pose in tqdm(self.poses):
            world_ray_dirs = torch.einsum(
                "ij,kj->ik",
                cam_ray_dirs,
                pose[:, :3],
            )
            world_ray_origins = pose[:, 3] + torch.zeros_like(cam_ray_dirs)
            world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
            world_rays = world_rays.reshape(self.img_wh[1], self.img_wh[0], 6)
            self.all_rays.append(world_rays)
        
        self.all_rays = torch.stack(self.all_rays)

    def render_frames(self, get_outputs, output_dir):
        with torch.no_grad():
            for i, rays in enumerate(tqdm(self.all_rays)):
                rays = rays.to(self.device)
                output = get_outputs(rays)
                uncertainty = output[..., -1].detach().cpu().numpy()
                uncertainty_img = np.uint8(np.clip(uncertainty, 0, 1) * 255)
                
                # Save the image
                Image.fromarray(uncertainty_img, mode="L").save(f"{output_dir}/video360/frame_{i:04d}.png")
                
                del rays
                torch.cuda.empty_cache()
        self._generate_mp4(output_dir)
    
    def _generate_mp4(self, output_dir, fps=30):
        frame_dir = os.path.join(output_dir, "video360")
        output_video = os.path.join(output_dir, "video360.mp4")
        
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        
        if not frame_files:
            print("No frames found to create video.")
            return
        
        first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
        height, width, _ = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        for frame_file in tqdm(frame_files):
            frame_path = os.path.join(frame_dir, frame_file)
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to {output_video}")
    


   