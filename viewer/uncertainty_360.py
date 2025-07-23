import math
from scipy.spatial.transform import Rotation as R

import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import pycolmap
import cv2
from data_loader import DataHandler # Assuming data_loader.py is in the same directory or path


def generate_circular_camera_poses(num_frames=120, radius=2.0, elevation=-0.8, 
                                 target_x=0, target_y=0, target_z=0, # New: target coordinates
                                 pitch_deg=0, yaw_deg=0, roll_deg=0,
                                 start_angle_deg=0): # New: starting angle for orbit
    """
    Generate camera poses in a circular path around a target point.
    
    Args:
        num_frames: Number of frames in the video
        radius: Radius of the circular path
        elevation: Height of the camera above the target point (relative to target_y)
        target_x, target_y, target_z: Coordinates of the point the camera should look at.
        pitch_deg, yaw_deg, roll_deg: Optional additional rotations for the camera (fixed offset).
        start_angle_deg: The initial angle (in degrees) of the camera on the circular path.
                         0 degrees is typically in front (-Z axis in a standard setup).
                         90 degrees (+X axis, "left" side), 180 degrees (+Z axis, "back"), etc.
        
    Returns:
        List of 3x4 camera-to-world matrices (rotation + translation)
    """
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)
    
    start_angle = math.radians(start_angle_deg) # Convert start_angle to radians

    poses = []
    
    # Define the point the camera will always look at
    look_at_point = np.array([target_x, target_y, target_z])

    for i in range(num_frames):
        # Calculate the current angle for this frame
        # Ensures a full 360-degree rotation from the start_angle
        angle = start_angle + (2 * math.pi * i / num_frames) 
        
        # Calculate camera position relative to the target
        x_offset = radius * math.cos(angle)
        z_offset = radius * math.sin(angle)
        # Y-offset is directly the elevation relative to the target_y
        y_offset = elevation

        # Add the target coordinates to the offsets to get the camera's world position
        position = np.array([
            look_at_point[0] + x_offset,
            look_at_point[1] + y_offset, 
            look_at_point[2] + z_offset
        ])
        
        # Calculate camera orientation (looks at the defined look_at_point)
        forward = look_at_point - position
        forward = forward / np.linalg.norm(forward)

        # Apply additional rotation offsets (pitch, yaw, roll)
        if yaw_deg != 0 or pitch_deg != 0 or roll_deg != 0:
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
            
            R_combined = Rz @ Rx @ Ry
            forward = R_combined @ forward


        # --- Robust Camera Orientation (Right and Up vectors) ---
        # Define a consistent 'world up' vector. Assuming Y is up in your world.
        world_up_vector = np.array([0, 1, 0])

        # Calculate the camera's 'right' vector. Cross product of forward and world_up.
        # This prevents skew/roll issues by ensuring 'right' is orthogonal to 'forward'
        # and as much as possible aligned with the horizontal plane.
        right = np.cross(forward, world_up_vector)
        # Handle the edge case where forward is perfectly aligned with world_up (looking straight down/up)
        # In this case, cross product would be zero, leading to NaNs.
        # Pick an arbitrary 'right' if forward is perfectly up/down, e.g., world's X-axis
        if np.linalg.norm(right) < 1e-6: # Check if cross product is close to zero
            right = np.array([1, 0, 0]) # Fallback: use world X-axis as right
            if np.dot(forward, right) > 0.9: # If forward is also close to X, pick Y
                 right = np.array([0, 1, 0])
        right = right / np.linalg.norm(right)

        # Calculate the camera's 'up' vector. Cross product of right and forward.
        # This ensures 'up' is perfectly orthogonal to both 'right' and 'forward'.
        up = np.cross(right, forward) 
        up = up / np.linalg.norm(up)
        
        # Build the 3x4 matrix (rotation + translation)
        c2w = np.eye(4)
        # Standard NeRF/COLMAP c2w convention:
        # c2w[:3, 0] = camera's X-axis (right)
        # c2w[:3, 1] = camera's Y-axis (up)
        # c2w[:3, 2] = camera's Z-axis (forward, pointing into the scene)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward 
        c2w[:3, 3] = position
        
        # Convert to 3x4 and return (dropping the last row of the 4x4 matrix)
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
    def __init__(self, device, dataset_args, additional_params, filter_out_in_image=False, filter_out_cells = False, filter_thresh=0.5):

        self.filter_out_in_image = filter_out_in_image
        self.filter_thresh = filter_thresh
        self.filter_out_cells = filter_out_cells
        self.device = device
        data_dir = "/mnt/hdd/team4/data/mipnerf360/bonsai" # This path is hardcoded, ensure it's correct
        self.colmap_dir = os.path.join(data_dir, "sparse/0/")
        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read(self.colmap_dir)
        self.dataset_args = dataset_args

        self.camera = list(self.reconstruction.cameras.values())[0]
        # It's best to use the dimensions directly from the COLMAP camera
        self.img_wh = [self.camera.width, self.camera.height] 
        self.camera.rescale(self.img_wh[0], self.img_wh[1]) # Rescale if the viewer resolution differs from COLMAP's original

        self.fx = self.camera.focal_length_x
        self.fy = self.camera.focal_length_y

        cam_ray_dirs = get_cam_ray_dirs(self.camera)

        # --- UPDATED CAMERA POSE GENERATION ---
        # Using the estimated scene center from your previous output: [-0.91107213, 2.506662, 3.7507117]
        if additional_params is None or not additional_params.get("radius", False):
            additional_params = {
                "radius": 4.2700000000000005, # Increased radius for more distance
                "elevation": -5.01, # Increased elevation for a more top-down view
                "target_x_offset": 0.47,
                "target_y_offset": 0.067,
                "target_z_offset": -0.609,
                "pitch_deg": -28.0,
                "yaw_deg": 0,
                "roll_deg": -7.7,
                "start_angle_deg": 10 # Set to 0 degrees to start from "front" (along -Z axis)
                                # Adjust this (e.g., 90, 180, 270) to change the initial side
            }
            self.poses = generate_circular_camera_poses(
                num_frames=120,
                radius=additional_params["radius"], # Increased radius for more distance
                elevation=additional_params["elevation"], # Increased elevation for a more top-down view
                target_x=-0.91107213 + additional_params["target_x_offset"],
                target_y=2.506662 + additional_params["target_y_offset"],
                target_z=3.7507117 + additional_params["target_z_offset"],
                pitch_deg=additional_params["pitch_deg"],
                yaw_deg=additional_params["yaw_deg"],
                roll_deg= additional_params["roll_deg"],
                start_angle_deg=additional_params["start_angle_deg"] # Set to 0 degrees to start from "front" (along -Z axis)
                                # Adjust this (e.g., 90, 180, 270) to change the initial side
            )
        else:
            self.poses = generate_circular_camera_poses(
                num_frames=120,
                radius=additional_params["radius"], # Increased radius for more distance
                elevation=additional_params["elevation"], # Increased elevation for a more top-down view
                target_x=-0.91107213 + additional_params["target_x_offset"],
                target_y=2.506662 + additional_params["target_y_offset"],
                target_z=3.7507117 + additional_params["target_z_offset"],
                pitch_deg=additional_params["pitch_deg"],
                yaw_deg=additional_params["yaw_deg"],
                roll_deg= additional_params["roll_deg"],
                start_angle_deg=additional_params["start_angle_deg"] # Set to 0 degrees to start from "front" (along -Z axis)
                                # Adjust this (e.g., 90, 180, 270) to change the initial side
            )
        # --- END UPDATED CAMERA POSE GENERATION ---

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
        """train_data_handler = DataHandler(
            self.dataset_args, rays_per_batch=750_000, device=self.device
        )
        iter2downsample = dict(
            zip(
                self.dataset_args.downsample_iterations,
                self.dataset_args.downsample,
            )
        )
        downsample = iter2downsample[0]
        train_data_handler.reload(split="train", downsample=downsample)"""
        with torch.no_grad():
            # Create the output directory if it doesn't exist
            os.makedirs(f"{output_dir}/video360", exist_ok=True)

            for i, rays in enumerate(tqdm(self.all_rays)):
                rays = rays.to(self.device)
                output = get_outputs(rays)
                
                # Check output dimensions
                # Assuming output is (H, W, C) where C is typically 3 (RGB) or 4 (RGB + Uncertainty)
                if output.shape[-1] == 4: # Assuming last channel is uncertainty
                    uncertainty = output[..., -1].detach().cpu().numpy()
                    rgb_img_data = output[..., :3].detach().cpu().numpy()
                elif output.shape[-1] == 3: # Only RGB
                    uncertainty = np.zeros_like(output[..., 0]).detach().cpu().numpy() # No uncertainty channel
                    rgb_img_data = output[..., :3].detach().cpu().numpy()
                else:
                    raise ValueError(f"Unexpected output channel count: {output.shape[-1]}")


                uncertainty_img = np.uint8(np.clip(uncertainty, 0, 1) * 255)
                rgb_img = np.uint8(np.clip(rgb_img_data, 0, 1) * 255) # RGB output

                if self.filter_out_in_image or self.filter_out_cells:
                    mask = uncertainty > self.filter_thresh
                    rgb_img[mask] = 255 # Apply mask to RGB image for visualization
                    uncertainty_img[mask] = 255
                    # Save RGB with filtered uncertainty
                    Image.fromarray(uncertainty_img, mode="L").save(f"{output_dir}/video360/frame_{i:04d}.png")
                    #Image.fromarray(rgb_img, mode="RGB").save(f"{output_dir}/video360/frame_{i:04d}.png")
                else:
                    # Save Uncertainty Image (grayscale)
                    Image.fromarray(uncertainty_img, mode="L").save(f"{output_dir}/video360/frame_{i:04d}.png")
                
                del rays
                torch.cuda.empty_cache()
        self._generate_mp4(output_dir)
    
    def _generate_mp4(self, output_dir, fps=30):
        frame_dir = os.path.join(output_dir, "video360")
        index = 0
        while True:
            output_video = os.path.join(output_dir, f"video360_{index}.mp4")
            if not os.path.exists(output_video):
                break
            index += 1
        
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        
        if not frame_files:
            print("No frames found to create video.")
            return
        
        first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
        height, width, _ = first_frame.shape
        
        # Use 'avc1' for H.264 compatible with more players.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Alternative: 'mp4v' for broader compatibility
                                                 # Or 'XVID' if mp4v fails and you need AVI
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_video}")
            print("Check if OpenCV codecs are properly installed (e.g., 'libx264' for H.264).")
            return

        for frame_file in tqdm(frame_files):
            frame_path = os.path.join(frame_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame {frame_file}. Skipping.")
                continue
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to {output_video}")