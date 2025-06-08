import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

class Uncertainty360Viewer:
    def __init__(self, data_handler, get_outputs):
        self.data_handler = data_handler
        self.get_outputs = get_outputs
        self.device = data_handler.device

    def render_360_video(self, output_dir, num_frames=120, resolution=None):
        os.makedirs(output_dir, exist_ok=True)
    
        center = self.data_handler.viewer_pos.to(self.device)
        #print(center)
        #exit(10)
        #target_center = torch.tensor([0, 0, 0], device=self.device)
        #center = target_center
        up = self.data_handler.viewer_up.to(self.device)
        original_forward = self.data_handler.viewer_forward.to(self.device)
        
        if resolution is None:
            img_wh = self.data_handler.img_wh
            H, W = img_wh[1], img_wh[0]  # 2078 height, 3118 width
        else:
            W, H = resolution

        
        fx = self.data_handler.fx
        fy = self.data_handler.fy
        
        if resolution is not None and resolution != self.data_handler.img_wh:
            scale_w = W / self.data_handler.img_wh[0]
            scale_h = H / self.data_handler.img_wh[1]
            fx = fx * scale_w
            fy = fy * scale_h
        
        radius = torch.norm(original_forward).item() * 1.5
        thetas = torch.linspace(0, 2 * torch.pi, num_frames, device=self.device)
        
        for i in range(num_frames):
            print(f"\nRendering frame {i+1}/{num_frames}")

            forward_base = original_forward / original_forward.norm()
            right = torch.cross(up, forward_base)
            right = right / right.norm()

            theta = thetas[i]
            eye = center + radius * (
                torch.cos(theta) * right + torch.sin(theta) * forward_base
            )

            view_dir = center - eye
            forward = view_dir / view_dir.norm()

            right = torch.cross(forward, up)
            right = right / right.norm()
            new_up = torch.cross(right, forward)
            
            c2w = torch.eye(4, device=self.device)
            c2w[:3, 0] = right
            c2w[:3, 1] = new_up
            c2w[:3, 2] = forward
            c2w[:3, 3] = eye
            
            ray_batch  = self.generate_rays_for_frame(c2w, H, W, fx, fy)
            print(f"Rendering frame {i+1}/{num_frames}")
            print(f"Ray batch shape: {ray_batch.shape}")
            ray_batch = ray_batch.to(self.device)
            output = self.get_outputs(ray_batch)
            uncertainty = output[..., -1:].detach().cpu().numpy().squeeze(-1) 
            uncertainty_img = np.uint8(np.clip(uncertainty, 0, 1) * 255)
            print(f"Saving frame {i:04d} with shape {uncertainty_img.shape}")
            Image.fromarray(uncertainty_img, mode="L").save(f"{output_dir}/video360/frame_{i:04d}.png")
            del ray_batch
            torch.cuda.empty_cache()
        print(f"Finished rendering {num_frames} frames to {output_dir}/video360")
        self._generate_mp4(output_dir)



    def generate_rays_for_frame(self, c2w, H, W, fx, fy):
        """Generate rays for a given camera pose."""
        device = c2w.device
        
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), 
                            torch.linspace(0, H-1, H), 
                            indexing='xy')
        i = i.t().to(self.device)
        j = j.t().to(self.device)

        dirs = torch.stack([(i - W * 0.5) / fx, 
                        -(j - H * 0.5) / fy, 
                        -torch.ones_like(i, device=device)], dim=-1)
        
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        
        return torch.cat([rays_o, rays_d], dim=-1)

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