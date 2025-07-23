import torch
import torch.nn.functional as F
from radfoam_model.render import TraceRays

class HessianAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ray_batch, model, component_idx):
        with torch.enable_grad():
            primal_points = model.primal_points.clone().detach()
            offsets = torch.ones_like(primal_points) * 0.01 * torch.randn_like(primal_points)
            offsets.requires_grad_()
            deformed_points = primal_points + offsets

            TraceRays.RETAIN_GRAPH = True
            rgba_output, _, _, _, _ = model(ray_batch, primal_points=deformed_points)
            rgb = rgba_output.view(-1, 3)
            colors = torch.sum(rgb, dim=0)

            grad_output = torch.zeros_like(colors)
            grad_output[component_idx] = 1.0

            grad_from_autograd = torch.autograd.grad(
                colors,
                offsets,
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]

            ctx.save_for_backward(offsets, grad_from_autograd)
            ctx.model = model
            ctx.component_idx = component_idx

            return grad_from_autograd

    @staticmethod
    def backward(ctx, grad_output):
        offsets, grad_from_autograd = ctx.saved_tensors
        model = ctx.model
        ray_batch = ctx.ray_batch
        component_idx = ctx.component_idx

        epsilon = 1e-5
        v = grad_output
        
         
        with torch.enable_grad():
            rgba_original, _, _, _, _ = model(ray_batch, primal_points=primal_points)
            rgba_perturbed, _, _, _, _ = model(ray_batch, primal_points=primal_points + epsilon * v)
            jv = (rgba_perturbed - rgba_original) / epsilon
            
            primal_points_grad = primal_points.clone().requires_grad_(True)
            rgba_output, _, _, _, _ = model(ray_batch, primal_points=primal_points_grad)
        
            jtjv = torch.autograd.grad(
                    outputs=rgba_output,
                    inputs=primal_points_grad,
                    grad_outputs=jv,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True
                )[0]
        
        return None, jtjv, None, None