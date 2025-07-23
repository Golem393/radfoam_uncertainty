class JacobianTimesJacobianFunction(torch.autograd.Function):
    """
    This is the trick from https://j-towns.github.io/2017/06/12/A-new-trick.html
    """
    
    @staticmethod
    def forward(ctx, model, ray_batch, primal_points, epsilon=1e-4):
        ctx.model = model
        ctx.ray_batch = ray_batch
        ctx.primal_points = primal_points
        ctx.epsilon = epsilon
        
        with torch.no_grad():
            rgba_original, _, _, _, _ = model(ray_batch, primal_points=primal_points)
            
            rgba_perturbed, _, _, _, _ = self.model(ray_batch, primal_points=primal_points + self.epsilon * v)
            
            jv = (rgba_perturbed - rgba_original) / self.epsilon
            
        primal_points_grad = primal_points.clone().detach().requires_grad_(True)
        
        TraceRays.RETAIN_GRAPH = True
        rgba_output, _, _, _, _ = self.model(ray_batch, primal_points=primal_points_grad)
        
        jtjv = torch.autograd.grad(
            outputs=rgba_output,
            inputs=primal_points_grad,
            grad_outputs=jv,
            retain_graph=False,
            create_graph=True,
            only_inputs=True
        )[0]
        
        return jtjv
    
    @staticmethod
    def backward(ctx, grad_output):
        model = ctx.model
        ray_batch = ctx.ray_batch
        primal_points = ctx.primal_points
        epsilon = ctx.epsilon
    
        
        with torch.no_grad():
            rgba_original, _, _, _, _ = model(ray_batch, primal_points=primal_points)
            
            jacobian_cols = []
            for i in range(primal_points.numel()):
                perturbation = torch.zeros_like(primal_points.view(-1))
                perturbation[i] = epsilon
                perturbation = perturbation.view(primal_points.shape)
                
                rgba_perturbed, _, _, _, _ = model(ray_batch, primal_points=primal_points + perturbation)
                jacobian_col = (rgba_perturbed - rgba_original) / epsilon
                jacobian_cols.append(jacobian_col.view(-1))
            
            jacobian = torch.stack(jacobian_cols, dim=1)
            
            grad_trace = torch.trace(grad_output)
            jtj_diagonal = torch.sum(jacobian ** 2, dim=0)
            
            grad_wrt_points = grad_trace * jtj_diagonal
            grad_wrt_points = grad_wrt_points.view(primal_points.shape)
        
        return None, None, grad_wrt_points, None