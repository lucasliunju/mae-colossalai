

import torch
import wandb
import numpy as np
# for input, target in dataset:
#     def closure():
#         optimizer.zero_grad()
#         output = model(input)
#         loss = loss_fn(output, target)
#         loss.backward()
#         return loss
#     optimizer.step(closure)


class Look_sam(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=2, k=5, pha=1, alpha=1, adaptive=False, is_first_rank=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(Look_sam, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.pha = pha
        self.eps = 1e-6
        self.is_first_rank = is_first_rank

    @torch.no_grad()
    def divide_grad_by_ga(self, ga):
        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad = p.grad / ga

    @torch.no_grad()
    def clip_grad(self):
        grad_norm = self._grad_norm()
        grad_norm_factor = 1.0 / grad_norm
        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad = p.grad * grad_norm_factor.clamp_max(1.0)

    @torch.no_grad()
    def sam_first_step(self, ga):

        self.divide_grad_by_ga(ga)
        self.clip_grad()

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + self.eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["grad_origin"] = p.grad.clone().detach()
                e_w = (torch.pow(p, 2)
                        if group["adaptive"] else 1.0) * p.grad * self.pha * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        self.zero_grad()

    @torch.no_grad()
    def sam_second_step(self, ga):
        self.divide_grad_by_ga(ga)
        self.clip_grad()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get back to "w" from "w + e(w)"
                p.data = self.state[p]["old_p"]

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_sam = p.grad
                grad_origin = self.state[p]["grad_origin"]
                # get back to "w" from "w + e(w)"
                p.data = self.state[p]["old_p"]
                cos = grad_sam * grad_origin / \
                    (grad_sam.norm(2) * grad_origin.norm(2) + self.eps)
                grad_v = grad_sam - \
                    grad_sam.norm(2) * cos * grad_origin / \
                    (grad_origin.norm(2) + self.eps)
                # grad_v = grad_sam - grad_origin
                
                    
                
                if self.is_first_rank and  np.random.random(1)<0.05:
                    if 'grad_v' in self.state[p]:
                        wandb.log({'gv_hist': wandb.Histogram(grad_v.clone().cpu())})
                    
                        old_gv = self.state[p]['grad_v'].clone()
                        new_gv = grad_v.clone()
                        # simi = torch.mean(old_gv*new_gv/(old_gv.norm(2) * new_gv.norm(2) + self.eps))
                        simi = torch.mean(torch.abs(new_gv-old_gv)/torch.abs(old_gv+self.eps))
                        wandb.log({'gv_simi': simi.cpu()})
                self.state[p]["grad_v"] = grad_v
                # p.grad = p.grad * grad_norm_factor.clamp_max(1.0)
        self.base_optimizer.step(if_norm=False, ga=1)
        self.zero_grad()

    @torch.no_grad()
    def ordinary_step(self, ga):
        self.divide_grad_by_ga(ga)
        self.clip_grad()
        # grad_norm = self._grad_norm()
        # grad_norm_factor = 1.0 / grad_norm
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.grad = p.grad + \
                    (self.alpha * p.grad.norm(2) /
                        (self.state[p]["grad_v"].norm(2) + self.eps)) * self.state[p]["grad_v"]
                
                # p.grad = p.grad * grad_norm_factor.clamp_max(1.0)
        self.base_optimizer.step(if_norm=False, ga=1)  # do the actual "sharpness-aware" update
        self.zero_grad()

    @torch.no_grad()
    def step(self, step_num, ga):
        if step_num == 1:
            self.sam_first_step(ga)
        if step_num == 2:
            self.sam_second_step(ga)
        if step_num == 3:
            self.ordinary_step(ga)

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0)
                         * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
