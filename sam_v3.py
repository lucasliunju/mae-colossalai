

import torch
import numpy as np
import wandb


class Look_layer_sam(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, k=5, pha=1.0, alpha=0.7, adaptive=False, is_first_rank=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(Look_layer_sam, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.pha = pha
        self.eps = 1e-14

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
        # self.clip_grad()
        # print('doing first step')
        grad_norm = self._grad_norm()
        # scale_viz = []
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + self.eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["grad_origin"] = p.grad.clone()

                # scale = group["rho"]/(p.grad.norm(2) + self.eps)

                lamb_scale = p.data.norm(2)/(p.grad.norm(2) + self.eps)
                # if np.random.random(1) < 0.01:
                #     print('lamb scale ', lamb_scale.cpu(), p.ndim)
                lamb_scale = lamb_scale.clamp_max(20)
                # lamb_scale = 1
                iscale = (torch.pow(p, 2)
                          if group["adaptive"] else 1.0) * self.pha * scale * lamb_scale
                # scale_viz.append(iscale)
                e_w = iscale * p.grad.to(p)
                # e_w = self.pha * lamb_scale * p.grad.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        # scale_viz = torch.Tensor(scale_viz)
        # if is_first_rank:
            # wandb.log({'scale_hist': wandb.Histogram(scale_viz)})
        self.zero_grad()
        # print('finish first step')

    @torch.no_grad()
    def sam_second_step(self, ga):
        # print('doing second step')
        self.divide_grad_by_ga(ga)
        # self.clip_grad()
        grad_norm = self._grad_norm()
        grad_norm_factor = 1.0 / grad_norm

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

                # grad_v = grad_sam - grad_origin
                #
                cos_g_gs = grad_origin * grad_sam / \
                            (grad_origin.norm(2) * grad_sam.norm(2) + self.eps)


                # grad_v = grad_sam -  grad_sam * grad_origin * grad_origin  / (grad_origin.norm(2) * grad_origin.norm(2) + self.eps )
                grad_v = grad_sam - grad_origin * cos_g_gs * grad_sam.norm(2) / (grad_origin.norm(2) + self.eps)


                cos_gv_g = grad_origin * grad_v / (grad_origin.norm(2) * grad_v.norm(2) + self.eps)
                cos_gv_gs = grad_origin * grad_sam / (grad_origin.norm(2) * grad_sam.norm(2) + self.eps)

                grad_v = grad_v * cos_gv_g

                grad_v = grad_sam - grad_v





                self.state[p]["grad_v"] = grad_v
                # p.grad = p.grad * grad_norm_factor.clamp_max(1.0)
        self.base_optimizer.step()
        self.zero_grad()
        # print('finish second step')

    @torch.no_grad()
    def ordinary_step(self, ga):
        # print('doing ord step')
        self.divide_grad_by_ga(ga)
        # self.clip_grad()
        grad_norm = self._grad_norm()
        grad_norm_factor = 1.0 / grad_norm
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_sam = p.grad + \
                    (self.alpha * p.grad.norm(2) /
                        (self.state[p]["grad_v"].norm(2) )) * self.state[p]["grad_v"]
                p.grad = grad_sam
                # p.grad = p.grad

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        self.zero_grad()
        # print('finish ord step')

    @torch.no_grad()
    def step(self, step_num, ga=1):
        # ga for gradient accumulation number
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
