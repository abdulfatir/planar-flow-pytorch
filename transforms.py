import torch
import torch.nn as nn
import numpy as np

class PlanarTransform(nn.Module):
    def __init__(self, dim=20):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.randn(()) * 0.01)
    def m(self, x):
        return -1 + torch.log(1 + torch.exp(x))
    def h(self, x):
        return torch.tanh(x)
    def h_prime(self, x):
        return 1 - torch.tanh(x) ** 2
    def forward(self, z, logdet=False):
        # z.size() = batch x dim
        u_dot_w = (self.u @ self.w.t()).view(())
        w_hat = self.w / torch.norm(self.w, p=2) # Unit vector in the direction of w
        u_hat = (self.m(u_dot_w) - u_dot_w) * (w_hat) + self.u # 1 x dim
        affine = z @ self.w.t() + self.b
        z_next = z + u_hat * self.h(affine) # batch x dim
        if logdet:
            psi = self.h_prime(affine) * self.w # batch x dim
            LDJ = -torch.log(torch.abs(psi @ u_hat.t() + 1) + 1e-8) # batch x 1
            return z_next, LDJ
        return z_next

if __name__ == '__main__':
    planar = PlanarTransform(dim=5)
    print([p.size() for p in planar.parameters()])
    planar.cuda()
    print([p.device for p in planar.parameters()])
    print([p.requires_grad for p in planar.parameters()])
    z0 = torch.randn(3, 5).cuda()
    print(planar(z0*4, True))