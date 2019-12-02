import torch
import torch.nn as nn
import numpy as np

from transforms import *

class PlanarFlow(nn.Module):
    def __init__(self, dim=20, K=16):
        super().__init__()
        self.transforms = nn.ModuleList([PlanarTransform(dim) for k in range(K)])
    def forward(self, z, logdet=False):
        zK = z
        SLDJ = 0.
        for transform in self.transforms:
            out = transform(zK, logdet=logdet)
            if logdet:
                SLDJ += out[1]
                zK = out[0]
            else:
                zK = out
                
        if logdet:
            return zK, SLDJ
        return zK

if __name__ == '__main__':
    planar = PlanarFlow(dim=5, K=4)
    print([p.size() for p in planar.parameters()])
    planar.cuda()
    z0 = torch.randn(3, 5).cuda()
    z0 = z0*4
    print(z0)
    print(planar(z0, True))