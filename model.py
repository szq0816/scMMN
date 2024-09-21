import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class my_model(nn.Module):
    def __init__(self, dims):
        super(my_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])
        self.mu = Parameter(torch.Tensor(dims[0], dims[1]))
        self.alpha = 1.0

    def forward(self, x, is_train=True, sigma=0.01):
        out1 = self.layers1(x)
        out2 = self.layers2(x)

        out1 = F.normalize(out1, dim=1, p=2)

        if is_train:
            out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cuda()
        else:
            out2 = F.normalize(out2, dim=1, p=2)
        return out1

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()

        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()



