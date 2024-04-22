from torch import nn
import torch


class bigModelQueen(nn.Module):

    def __init__(self, dim, gpu, dropout, task, max_feat):
        super().__init__()
        self.dim = dim
        if task == "regression":
            out_dim = 1
        else:
            out_dim = max_feat

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Linear(self.dim * 4, out_dim)
        )

    def forward(self, x):
        # print(self.out[0].weight)
        return self.mlp(x)
        # mlp_embedding = self.act(self.mlp(x))
        # return self.out(mlp_embedding)
