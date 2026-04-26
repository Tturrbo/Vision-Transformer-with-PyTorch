import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.trunc_normal_(self.conv2d.weight, std=0.02)
        nn.init.zeros_(self.conv2d.bias)

    def forward(self, X):
        X = self.conv2d(X)
        X = X.flatten(start_dim=2)
        return X.transpose(1,2)