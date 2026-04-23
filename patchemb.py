import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super().__init__()
        self.conv2d = nn.Conv2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        X = self.conv2d(X)
        X = X.flatten(start_dim=2)
        return X.transpose(1,2)