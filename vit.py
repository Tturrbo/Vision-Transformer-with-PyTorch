import torch
import torch.nn as nn
from patchemb import PatchEmbedding

class ViT(nn.Module):
    def __init__(self, image_size=224, embed_dim=768, in_channels=3, patch_size=16,
                 dropout=0.1, num_heads=12, ff_dim=3072, depth=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        cls_init = torch.randn(1,1,embed_dim) ** 0.02
        self.cls_token = nn.Parameter(cls_init)
        num_patches = (image_size // patch_size) ** 2
        pos_init = torch.randn(1,1+num_patches,embed_dim) * 0.02
        self.pos_embed = nn.Parameter(pos_init)
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                        dropout=dropout, dim_feedforward=ff_dim, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, X):
        Z = self.patch_embed(X)
        cls_expd = self.cls_token.expand(Z.shape[0], -1, -1)
        Z = torch.cat((cls_expd, Z), dim=1)
        Z = Z + self.pos_embed
        Z = self.dropout(Z)
        Z = self.encoder(Z)
        Z = self.layer_norm(Z[:, 0])
        logits = self.output(Z)
        return logits


