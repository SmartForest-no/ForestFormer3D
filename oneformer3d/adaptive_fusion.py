import torch
import torch.nn as nn
import torch.nn.functional as F


class SEGateFusion(nn.Module):
    """Squeeze-and-Excitation fusion for Embed/Offset FPS inputs.

    Input:
        embed_feat:  (N, embed_dim)
        offset_feat: (N, offset_dim)
    Output:
        fused_feat:  (N, embed_dim + offset_dim)
    """

    def __init__(self, embed_dim=5, offset_dim=3, reduction=2, norm_eps=1e-6):
        super().__init__()
        total_dim = embed_dim + offset_dim
        hidden_dim = max(total_dim // reduction, 4)

        self.norm_eps = norm_eps
        self.se = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_dim),
            nn.Sigmoid())

    def forward(self, embed_feat, offset_feat):
        embed_norm = F.normalize(embed_feat, p=2, dim=-1, eps=self.norm_eps)
        offset_norm = F.normalize(offset_feat, p=2, dim=-1, eps=self.norm_eps)
        concat_feat = torch.cat([embed_norm, offset_norm], dim=-1)
        channel_weights = self.se(concat_feat.mean(dim=0, keepdim=True))
        return concat_feat * channel_weights
