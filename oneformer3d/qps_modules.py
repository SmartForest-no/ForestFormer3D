import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps

from .adaptive_fusion import SEGateFusion


class OffsetGuidedResidualDistance(nn.Module):
    """Learnable residual fusion: embed + alpha * proj(offset)."""

    def __init__(self, embed_dim=5, offset_dim=3, alpha_init=0.1, norm_eps=1e-6):
        super().__init__()
        self.proj = nn.Linear(offset_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.norm_eps = float(norm_eps)

    def forward(self, embed_feat, offset_feat):
        embed_norm = F.normalize(embed_feat, p=2, dim=-1, eps=self.norm_eps)
        offset_norm = F.normalize(offset_feat, p=2, dim=-1, eps=self.norm_eps)
        fused = embed_norm + self.alpha * self.proj(offset_norm)
        return F.normalize(fused, p=2, dim=-1, eps=self.norm_eps)


def build_centroid_offset_targets(voxel_coords, voxel_instance_labels, voxel_size=None):
    """Build per-voxel centroid offset targets.

    Args:
        voxel_coords (Tensor): shape (N, 3). Can be voxel indices or metric coords.
        voxel_instance_labels (Tensor): shape (N,).
        voxel_size (float | None): If provided, `voxel_coords` will be scaled by
            voxel_size to recover metric coordinates from voxel indices.

    Returns:
        Tuple[Tensor, Tensor]:
            - offset targets with shape (N, 3)
            - foreground mask with shape (N,)
    """
    voxel_coords = voxel_coords.float()
    if voxel_size is not None:
        voxel_coords = voxel_coords * float(voxel_size)

    centroid_offset_labels = torch.zeros_like(voxel_coords)
    unique_instances = torch.unique(voxel_instance_labels)
    for inst_id in unique_instances:
        if inst_id < 0:
            continue
        inst_mask = voxel_instance_labels == inst_id
        if inst_mask.any():
            inst_coords = voxel_coords[inst_mask]
            centroid = inst_coords.mean(dim=0)
            centroid_offset_labels[inst_mask] = centroid - inst_coords

    fg_mask = voxel_instance_labels >= 0
    return centroid_offset_labels, fg_mask


class QueryPointSelector(nn.Module):
    """Unified query-point selector for baseline / SE / TCFPS / OGRD."""

    def __init__(self,
                 query_point_num,
                 fps_mode='baseline',
                 se_reduction=2,
                 tcfps_embed_ratio=0.6,
                 ogrd_alpha=0.1):
        super().__init__()
        self.query_point_num = query_point_num
        self.fps_mode = fps_mode.lower()
        if self.fps_mode == 'embed':
            self.fps_mode = 'baseline'
        if self.fps_mode not in ('baseline', 'se', 'tcfps', 'ogrd'):
            raise ValueError(f'Unsupported fps_mode: {fps_mode}')
        self.tcfps_embed_ratio = float(tcfps_embed_ratio)
        self.ogrd_alpha_init = float(ogrd_alpha)
        if not (0.0 <= self.tcfps_embed_ratio <= 1.0):
            raise ValueError('tcfps_embed_ratio must be in [0, 1].')
        self.se_fusion = SEGateFusion(
            embed_dim=5, offset_dim=3, reduction=se_reduction)
        self.ogrd_fusion = OffsetGuidedResidualDistance(
            embed_dim=5, offset_dim=3, alpha_init=self.ogrd_alpha_init)

    def _fps_by_budget(self, feat, budget):
        if feat.numel() == 0:
            return feat.new_zeros((0,), dtype=torch.long)
        n_points = feat.size(0)
        if budget <= 0:
            return torch.zeros((0,), dtype=torch.long, device=feat.device)
        if n_points <= budget:
            return torch.arange(n_points, device=feat.device)
        ratio = min(float(budget) / float(n_points), 1.0)
        batch_tensor = torch.zeros(
            n_points, dtype=torch.long, device=feat.device)
        sampled = fps(feat, batch_tensor, ratio=ratio)
        if sampled.numel() > budget:
            sampled = sampled[:budget]
        return sampled

    def build_ogrd_features(self, embed_feats, offset_feats):
        return self.ogrd_fusion(embed_feats, offset_feats)

    def _select_local_indices_tcfps(self, embed_feats, offset_feats):
        num_candidates = embed_feats.size(0)
        if num_candidates <= self.query_point_num:
            return torch.arange(num_candidates, device=embed_feats.device)

        n_embed = int(round(self.query_point_num * self.tcfps_embed_ratio))
        n_embed = min(max(n_embed, 0), self.query_point_num)
        n_offset = self.query_point_num - n_embed

        merged = []
        if n_embed > 0:
            embed_norm = F.normalize(embed_feats, p=2, dim=-1)
            merged.append(self._fps_by_budget(embed_norm, n_embed))
        if n_offset > 0:
            offset_norm = F.normalize(offset_feats, p=2, dim=-1)
            merged.append(self._fps_by_budget(offset_norm, n_offset))

        if not merged:
            return torch.arange(
                min(self.query_point_num, num_candidates),
                device=embed_feats.device)

        merged_indices = torch.cat(merged, dim=0)
        seen = torch.zeros(
            num_candidates, dtype=torch.bool, device=embed_feats.device)
        keep = []
        for idx in merged_indices:
            idx_int = int(idx.item())
            if not seen[idx_int]:
                seen[idx_int] = True
                keep.append(idx_int)
            if len(keep) >= self.query_point_num:
                break

        if len(keep) < self.query_point_num:
            remain = torch.where(~seen)[0]
            need = min(self.query_point_num - len(keep), int(remain.numel()))
            if need > 0:
                keep.extend(remain[:need].tolist())

        return torch.tensor(
            keep[:self.query_point_num],
            dtype=torch.long,
            device=embed_feats.device)

    def select_indices(self, candidate_indices, embed_logits, offset_logits):
        if candidate_indices.numel() == 0:
            return candidate_indices
        if candidate_indices.numel() == 1:
            return candidate_indices

        embed_feats = embed_logits[candidate_indices]
        offset_feats = offset_logits[candidate_indices]

        if self.fps_mode == 'tcfps':
            local_indices = self._select_local_indices_tcfps(
                embed_feats, offset_feats)
            return candidate_indices[local_indices]

        if self.fps_mode == 'baseline':
            fps_input = F.normalize(embed_feats, p=2, dim=-1)
        elif self.fps_mode == 'se':
            fps_input = self.se_fusion(embed_feats, offset_feats)
        else:
            fps_input = self.build_ogrd_features(embed_feats, offset_feats)
        local_indices = self._fps_by_budget(fps_input, self.query_point_num)
        return candidate_indices[local_indices]
