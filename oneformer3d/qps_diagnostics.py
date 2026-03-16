import json
import os
from statistics import mean, median

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean


def _safe_ratio(numerator, denominator):
    denominator = int(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _to_tensor(value, device, dtype=torch.long):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def aggregate_majority_labels(point_labels, voxel_indices):
    """Aggregate per-point labels to per-voxel labels by majority vote."""
    if point_labels.numel() == 0:
        return point_labels.new_empty((0,), dtype=torch.long)

    point_labels = point_labels.long().view(-1)
    voxel_indices = voxel_indices.long().view(-1)
    unique_labels, inverse_labels = torch.unique(
        point_labels, sorted=True, return_inverse=True)
    one_hot = F.one_hot(
        inverse_labels, num_classes=unique_labels.numel()).float()
    label_counts = scatter_add(one_hot, voxel_indices, dim=0)
    majority_indices = label_counts.argmax(dim=1)
    return unique_labels[majority_indices]


def build_scene_small_tree_profile(scene_points_xyz,
                                   scene_semantic_labels,
                                   scene_instance_labels,
                                   small_tree_ratio=1.0 / 3.0):
    """Build scene-level small-tree ids using tree height threshold."""
    scene_points_xyz = torch.as_tensor(scene_points_xyz)[..., :3].float()
    scene_semantic_labels = torch.as_tensor(scene_semantic_labels).long().view(-1)
    scene_instance_labels = torch.as_tensor(scene_instance_labels).long().view(-1)

    valid_mask = (scene_semantic_labels > 0) & (scene_instance_labels >= 0)
    if not valid_mask.any():
        return {
            'small_tree_ratio': float(small_tree_ratio),
            'max_tree_height': 0.0,
            'small_tree_height_threshold': 0.0,
            'small_instance_ids': torch.empty((0,), dtype=torch.long),
            'instance_heights': {},
        }

    instance_ids = torch.unique(scene_instance_labels[valid_mask], sorted=True)
    instance_heights = {}
    max_tree_height = 0.0
    for inst_id in instance_ids.tolist():
        inst_mask = valid_mask & (scene_instance_labels == int(inst_id))
        z_values = scene_points_xyz[inst_mask, 2]
        if z_values.numel() == 0:
            continue
        tree_height = float((z_values.max() - z_values.min()).item())
        instance_heights[int(inst_id)] = tree_height
        max_tree_height = max(max_tree_height, tree_height)

    height_threshold = float(max_tree_height) * float(small_tree_ratio)
    small_ids = sorted(
        inst_id for inst_id, tree_height in instance_heights.items()
        if tree_height <= height_threshold)
    return {
        'small_tree_ratio': float(small_tree_ratio),
        'max_tree_height': float(max_tree_height),
        'small_tree_height_threshold': float(height_threshold),
        'small_instance_ids': torch.tensor(small_ids, dtype=torch.long),
        'instance_heights': instance_heights,
    }


def _build_instance_table(voxel_instance_labels, gt_tree_mask,
                          scene_small_instance_ids=None):
    valid_mask = gt_tree_mask & (voxel_instance_labels >= 0)
    if not valid_mask.any():
        empty = voxel_instance_labels.new_empty((0,), dtype=torch.long)
        return {
            'instance_ids': empty,
            'counts': empty,
            'small_ids': empty,
        }

    instance_ids, counts = torch.unique(
        voxel_instance_labels[valid_mask], sorted=True, return_counts=True)
    if scene_small_instance_ids is None:
        small_ids = instance_ids.new_empty((0,), dtype=torch.long)
    else:
        scene_small_instance_ids = torch.as_tensor(
            scene_small_instance_ids, device=instance_ids.device,
            dtype=torch.long)
        if scene_small_instance_ids.numel() == 0:
            small_ids = instance_ids.new_empty((0,), dtype=torch.long)
        else:
            small_ids = instance_ids[torch.isin(instance_ids, scene_small_instance_ids)]
    return {
        'instance_ids': instance_ids,
        'counts': counts.long(),
        'small_ids': small_ids,
    }


def _coverage_from_indices(indices, voxel_instance_labels, gt_tree_mask,
                           instance_ids, small_ids, max_missed_ids):
    if indices is None or indices.numel() == 0 or instance_ids.numel() == 0:
        covered = instance_ids.new_empty((0,), dtype=torch.long)
    else:
        valid = gt_tree_mask[indices] & (voxel_instance_labels[indices] >= 0)
        covered = torch.unique(
            voxel_instance_labels[indices][valid], sorted=True)

    if covered.numel() == 0:
        covered = instance_ids.new_empty((0,), dtype=torch.long)

    if covered.numel() > 0:
        covered_sorted = torch.sort(covered).values
    else:
        covered_sorted = covered

    covered_set = set(covered_sorted.tolist())
    missed = [int(v) for v in instance_ids.tolist() if int(v) not in covered_set]
    small_set = set(int(v) for v in small_ids.tolist())
    missed_small = [v for v in missed if v in small_set]

    return {
        'covered_ids': covered_sorted,
        'covered_instances': int(covered_sorted.numel()),
        'covered_small_instances': int(sum(
            1 for v in covered_sorted.tolist() if int(v) in small_set)),
        'instance_coverage': _safe_ratio(covered_sorted.numel(), instance_ids.numel()),
        'small_tree_coverage': _safe_ratio(
            sum(1 for v in covered_sorted.tolist() if int(v) in small_set),
            small_ids.numel()),
        'missed_instance_count': len(missed),
        'missed_small_instance_count': len(missed_small),
        'missed_instance_ids_topk': missed[:max_missed_ids],
        'missed_small_instance_ids_topk': missed_small[:max_missed_ids],
    }


def _query_allocation(indices, voxel_instance_labels, gt_tree_mask,
                      instance_ids, small_ids):
    if instance_ids.numel() == 0:
        return {
            'mean_queries_per_instance': 0.0,
            'std_queries_per_instance': 0.0,
            'min_queries_per_instance': 0,
            'max_queries_per_instance': 0,
            'zero_query_instances': 0,
            'zero_query_instance_ratio': 0.0,
            'zero_query_small_instances': 0,
            'zero_query_small_ratio': 0.0,
        }

    allocations = torch.zeros(
        instance_ids.numel(), device=instance_ids.device, dtype=torch.long)
    if indices is not None and indices.numel() > 0:
        valid = gt_tree_mask[indices] & (voxel_instance_labels[indices] >= 0)
        selected_ids = voxel_instance_labels[indices][valid]
        if selected_ids.numel() > 0:
            query_ids, counts = torch.unique(
                selected_ids, sorted=True, return_counts=True)
            pos = torch.searchsorted(instance_ids, query_ids)
            allocations[pos] = counts.long()

    alloc_float = allocations.float()
    zero_small = 0
    if small_ids.numel() > 0:
        pos_small = torch.searchsorted(instance_ids, small_ids)
        zero_small = int((allocations[pos_small] == 0).sum().item())

    return {
        'mean_queries_per_instance': float(alloc_float.mean().item()),
        'std_queries_per_instance': float(
            alloc_float.std(unbiased=False).item()),
        'min_queries_per_instance': int(allocations.min().item()),
        'max_queries_per_instance': int(allocations.max().item()),
        'zero_query_instances': int((allocations == 0).sum().item()),
        'zero_query_instance_ratio': _safe_ratio(
            int((allocations == 0).sum().item()), instance_ids.numel()),
        'zero_query_small_instances': zero_small,
        'zero_query_small_ratio': _safe_ratio(zero_small, small_ids.numel()),
    }


def _normalized_centroid_distance(indices, voxel_coords, voxel_instance_labels,
                                  gt_tree_mask, instance_ids):
    if indices is None or indices.numel() == 0 or instance_ids.numel() == 0:
        return {
            'mean_normalized_centroid_distance': 0.0,
            'max_normalized_centroid_distance': 0.0,
        }

    values = []
    for inst_id in instance_ids.tolist():
        inst_mask = gt_tree_mask & (voxel_instance_labels == int(inst_id))
        if not inst_mask.any():
            continue
        inst_coords = voxel_coords[inst_mask]
        centroid = inst_coords.mean(dim=0)
        rms_radius = torch.sqrt(
            ((inst_coords - centroid) ** 2).sum(dim=1).mean()).clamp_min(1e-6)

        query_mask = gt_tree_mask[indices] & (
            voxel_instance_labels[indices] == int(inst_id))
        if not query_mask.any():
            continue
        query_coords = voxel_coords[indices[query_mask]]
        normalized = torch.norm(query_coords - centroid, dim=1) / rms_radius
        values.extend(normalized.tolist())

    if not values:
        return {
            'mean_normalized_centroid_distance': 0.0,
            'max_normalized_centroid_distance': 0.0,
        }

    return {
        'mean_normalized_centroid_distance': float(mean(values)),
        'max_normalized_centroid_distance': float(max(values)),
    }


def _pairwise_distance_summary(indices, voxel_coords):
    if indices is None or indices.numel() < 2:
        return {
            'pairwise_distance_mean': 0.0,
            'pairwise_distance_min': 0.0,
            'pairwise_distance_max': 0.0,
        }

    coords = voxel_coords[indices].float()
    dist = torch.cdist(coords, coords)
    tri = torch.triu_indices(coords.shape[0], coords.shape[0], offset=1)
    values = dist[tri[0], tri[1]]
    return {
        'pairwise_distance_mean': float(values.mean().item()),
        'pairwise_distance_min': float(values.min().item()),
        'pairwise_distance_max': float(values.max().item()),
    }


def _height_band_profile(gt_tree_mask, voxel_coords, named_indices):
    gt_coords = voxel_coords[gt_tree_mask]
    if gt_coords.numel() == 0:
        empty = {
            'low': {'count': 0, 'ratio': 0.0},
            'mid': {'count': 0, 'ratio': 0.0},
            'high': {'count': 0, 'ratio': 0.0},
        }
        return {'gt_tree_voxels': empty, **{k: empty for k in named_indices}}

    gt_z = gt_coords[:, 2].float()
    q1 = torch.quantile(gt_z, 1.0 / 3.0)
    q2 = torch.quantile(gt_z, 2.0 / 3.0)

    def build_profile(z_values):
        if z_values.numel() == 0:
            return {
                'low': {'count': 0, 'ratio': 0.0},
                'mid': {'count': 0, 'ratio': 0.0},
                'high': {'count': 0, 'ratio': 0.0},
            }
        low = int((z_values <= q1).sum().item())
        mid = int(((z_values > q1) & (z_values <= q2)).sum().item())
        high = int((z_values > q2).sum().item())
        total = max(low + mid + high, 1)
        return {
            'low': {'count': low, 'ratio': _safe_ratio(low, total)},
            'mid': {'count': mid, 'ratio': _safe_ratio(mid, total)},
            'high': {'count': high, 'ratio': _safe_ratio(high, total)},
        }

    profile = {'gt_tree_voxels': build_profile(gt_z)}
    for name, indices in named_indices.items():
        if indices is None or indices.numel() == 0:
            selected_z = gt_z.new_empty((0,))
        else:
            valid = gt_tree_mask[indices]
            selected_z = voxel_coords[indices[valid], 2].float()
        profile[name] = build_profile(selected_z)
    return profile


def _summarize_scores(scores):
    if scores is None or len(scores) == 0:
        return {'min': 0.0, 'mean': 0.0, 'max': 0.0}
    if isinstance(scores, torch.Tensor):
        if scores.numel() == 0:
            return {'min': 0.0, 'mean': 0.0, 'max': 0.0}
        values = scores.float()
        return {
            'min': float(values.min().item()),
            'mean': float(values.mean().item()),
            'max': float(values.max().item()),
        }
    scores = [float(v) for v in scores]
    return {
        'min': float(min(scores)),
        'mean': float(mean(scores)),
        'max': float(max(scores)),
    }


def build_region_qps_diagnostics(
        region_idx,
        region_center_xy,
        fps_mode,
        query_budget,
        pc1_count,
        pc2_count,
        pc3_points,
        inverse_mapping,
        bi_tree_prob,
        tree_indices,
        selected_indices,
        decoder_retained_indices,
        external_retained_indices,
        decoder_scores,
        external_scores,
        gt_semantic_point_labels,
        gt_instance_point_labels,
        scene_small_instance_ids=None,
        small_tree_ratio=1.0 / 3.0,
        small_tree_height_threshold=0.0,
        scene_max_tree_height=0.0,
        candidate_prob_threshold=0.3,
        max_missed_ids=20):
    """Build a per-region diagnostic payload for QPS analysis."""
    device = pc3_points.device
    voxel_superpoints = torch.unique(inverse_mapping, return_inverse=True)[1]
    voxel_coords = scatter_mean(pc3_points.float(), voxel_superpoints, dim=0)
    num_voxels = int(voxel_coords.shape[0])

    tree_indices = tree_indices.long()
    selected_indices = selected_indices.long()
    decoder_retained_indices = decoder_retained_indices.long()
    external_retained_indices = external_retained_indices.long()

    region_payload = {
        'region_index': int(region_idx),
        'region_center_xy': [float(region_center_xy[0]), float(region_center_xy[1])],
        'fps_mode': fps_mode,
        'query_budget': int(query_budget),
        'candidate_prob_threshold': float(candidate_prob_threshold),
        'small_tree_ratio': float(small_tree_ratio),
        'small_tree_height_threshold': float(small_tree_height_threshold),
        'scene_max_tree_height': float(scene_max_tree_height),
        'counts': {
            'pc1_points': int(pc1_count),
            'pc2_points': int(pc2_count),
            'pc3_points': int(pc3_points.shape[0]),
            'voxels': num_voxels,
            'candidate_voxels': int(tree_indices.numel()),
            'selected_queries': int(selected_indices.numel()),
            'decoder_retained_queries': int(decoder_retained_indices.numel()),
            'external_retained_queries': int(external_retained_indices.numel()),
        },
        'candidate_score_stats': _summarize_scores(
            bi_tree_prob[tree_indices] if tree_indices.numel() else None),
        'decoder_score_stats': _summarize_scores(decoder_scores),
        'external_score_stats': _summarize_scores(external_scores),
    }

    if gt_semantic_point_labels is None or gt_instance_point_labels is None:
        region_payload['diagnostic_available'] = False
        return region_payload

    gt_semantic_point_labels = _to_tensor(
        gt_semantic_point_labels, device=device, dtype=torch.long)
    gt_instance_point_labels = _to_tensor(
        gt_instance_point_labels, device=device, dtype=torch.long)

    voxel_semantic_labels = aggregate_majority_labels(
        gt_semantic_point_labels, voxel_superpoints)
    voxel_instance_labels = aggregate_majority_labels(
        gt_instance_point_labels, voxel_superpoints)
    gt_tree_mask = voxel_semantic_labels > 0

    instance_table = _build_instance_table(
        voxel_instance_labels, gt_tree_mask, scene_small_instance_ids)
    instance_ids = instance_table['instance_ids']
    small_ids = instance_table['small_ids']

    pred_tree_mask = torch.zeros(
        num_voxels, dtype=torch.bool, device=device)
    if tree_indices.numel():
        pred_tree_mask[tree_indices] = True

    tp = int((pred_tree_mask & gt_tree_mask).sum().item())
    fp = int((pred_tree_mask & ~gt_tree_mask).sum().item())
    fn = int((~pred_tree_mask & gt_tree_mask).sum().item())

    candidate_cov = _coverage_from_indices(
        tree_indices, voxel_instance_labels, gt_tree_mask,
        instance_ids, small_ids, max_missed_ids)
    selected_cov = _coverage_from_indices(
        selected_indices, voxel_instance_labels, gt_tree_mask,
        instance_ids, small_ids, max_missed_ids)
    decoder_cov = _coverage_from_indices(
        decoder_retained_indices, voxel_instance_labels, gt_tree_mask,
        instance_ids, small_ids, max_missed_ids)
    external_cov = _coverage_from_indices(
        external_retained_indices, voxel_instance_labels, gt_tree_mask,
        instance_ids, small_ids, max_missed_ids)

    selected_centroid = _normalized_centroid_distance(
        selected_indices, voxel_coords, voxel_instance_labels,
        gt_tree_mask, instance_ids)
    decoder_centroid = _normalized_centroid_distance(
        decoder_retained_indices, voxel_coords, voxel_instance_labels,
        gt_tree_mask, instance_ids)
    external_centroid = _normalized_centroid_distance(
        external_retained_indices, voxel_coords, voxel_instance_labels,
        gt_tree_mask, instance_ids)

    selected_valid_queries = 0
    if selected_indices.numel():
        selected_gt_mask = gt_tree_mask[selected_indices] & (
            voxel_instance_labels[selected_indices] >= 0)
        selected_valid_queries = int(selected_gt_mask.sum().item())

    decoder_valid_queries = 0
    if decoder_retained_indices.numel():
        decoder_gt_mask = gt_tree_mask[decoder_retained_indices] & (
            voxel_instance_labels[decoder_retained_indices] >= 0)
        decoder_valid_queries = int(decoder_gt_mask.sum().item())

    external_valid_queries = 0
    if external_retained_indices.numel():
        external_gt_mask = gt_tree_mask[external_retained_indices] & (
            voxel_instance_labels[external_retained_indices] >= 0)
        external_valid_queries = int(external_gt_mask.sum().item())

    region_payload['diagnostic_available'] = True
    region_payload['counts'].update({
        'gt_tree_voxels': int(gt_tree_mask.sum().item()),
        'gt_tree_instances': int(instance_ids.numel()),
        'gt_small_tree_instances': int(small_ids.numel()),
        'selected_valid_gt_queries': selected_valid_queries,
        'decoder_valid_gt_queries': decoder_valid_queries,
        'external_valid_gt_queries': external_valid_queries,
    })
    region_payload['bi_semantic'] = {
        'tree_voxel_precision': _safe_ratio(tp, tp + fp),
        'tree_voxel_recall': _safe_ratio(tp, tp + fn),
        'tree_voxel_f1': _safe_ratio(2 * tp, 2 * tp + fp + fn),
        'candidate_instance_coverage': candidate_cov['instance_coverage'],
        'candidate_small_tree_coverage': candidate_cov['small_tree_coverage'],
        'candidate_missed_instance_count': candidate_cov['missed_instance_count'],
        'candidate_missed_small_instance_count': candidate_cov['missed_small_instance_count'],
        'candidate_missed_instance_ids_topk': candidate_cov['missed_instance_ids_topk'],
        'candidate_missed_small_instance_ids_topk': candidate_cov['missed_small_instance_ids_topk'],
        'tree_prob_on_gt_tree_voxels': _summarize_scores(
            bi_tree_prob[gt_tree_mask])['mean'] if gt_tree_mask.any() else 0.0,
        'tree_prob_on_gt_bg_voxels': _summarize_scores(
            bi_tree_prob[~gt_tree_mask])['mean'] if (~gt_tree_mask).any() else 0.0,
    }

    selected_allocation = _query_allocation(
        selected_indices, voxel_instance_labels, gt_tree_mask,
        instance_ids, small_ids)
    decoder_allocation = _query_allocation(
        decoder_retained_indices, voxel_instance_labels, gt_tree_mask,
        instance_ids, small_ids)
    external_allocation = _query_allocation(
        external_retained_indices, voxel_instance_labels, gt_tree_mask,
        instance_ids, small_ids)

    region_payload['query_selection'] = {
        'instance_coverage': selected_cov['instance_coverage'],
        'small_tree_coverage': selected_cov['small_tree_coverage'],
        'coverage_drop_from_candidates': (
            candidate_cov['instance_coverage'] - selected_cov['instance_coverage']),
        'small_tree_drop_from_candidates': (
            candidate_cov['small_tree_coverage'] - selected_cov['small_tree_coverage']),
        'valid_query_ratio': _safe_ratio(
            selected_valid_queries, selected_indices.numel()),
        **selected_allocation,
        **selected_centroid,
        **_pairwise_distance_summary(selected_indices, voxel_coords),
    }
    region_payload['decoder'] = {
        'instance_coverage': decoder_cov['instance_coverage'],
        'small_tree_coverage': decoder_cov['small_tree_coverage'],
        'external_instance_coverage': external_cov['instance_coverage'],
        'external_small_tree_coverage': external_cov['small_tree_coverage'],
        'query_survival_ratio': _safe_ratio(
            decoder_retained_indices.numel(), selected_indices.numel()),
        'external_query_survival_ratio': _safe_ratio(
            external_retained_indices.numel(), selected_indices.numel()),
        'valid_query_ratio': _safe_ratio(
            decoder_valid_queries, decoder_retained_indices.numel()),
        'external_valid_query_ratio': _safe_ratio(
            external_valid_queries, external_retained_indices.numel()),
        **decoder_allocation,
        'external_zero_query_instances': external_allocation['zero_query_instances'],
        'external_zero_query_instance_ratio': external_allocation[
            'zero_query_instance_ratio'],
        'external_zero_query_small_instances': external_allocation[
            'zero_query_small_instances'],
        'external_zero_query_small_ratio': external_allocation[
            'zero_query_small_ratio'],
        'external_mean_queries_per_instance': external_allocation[
            'mean_queries_per_instance'],
        'external_std_queries_per_instance': external_allocation[
            'std_queries_per_instance'],
        'external_min_queries_per_instance': external_allocation[
            'min_queries_per_instance'],
        'external_max_queries_per_instance': external_allocation[
            'max_queries_per_instance'],
        **decoder_centroid,
        'external_mean_normalized_centroid_distance': external_centroid[
            'mean_normalized_centroid_distance'],
        'external_max_normalized_centroid_distance': external_centroid[
            'max_normalized_centroid_distance'],
    }
    region_payload['height_profile'] = _height_band_profile(
        gt_tree_mask,
        voxel_coords,
        {
            'candidate_voxels': tree_indices,
            'selected_queries': selected_indices,
            'decoder_retained_queries': decoder_retained_indices,
            'external_retained_queries': external_retained_indices,
        })
    return region_payload


class QPSDiagnosticRecorder:
    """Collect per-region diagnostics and export one JSON per scene."""

    def __init__(self,
                 scene_name,
                 lidar_path,
                 output_dir,
                 fps_mode,
                 query_budget,
                 small_tree_ratio=1.0 / 3.0,
                 small_tree_height_threshold=0.0,
                 scene_max_tree_height=0.0,
                 scene_small_instance_ids=None,
                 candidate_prob_threshold=0.3,
                 max_missed_ids=20,
                 print_to_stdout=True,
                 region_stride=1):
        self.scene_name = scene_name
        self.lidar_path = lidar_path
        self.output_dir = output_dir
        self.fps_mode = fps_mode
        self.query_budget = int(query_budget)
        self.small_tree_ratio = float(small_tree_ratio)
        self.small_tree_height_threshold = float(small_tree_height_threshold)
        self.scene_max_tree_height = float(scene_max_tree_height)
        self.scene_small_instance_ids = scene_small_instance_ids
        self.candidate_prob_threshold = float(candidate_prob_threshold)
        self.max_missed_ids = int(max_missed_ids)
        self.print_to_stdout = bool(print_to_stdout)
        self.region_stride = max(int(region_stride), 1)
        self.regions = []
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def output_path(self):
        return os.path.join(self.output_dir, f'{self.scene_name}_qps_diag.json')

    def should_record(self, region_idx):
        return int(region_idx) % self.region_stride == 0

    def record_region(self, region_payload):
        self.regions.append(region_payload)
        if not self.print_to_stdout or not region_payload.get('diagnostic_available', False):
            return
        bi = region_payload['bi_semantic']
        qs = region_payload['query_selection']
        dec = region_payload['decoder']
        print(
            '[QPS-DIAG] '
            f"scene={self.scene_name} "
            f"region={region_payload['region_index']} "
            f"cand_cov={bi['candidate_instance_coverage']:.3f} "
            f"query_cov={qs['instance_coverage']:.3f} "
            f"query_small={qs['small_tree_coverage']:.3f} "
            f"dec_cov={dec['instance_coverage']:.3f} "
            f"ext_cov={dec['external_instance_coverage']:.3f}"
        )

    def finalize(self):
        payload = {
            'scene_name': self.scene_name,
            'lidar_path': self.lidar_path,
            'fps_mode': self.fps_mode,
            'query_budget': self.query_budget,
            'small_tree_ratio': self.small_tree_ratio,
            'small_tree_height_threshold': self.small_tree_height_threshold,
            'scene_max_tree_height': self.scene_max_tree_height,
            'candidate_prob_threshold': self.candidate_prob_threshold,
            'region_stride': self.region_stride,
            'num_recorded_regions': len(self.regions),
            'regions': self.regions,
            'scene_summary': self._build_scene_summary(),
        }
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        if self.print_to_stdout:
            print(f'[QPS-DIAG] saved {self.output_path}')
        return self.output_path

    def _collect_metric(self, extractor):
        values = []
        for region in self.regions:
            if not region.get('diagnostic_available', False):
                continue
            values.append(float(extractor(region)))
        if not values:
            return {'mean': 0.0, 'median': 0.0, 'min': 0.0}
        return {
            'mean': float(mean(values)),
            'median': float(median(values)),
            'min': float(min(values)),
        }

    def _build_scene_summary(self):
        diagnostic_regions = [
            region for region in self.regions
            if region.get('diagnostic_available', False)
        ]
        if not diagnostic_regions:
            return {
                'diagnostic_region_count': 0,
                'region_average': {},
                'region_median': {},
                'worst_region': {},
            }

        metrics = {
            'bi_tree_voxel_recall': lambda r: r['bi_semantic']['tree_voxel_recall'],
            'bi_tree_voxel_precision': lambda r: r['bi_semantic']['tree_voxel_precision'],
            'candidate_instance_coverage': lambda r: r['bi_semantic']['candidate_instance_coverage'],
            'candidate_small_tree_coverage': lambda r: r['bi_semantic']['candidate_small_tree_coverage'],
            'query_instance_coverage': lambda r: r['query_selection']['instance_coverage'],
            'query_small_tree_coverage': lambda r: r['query_selection']['small_tree_coverage'],
            'query_valid_ratio': lambda r: r['query_selection']['valid_query_ratio'],
            'query_zero_instance_ratio': lambda r: r['query_selection']['zero_query_instance_ratio'],
            'query_zero_small_ratio': lambda r: r['query_selection']['zero_query_small_ratio'],
            'query_mean_queries_per_instance': lambda r: r['query_selection']['mean_queries_per_instance'],
            'query_mean_centroid_distance': lambda r: r['query_selection']['mean_normalized_centroid_distance'],
            'decoder_instance_coverage': lambda r: r['decoder']['instance_coverage'],
            'decoder_small_tree_coverage': lambda r: r['decoder']['small_tree_coverage'],
            'decoder_external_instance_coverage': lambda r: r['decoder']['external_instance_coverage'],
            'decoder_external_small_tree_coverage': lambda r: r['decoder']['external_small_tree_coverage'],
            'decoder_query_survival_ratio': lambda r: r['decoder']['query_survival_ratio'],
            'decoder_external_query_survival_ratio': lambda r: r['decoder']['external_query_survival_ratio'],
        }

        region_average = {}
        region_median = {}
        for name, extractor in metrics.items():
            stats = self._collect_metric(extractor)
            region_average[name] = stats['mean']
            region_median[name] = stats['median']

        worst_region = min(
            diagnostic_regions,
            key=lambda r: (
                r['query_selection']['small_tree_coverage'],
                r['query_selection']['instance_coverage'],
            ))
        return {
            'diagnostic_region_count': len(diagnostic_regions),
            'region_average': region_average,
            'region_median': region_median,
            'worst_region': {
                'region_index': int(worst_region['region_index']),
                'query_instance_coverage': float(
                    worst_region['query_selection']['instance_coverage']),
                'query_small_tree_coverage': float(
                    worst_region['query_selection']['small_tree_coverage']),
                'decoder_external_instance_coverage': float(
                    worst_region['decoder']['external_instance_coverage']),
            },
        }


def maybe_build_qps_diagnostic_recorder(scene_name,
                                        lidar_path,
                                        output_root,
                                        fps_mode,
                                        query_budget,
                                        scene_small_tree_profile=None):
    enabled = os.environ.get('FF3D_QPS_DIAG', '0') == '1'
    diag_dir = os.environ.get('FF3D_QPS_DIAG_DIR')
    if not enabled and not diag_dir:
        return None

    output_dir = diag_dir or os.path.join(output_root, 'qps_diag')
    small_tree_ratio = float(
        os.environ.get('FF3D_QPS_DIAG_SMALL_TREE_RATIO', 1.0 / 3.0))
    scene_small_tree_profile = scene_small_tree_profile or {}
    return QPSDiagnosticRecorder(
        scene_name=scene_name,
        lidar_path=lidar_path,
        output_dir=output_dir,
        fps_mode=fps_mode,
        query_budget=query_budget,
        small_tree_ratio=small_tree_ratio,
        small_tree_height_threshold=scene_small_tree_profile.get(
            'small_tree_height_threshold', 0.0),
        scene_max_tree_height=scene_small_tree_profile.get(
            'max_tree_height', 0.0),
        scene_small_instance_ids=scene_small_tree_profile.get(
            'small_instance_ids'),
        candidate_prob_threshold=float(
            os.environ.get('FF3D_QPS_DIAG_TREE_PROB_THR', 0.3)),
        max_missed_ids=int(
            os.environ.get('FF3D_QPS_DIAG_MAX_MISSED_IDS', 20)),
        print_to_stdout=os.environ.get('FF3D_QPS_DIAG_STDOUT', '1') != '0',
        region_stride=int(os.environ.get('FF3D_QPS_DIAG_REGION_STRIDE', 1)),
    )
