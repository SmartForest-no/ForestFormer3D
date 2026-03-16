import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from glob import glob

import numpy as np
from plyfile import PlyData, PlyElement

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


TREE_CLASS = 2
STUFF_CLASSES = (1,)
THING_CLASSES = (2, 3)
SEM_CLASS_IDS = [1, 2, 3]
SEM_CLASS_NAMES = {1: "ground", 2: "wood", 3: "leaf"}


def read_ply_fields(ply_path):
    data = PlyData.read(ply_path)
    if "vertex" not in data:
        raise ValueError(f"Missing vertex element in {ply_path}")
    vertex = data["vertex"].data
    fields = vertex.dtype.names
    required = [
        "x",
        "y",
        "z",
        "semantic_pred",
        "semantic_gt",
        "instance_pred",
        "instance_gt",
    ]
    missing = [field for field in required if field not in fields]
    if missing:
        raise ValueError(f"{ply_path} missing fields: {missing}")
    output = {name: vertex[name] for name in required}
    output["vertex"] = vertex
    return output


def binarize_semantic_labels(semantic_labels):
    semantic_labels = semantic_labels.astype(np.int64).copy()
    semantic_labels[np.isin(semantic_labels, STUFF_CLASSES)] = 1
    semantic_labels[np.isin(semantic_labels, THING_CLASSES)] = TREE_CLASS
    return semantic_labels


def majority_label(values):
    if values.size == 0:
        return -1
    unique_values, counts = np.unique(values, return_counts=True)
    return int(unique_values[np.argmax(counts)])


def prepare_scene_arrays(data_dict):
    xyz_full = np.stack(
        [data_dict["x"], data_dict["y"], data_dict["z"]], axis=1).astype(
            np.float32)
    sem_gt_full = data_dict["semantic_gt"].astype(np.int64) + 1
    sem_pred_full = data_dict["semantic_pred"].astype(np.int64) + 1
    gt_inst_full = data_dict["instance_gt"].astype(np.int64)
    pred_inst_full = data_dict["instance_pred"].astype(np.int64)

    eval_mask = (
        ((sem_gt_full != 0) & (sem_gt_full != 1)) |
        ((sem_pred_full != 0) & (sem_pred_full != 1)))

    return {
        "xyz_full": xyz_full,
        "sem_gt_full": sem_gt_full,
        "sem_pred_full": sem_pred_full,
        "gt_inst_full": gt_inst_full,
        "pred_inst_full": pred_inst_full,
        "eval_mask": eval_mask,
        "xyz_eval": xyz_full[eval_mask],
        "sem_gt_eval": binarize_semantic_labels(sem_gt_full[eval_mask]),
        "sem_pred_eval": binarize_semantic_labels(sem_pred_full[eval_mask]),
        "gt_inst_eval": gt_inst_full[eval_mask],
        "pred_inst_eval": pred_inst_full[eval_mask],
    }


def build_instance_semantic_index(instance_labels, semantic_labels):
    ids_by_semantic = defaultdict(list)
    for inst_id in np.unique(instance_labels):
        inst_id = int(inst_id)
        if inst_id == -1:
            continue
        inst_mask = instance_labels == inst_id
        sem_id = majority_label(semantic_labels[inst_mask])
        ids_by_semantic[sem_id].append(inst_id)
    for sem_id in ids_by_semantic:
        ids_by_semantic[sem_id].sort()
    return ids_by_semantic


def compute_instance_stats(gt_inst, pred_inst, gt_ids, pred_ids):
    gt_ids = np.asarray(sorted(int(gid) for gid in gt_ids), dtype=np.int64)
    pred_ids = np.asarray(sorted(int(pid) for pid in pred_ids), dtype=np.int64)

    gt_counts = {int(gid): int(np.sum(gt_inst == gid)) for gid in gt_ids}
    pred_counts = {int(pid): int(np.sum(pred_inst == pid)) for pid in pred_ids}

    best_pred = {int(gid): (-1, 0.0) for gid in gt_ids}
    best_gt = {int(pid): (-1, 0.0) for pid in pred_ids}
    pair_iou = {}

    if gt_ids.size == 0 or pred_ids.size == 0:
        return gt_ids, pred_ids, gt_counts, pred_counts, best_pred, best_gt, pair_iou

    valid_mask = np.isin(gt_inst, gt_ids) & np.isin(pred_inst, pred_ids)
    if not np.any(valid_mask):
        return gt_ids, pred_ids, gt_counts, pred_counts, best_pred, best_gt, pair_iou

    pairs = np.stack([gt_inst[valid_mask], pred_inst[valid_mask]], axis=1)
    unique_pairs, inter_counts = np.unique(pairs, axis=0, return_counts=True)
    for (gid, pid), inter in zip(unique_pairs, inter_counts):
        gid = int(gid)
        pid = int(pid)
        union = gt_counts.get(gid, 0) + pred_counts.get(pid, 0) - int(inter)
        if union <= 0:
            continue
        iou = float(inter) / float(union)
        pair_iou[(gid, pid)] = iou
        if iou > best_pred[gid][1]:
            best_pred[gid] = (pid, iou)
        if iou > best_gt[pid][1]:
            best_gt[pid] = (gid, iou)

    return gt_ids, pred_ids, gt_counts, pred_counts, best_pred, best_gt, pair_iou


def compute_instance_sizes(xyz, instance_labels, instance_ids, metric):
    sizes = {}
    for inst_id in instance_ids:
        inst_mask = instance_labels == inst_id
        points = xyz[inst_mask]
        if points.shape[0] == 0:
            sizes[int(inst_id)] = 0.0
            continue
        if metric == "points":
            sizes[int(inst_id)] = float(points.shape[0])
        elif metric == "height":
            sizes[int(inst_id)] = float(np.max(points[:, 2]) - np.min(points[:, 2]))
        elif metric == "volume":
            dx = float(np.max(points[:, 0]) - np.min(points[:, 0]))
            dy = float(np.max(points[:, 1]) - np.min(points[:, 1]))
            dz = float(np.max(points[:, 2]) - np.min(points[:, 2]))
            sizes[int(inst_id)] = dx * dy * dz
        else:
            raise ValueError(f"Unknown size metric: {metric}")
    return sizes


def quantile_edges(values, quantiles):
    if len(values) == 0:
        return []
    return [float(x) for x in np.quantile(values, quantiles)]


def assign_quantile_bin(value, edges):
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def write_diag_ply(
        ply_path,
        out_path,
        gt_inst,
        pred_inst,
        valid_gt_ids,
        valid_pred_ids,
        gt_match,
        pred_match):
    vertex = PlyData.read(ply_path)["vertex"].data
    num_points = vertex.shape[0]
    diag_gt = np.zeros(num_points, dtype=np.int32)
    diag_pred = np.zeros(num_points, dtype=np.int32)

    valid_gt_ids = np.asarray(sorted(int(v) for v in valid_gt_ids), dtype=np.int64)
    valid_pred_ids = np.asarray(sorted(int(v) for v in valid_pred_ids), dtype=np.int64)
    matched_gt_ids = np.asarray(
        sorted(int(gid) for gid, matched in gt_match.items() if matched),
        dtype=np.int64)
    matched_pred_ids = np.asarray(
        sorted(int(pid) for pid, matched in pred_match.items() if matched),
        dtype=np.int64)

    if valid_gt_ids.size > 0:
        gt_valid_mask = np.isin(gt_inst, valid_gt_ids)
        diag_gt[gt_valid_mask] = 2
        if matched_gt_ids.size > 0:
            diag_gt[gt_valid_mask & np.isin(gt_inst, matched_gt_ids)] = 1
    if valid_pred_ids.size > 0:
        pred_valid_mask = np.isin(pred_inst, valid_pred_ids)
        diag_pred[pred_valid_mask] = 2
        if matched_pred_ids.size > 0:
            diag_pred[pred_valid_mask & np.isin(pred_inst, matched_pred_ids)] = 1

    new_dtype = vertex.dtype.descr + [
        ("diag_gt_label", "i4"),
        ("diag_pred_label", "i4"),
    ]
    output = np.empty(num_points, dtype=new_dtype)
    for name in vertex.dtype.names:
        output[name] = vertex[name]
    output["diag_gt_label"] = diag_gt
    output["diag_pred_label"] = diag_pred

    element = PlyElement.describe(output, "vertex")
    PlyData([element], text=True).write(out_path)


def compute_semantic_iou(sem_gt, sem_pred, class_ids, mask=None):
    if mask is None:
        mask = np.ones_like(sem_gt, dtype=bool)
    iou = {}
    valid_ious = []
    for cid in class_ids:
        gt_mask = (sem_gt == cid) & mask
        pred_mask = (sem_pred == cid) & mask
        inter = np.count_nonzero(gt_mask & pred_mask)
        union = np.count_nonzero(gt_mask | pred_mask)
        iou_value = 0.0 if union == 0 else inter / union
        iou[cid] = float(iou_value)
        if np.count_nonzero(gt_mask) > 0:
            valid_ious.append(iou_value)
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return iou, miou


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ply_dir", help="Directory containing *.ply with GT/Pred fields.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same as ply_dir).")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--miss-iou", type=float, default=0.05)
    parser.add_argument("--merge-low", type=float, default=0.2)
    parser.add_argument("--merge-high", type=float, default=0.4)
    parser.add_argument(
        "--size-metric",
        choices=["points", "height", "volume", "all"],
        default="all")
    parser.add_argument("--max-frac", type=float, default=1.0 / 3.0)
    parser.add_argument("--quantiles", type=str, default="0.33,0.66")
    parser.add_argument(
        "--bucket-scope",
        choices=["per-plot", "global"],
        default="per-plot")
    parser.add_argument("--no-visual", action="store_true")
    args = parser.parse_args()

    ply_dir = args.ply_dir
    out_dir = args.out_dir or ply_dir
    os.makedirs(out_dir, exist_ok=True)

    ply_files = sorted(glob(os.path.join(ply_dir, "*.ply")))
    if not ply_files:
        raise SystemExit(f"No ply found under {ply_dir}")

    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    quantiles = [value for value in quantiles if 0.0 < value < 1.0]

    size_metrics = ["points", "height", "volume"]
    metrics_to_run = size_metrics if args.size_metric == "all" else [args.size_metric]
    global_sizes = {metric: [] for metric in size_metrics}
    per_scene_sizes = {}

    for ply_path in ply_files:
        data_dict = read_ply_fields(ply_path)
        scene_arrays = prepare_scene_arrays(data_dict)
        gt_ids_by_semantic = build_instance_semantic_index(
            scene_arrays["gt_inst_eval"], scene_arrays["sem_gt_eval"])
        gt_tree_ids = gt_ids_by_semantic.get(TREE_CLASS, [])
        sizes_all = {}
        for metric in size_metrics:
            sizes_all[metric] = compute_instance_sizes(
                scene_arrays["xyz_eval"],
                scene_arrays["gt_inst_eval"],
                gt_tree_ids,
                metric)
            global_sizes[metric].extend(list(sizes_all[metric].values()))
        per_scene_sizes[ply_path] = sizes_all

    if args.bucket_scope == "global":
        global_edges = {
            metric: quantile_edges(global_sizes[metric], quantiles)
            for metric in size_metrics
        }
    else:
        global_edges = None

    summary_rows = []
    global_tp_pred = 0
    global_tp_gt = 0
    global_fp = 0
    global_gt = 0
    global_pred = 0
    global_ovmax = []
    global_merge_gt = 0
    global_merge_pred = 0
    global_iou_tp = 0.0
    global_scene_mucov = []
    global_scene_mwcov = []
    global_scene_precision = []
    global_scene_recall = []
    global_scene_f1 = []
    global_scene_sq = []
    global_scene_pq = []
    global_sem_iou_acc = {cid: [] for cid in SEM_CLASS_IDS}
    global_sem_miou = []

    iterable = ply_files
    if tqdm is not None:
        iterable = tqdm(ply_files, desc="Diagnosing scenes", unit="scene")

    for index, ply_path in enumerate(iterable, start=1):
        if tqdm is None:
            print(
                f"[{index}/{len(ply_files)}] {os.path.basename(ply_path)}",
                flush=True)

        data_dict = read_ply_fields(ply_path)
        scene_arrays = prepare_scene_arrays(data_dict)
        gt_ids_by_semantic = build_instance_semantic_index(
            scene_arrays["gt_inst_eval"], scene_arrays["sem_gt_eval"])
        pred_ids_by_semantic = build_instance_semantic_index(
            scene_arrays["pred_inst_eval"], scene_arrays["sem_pred_eval"])
        gt_tree_ids = gt_ids_by_semantic.get(TREE_CLASS, [])
        pred_tree_ids = pred_ids_by_semantic.get(TREE_CLASS, [])

        gt_ids, pred_ids, gt_counts, pred_counts, best_pred, best_gt, _ = \
            compute_instance_stats(
                scene_arrays["gt_inst_eval"],
                scene_arrays["pred_inst_eval"],
                gt_tree_ids,
                pred_tree_ids)

        gt_match = {
            int(gid): best_pred[int(gid)][1] >= args.match_iou
            for gid in gt_ids
        }
        pred_match = {
            int(pid): best_gt[int(pid)][1] >= args.match_iou
            for pid in pred_ids
        }

        tp_gt = sum(1 for gid in gt_ids if gt_match.get(int(gid), False))
        tp_pred = sum(1 for pid in pred_ids if pred_match.get(int(pid), False))
        fp = len(pred_ids) - tp_pred
        fn = max(len(gt_ids) - tp_pred, 0)
        recall = tp_pred / len(gt_ids) if len(gt_ids) else 0.0
        precision = tp_pred / len(pred_ids) if len(pred_ids) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)

        ovmax = np.array(
            [best_pred[int(gid)][1] for gid in gt_ids],
            dtype=np.float32) if len(gt_ids) else np.zeros((0,), dtype=np.float32)
        mucov = float(np.mean(ovmax)) if ovmax.size else 0.0
        total_gt_points = sum(gt_counts.values())
        mwcov = (
            float(sum(best_pred[int(gid)][1] * gt_counts[int(gid)] for gid in gt_ids)) /
            float(total_gt_points)
            if total_gt_points > 0 else 0.0)
        miss_frac = float(np.mean(ovmax <= args.miss_iou)) if ovmax.size else 0.0
        merge_frac = float(
            np.mean((ovmax >= args.merge_low) & (ovmax <= args.merge_high))
        ) if ovmax.size else 0.0

        iou_tp = sum(
            best_gt[int(pid)][1]
            for pid in pred_ids
            if pred_match.get(int(pid), False))
        sq = float(iou_tp) / float(tp_pred) if tp_pred > 0 else 0.0
        pq = sq * f1

        best_pred_ids = [
            best_pred[int(gid)][0]
            for gid in gt_ids
            if best_pred[int(gid)][0] != -1
        ]
        pred_to_gt = Counter(best_pred_ids)
        merge_pred = sum(1 for _, count in pred_to_gt.items() if count > 1)
        merge_gt = sum(count for _, count in pred_to_gt.items() if count > 1)
        merge_rate = merge_gt / len(gt_ids) if len(gt_ids) else 0.0

        xyz_full = scene_arrays["xyz_full"]
        xs = xyz_full[:, 0]
        ys = xyz_full[:, 1]
        area = float((np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys)))
        density = (len(gt_ids) / area) if area > 0 else 0.0

        sizes_all = per_scene_sizes[ply_path]
        size_thresholds = {}
        recall_small_by_metric = {}
        recall_large_by_metric = {}
        for metric in metrics_to_run:
            sizes = sizes_all[metric]
            size_values = np.array(
                [sizes[int(gid)] for gid in gt_ids], dtype=np.float32)
            size_max = float(np.max(size_values)) if size_values.size else 0.0
            size_thr = size_max * args.max_frac
            small_ids = [
                int(gid) for gid in gt_ids if sizes[int(gid)] <= size_thr
            ]
            large_ids = [
                int(gid) for gid in gt_ids if sizes[int(gid)] > size_thr
            ]
            recall_small = (
                sum(1 for gid in small_ids if gt_match.get(gid, False)) /
                len(small_ids) if small_ids else 0.0)
            recall_large = (
                sum(1 for gid in large_ids if gt_match.get(gid, False)) /
                len(large_ids) if large_ids else 0.0)
            size_thresholds[metric] = size_thr
            recall_small_by_metric[metric] = recall_small
            recall_large_by_metric[metric] = recall_large

        size_buckets = {}
        size_bins_lookup = {}
        for metric in size_metrics:
            sizes = sizes_all[metric]
            if args.bucket_scope == "global":
                edges = global_edges[metric]
            else:
                edges = quantile_edges(list(sizes.values()), quantiles)
            values = list(sizes.values())
            vmin = float(min(values)) if values else 0.0
            vmax = float(max(values)) if values else 0.0

            gt_bin_of = {}
            bins = defaultdict(lambda: {"count": 0, "tp": 0, "mucov_sum": 0.0})
            for gid in gt_ids:
                gid = int(gid)
                bin_index = assign_quantile_bin(sizes[gid], edges)
                gt_bin_of[gid] = bin_index
                bins[bin_index]["count"] += 1
                if gt_match.get(gid, False):
                    bins[bin_index]["tp"] += 1
                bins[bin_index]["mucov_sum"] += best_pred[gid][1]

            pred_bin = defaultdict(lambda: {"count": 0, "tp": 0})
            for pid in pred_ids:
                pid = int(pid)
                best_gid, best_iou = best_gt[pid]
                if best_gid == -1 or best_gid not in gt_bin_of:
                    continue
                bin_index = gt_bin_of[best_gid]
                pred_bin[bin_index]["count"] += 1
                if best_iou >= args.match_iou:
                    pred_bin[bin_index]["tp"] += 1

            point_bins = np.full(scene_arrays["gt_inst_full"].shape, -1, dtype=np.int32)
            for gid, bin_index in gt_bin_of.items():
                point_bins[scene_arrays["gt_inst_full"] == gid] = bin_index

            bin_stats = []
            for bin_index in range(len(edges) + 1):
                low = vmin if bin_index == 0 else edges[bin_index - 1]
                high = vmax if bin_index == len(edges) else edges[bin_index]
                gt_count = bins[bin_index]["count"]
                gt_tp = bins[bin_index]["tp"]
                recall_bin = gt_tp / gt_count if gt_count else 0.0
                pred_count = pred_bin[bin_index]["count"]
                pred_tp = pred_bin[bin_index]["tp"]
                precision_bin = pred_tp / pred_count if pred_count else 0.0
                f1_bin = (
                    2 * precision_bin * recall_bin / (precision_bin + recall_bin)
                    if (precision_bin + recall_bin) > 0 else 0.0)
                mucov_bin = (
                    bins[bin_index]["mucov_sum"] / gt_count if gt_count else 0.0)

                mask_bin = point_bins == bin_index
                sem_iou_bin, sem_miou_bin = compute_semantic_iou(
                    scene_arrays["sem_gt_full"],
                    scene_arrays["sem_pred_full"],
                    SEM_CLASS_IDS,
                    mask=mask_bin)
                sem_iou_named = {
                    SEM_CLASS_NAMES[cid]: sem_iou_bin[cid]
                    for cid in SEM_CLASS_IDS
                }
                bin_stats.append({
                    "bin": bin_index,
                    "range": [float(low), float(high)],
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "precision": precision_bin,
                    "recall": recall_bin,
                    "f1": f1_bin,
                    "mucov": mucov_bin,
                    "semantic_iou": sem_iou_named,
                    "semantic_miou": sem_miou_bin,
                })

            size_buckets[metric] = {"edges": edges, "bins": bin_stats}
            size_bins_lookup[metric] = gt_bin_of

        scene_name = os.path.splitext(os.path.basename(ply_path))[0]
        sem_iou_scene, sem_miou_scene = compute_semantic_iou(
            scene_arrays["sem_gt_full"],
            scene_arrays["sem_pred_full"],
            SEM_CLASS_IDS)
        sem_iou_scene_named = {
            SEM_CLASS_NAMES[cid]: sem_iou_scene[cid]
            for cid in SEM_CLASS_IDS
        }
        scene_row = {
            "scene": scene_name,
            "gt_count": len(gt_ids),
            "pred_count": len(pred_ids),
            "tp_gt": tp_gt,
            "tp_pred": tp_pred,
            "fn_gt": fn,
            "fp_pred": fp,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "mucov": mucov,
            "mwcov": mwcov,
            "sq": sq,
            "pq": pq,
            "density": density,
            "area_xy": area,
            "size_metric": args.size_metric,
            "size_threshold": (
                size_thresholds if args.size_metric == "all"
                else size_thresholds[args.size_metric]),
            "recall_small": (
                recall_small_by_metric if args.size_metric == "all"
                else recall_small_by_metric[args.size_metric]),
            "recall_large": (
                recall_large_by_metric if args.size_metric == "all"
                else recall_large_by_metric[args.size_metric]),
            "size_buckets": size_buckets,
            "semantic_iou": sem_iou_scene_named,
            "semantic_miou": sem_miou_scene,
            "ovmax_mean": float(np.mean(ovmax)) if ovmax.size else 0.0,
            "ovmax_median": float(np.median(ovmax)) if ovmax.size else 0.0,
            "ovmax_miss_frac": miss_frac,
            "ovmax_merge_frac": merge_frac,
            "merge_pred_count": merge_pred,
            "merge_gt_count": merge_gt,
            "merge_rate": merge_rate,
        }
        summary_rows.append(scene_row)

        global_tp_pred += tp_pred
        global_tp_gt += tp_gt
        global_fp += fp
        global_gt += len(gt_ids)
        global_pred += len(pred_ids)
        global_iou_tp += iou_tp
        global_ovmax.extend(list(ovmax))
        global_merge_gt += merge_gt
        global_merge_pred += merge_pred
        global_scene_mucov.append(mucov)
        global_scene_mwcov.append(mwcov)
        global_scene_precision.append(precision)
        global_scene_recall.append(recall)
        global_scene_f1.append(f1)
        global_scene_sq.append(sq)
        global_scene_pq.append(pq)
        for cid in SEM_CLASS_IDS:
            global_sem_iou_acc[cid].append(sem_iou_scene[cid])
        global_sem_miou.append(sem_miou_scene)

        if not args.no_visual:
            out_ply = os.path.join(out_dir, f"{scene_name}_diag.ply")
            write_diag_ply(
                ply_path,
                out_ply,
                scene_arrays["gt_inst_full"],
                scene_arrays["pred_inst_full"],
                gt_ids,
                pred_ids,
                gt_match,
                pred_match)

        pred_sizes_all = {}
        for metric in size_metrics:
            pred_sizes_all[metric] = compute_instance_sizes(
                scene_arrays["xyz_eval"],
                scene_arrays["pred_inst_eval"],
                pred_ids,
                metric)

        gt_rows = []
        for gid in gt_ids:
            gid = int(gid)
            row = {
                "gt_id": gid,
                "ovmax": float(best_pred[gid][1]),
                "best_pred_id": int(best_pred[gid][0]),
            }
            for metric in size_metrics:
                row[f"{metric}"] = float(sizes_all[metric].get(gid, 0.0))
                row[f"{metric}_bin"] = int(size_bins_lookup[metric].get(gid, -1))
            gt_rows.append(row)
        gt_csv = os.path.join(out_dir, f"{scene_name}_ovmax_gt.csv")
        gt_fields = [
            "gt_id",
            "ovmax",
            "best_pred_id",
            "points",
            "points_bin",
            "height",
            "height_bin",
            "volume",
            "volume_bin",
        ]
        write_csv(gt_csv, gt_rows, gt_fields)

        pred_rows = []
        for pid in pred_ids:
            pid = int(pid)
            best_gid, best_iou = best_gt[pid]
            row = {
                "pred_id": pid,
                "ovmax": float(best_iou),
                "best_gt_id": int(best_gid),
            }
            for metric in size_metrics:
                row[f"{metric}"] = float(pred_sizes_all[metric].get(pid, 0.0))
                row[f"{metric}_gtbin"] = (
                    int(size_bins_lookup[metric].get(best_gid, -1))
                    if best_gid != -1 else -1)
            pred_rows.append(row)
        pred_csv = os.path.join(out_dir, f"{scene_name}_ovmax_pred.csv")
        pred_fields = [
            "pred_id",
            "ovmax",
            "best_gt_id",
            "points",
            "points_gtbin",
            "height",
            "height_gtbin",
            "volume",
            "volume_gtbin",
        ]
        write_csv(pred_csv, pred_rows, pred_fields)

        out_json = os.path.join(out_dir, f"{scene_name}_diagnose.json")
        with open(out_json, "w", encoding="utf-8") as file:
            json.dump(scene_row, file, indent=2)

    global_ovmax = np.array(
        global_ovmax, dtype=np.float32) if global_ovmax else np.zeros(
            (0,), dtype=np.float32)
    global_precision = (
        float(global_tp_pred) / float(global_pred) if global_pred else 0.0)
    global_recall = (
        float(global_tp_pred) / float(global_gt) if global_gt else 0.0)
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall)
        if (global_precision + global_recall) > 0 else 0.0)
    global_sq = (
        float(global_iou_tp) / float(global_tp_pred)
        if global_tp_pred > 0 else 0.0)
    global_pq = global_sq * global_f1

    global_summary = {
        "gt_count": global_gt,
        "pred_count": global_pred,
        "tp_gt": global_tp_gt,
        "tp_pred": global_tp_pred,
        "fn_gt": max(global_gt - global_tp_pred, 0),
        "fp_pred": global_fp,
        "recall": global_recall,
        "precision": global_precision,
        "f1": global_f1,
        "sq": global_sq,
        "pq": global_pq,
        "mMUCov": float(np.mean(global_scene_mucov)) if global_scene_mucov else 0.0,
        "mMWCov": float(np.mean(global_scene_mwcov)) if global_scene_mwcov else 0.0,
        "ovmax_mean": float(np.mean(global_ovmax)) if global_ovmax.size else 0.0,
        "ovmax_median": float(np.median(global_ovmax)) if global_ovmax.size else 0.0,
        "merge_gt_count": global_merge_gt,
        "merge_pred_count": global_merge_pred,
        "merge_rate": (global_merge_gt / global_gt) if global_gt else 0.0,
        "scene_macro_mean": {
            "precision": float(np.mean(global_scene_precision)) if global_scene_precision else 0.0,
            "recall": float(np.mean(global_scene_recall)) if global_scene_recall else 0.0,
            "f1": float(np.mean(global_scene_f1)) if global_scene_f1 else 0.0,
            "sq": float(np.mean(global_scene_sq)) if global_scene_sq else 0.0,
            "pq": float(np.mean(global_scene_pq)) if global_scene_pq else 0.0,
        },
        "semantic_iou": {
            SEM_CLASS_NAMES[cid]: (
                float(np.mean(global_sem_iou_acc[cid]))
                if global_sem_iou_acc[cid] else 0.0)
            for cid in SEM_CLASS_IDS
        },
        "semantic_miou": float(np.mean(global_sem_miou)) if global_sem_miou else 0.0,
    }

    with open(
            os.path.join(out_dir, "diagnosis_summary.json"),
            "w",
            encoding="utf-8") as file:
        json.dump(global_summary, file, indent=2)

    with open(
            os.path.join(out_dir, "diagnosis_per_scene.json"),
            "w",
            encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2)


if __name__ == "__main__":
    main()
