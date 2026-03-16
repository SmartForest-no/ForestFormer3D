#!/usr/bin/env python3
"""
Compute plot-level and tree-level statistics from ForAINetV2-style PLYs.

Expected PLY fields (binary):
  - x, y, z
  - semantic_seg (int, 1/2 = non-tree/tree)
  - treeID (int, 0 = non-tree/unlabeled, 1..N = tree instance)

Outputs:
  - Per-plot summary (stdout)
  - Overall summary across plots and trees (stdout)
  - Optional CSV files for per-plot and per-tree stats
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PLY_TYPE_MAP = {
    "int8": "i1",
    "char": "i1",
    "uint8": "u1",
    "uchar": "u1",
    "int16": "i2",
    "short": "i2",
    "uint16": "u2",
    "ushort": "u2",
    "int32": "i4",
    "int": "i4",
    "uint32": "u4",
    "uint": "u4",
    "float32": "f4",
    "float": "f4",
    "float64": "f8",
    "double": "f8",
    "int64": "i8",
    "uint64": "u8",
}


def parse_header(ply_path: Path) -> Tuple[str | None, int | None, List[Tuple[str, str]], int]:
    fmt = None
    num_points = None
    fields: List[Tuple[str, str]] = []
    in_vertex = False
    with ply_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(b"format "):
                parts = line.split()
                if len(parts) >= 2:
                    fmt = parts[1].decode()
            if line == b"end_header":
                data_offset = f.tell()
                break
            if line.startswith(b"element vertex"):
                in_vertex = True
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        num_points = int(parts[2])
                    except ValueError:
                        num_points = None
                continue
            if line.startswith(b"element ") and not line.startswith(b"element vertex"):
                in_vertex = False
                continue
            if in_vertex and line.startswith(b"property "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] != b"list":
                    typ = parts[1].decode()
                    name = parts[2].decode()
                    fields.append((name, typ))
        else:
            data_offset = f.tell()
    return fmt, num_points, fields, data_offset


def read_ply_binary(ply_path: Path) -> np.ndarray:
    fmt, num_points, fields, data_offset = parse_header(ply_path)
    if fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"{ply_path.name}: unsupported PLY format '{fmt}' (binary only).")
    if num_points is None or not fields:
        raise ValueError(f"{ply_path.name}: missing vertex info in header.")

    endian = "<" if fmt == "binary_little_endian" else ">"
    dtype_list = []
    for name, typ in fields:
        if typ not in PLY_TYPE_MAP:
            raise ValueError(f"{ply_path.name}: unsupported type '{typ}'")
        dtype_list.append((name, endian + PLY_TYPE_MAP[typ]))

    with ply_path.open("rb") as f:
        f.seek(data_offset)
        data = np.fromfile(f, dtype=np.dtype(dtype_list), count=num_points)
    return data


def _clamp(val: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, val))


def compute_tree_stats(
    xyz: np.ndarray,
    tree_ids: np.ndarray,
    trunk_z_frac: float,
    trunk_radius_factor: float,
    trunk_radius_min: float,
    trunk_radius_max: float,
) -> Dict[str, float]:
    unique_ids = np.unique(tree_ids)
    unique_ids = unique_ids[unique_ids > 0]
    if unique_ids.size == 0:
        return {
            "num_trees": 0,
            "mean_pts_per_tree": 0.0,
            "min_pts_per_tree": 0.0,
            "max_pts_per_tree": 0.0,
            "crown_diam_min": 0.0,
            "crown_diam_mean": 0.0,
            "crown_diam_max": 0.0,
            "crown_eq_diam_min": 0.0,
            "crown_eq_diam_mean": 0.0,
            "crown_eq_diam_max": 0.0,
            "height_min": 0.0,
            "height_mean": 0.0,
            "height_max": 0.0,
            "trunk_ratio_mean": 0.0,
        }

    pts_per_tree = []
    crown_diams = []
    crown_eq_diams = []
    heights = []
    trunk_ratios = []

    for tid in unique_ids:
        mask = tree_ids == tid
        pts = xyz[mask]
        n_pts = pts.shape[0]
        if n_pts == 0:
            continue
        pts_per_tree.append(n_pts)

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        dx = float(x.max() - x.min())
        dy = float(y.max() - y.min())
        crown_diam = max(dx, dy)
        crown_diams.append(crown_diam)

        crown_area = max(dx * dy, 0.0)
        crown_eq_diam = 2.0 * math.sqrt(crown_area / math.pi) if crown_area > 0 else 0.0
        crown_eq_diams.append(crown_eq_diam)

        height = float(z.max() - z.min())
        heights.append(height)

        # Trunk heuristic: lower trunk_z_frac of height + within trunk radius
        z_min = float(z.min())
        z_max = float(z.max())
        trunk_z_max = z_min + trunk_z_frac * (z_max - z_min)
        center_x = float(np.median(x))
        center_y = float(np.median(y))
        trunk_radius = _clamp(trunk_radius_factor * crown_diam, trunk_radius_min, trunk_radius_max)
        radial2 = (x - center_x) ** 2 + (y - center_y) ** 2
        trunk_mask = (z <= trunk_z_max) & (radial2 <= trunk_radius**2)
        trunk_ratio = float(trunk_mask.sum()) / float(n_pts)
        trunk_ratios.append(trunk_ratio)

    def _stat(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0
        arr = np.array(vals, dtype=np.float64)
        return float(arr.min()), float(arr.mean()), float(arr.max())

    pts_min, pts_mean, pts_max = _stat(pts_per_tree)
    cd_min, cd_mean, cd_max = _stat(crown_diams)
    ce_min, ce_mean, ce_max = _stat(crown_eq_diams)
    h_min, h_mean, h_max = _stat(heights)
    tr_min, tr_mean, tr_max = _stat(trunk_ratios)
    return {
        "num_trees": int(len(unique_ids)),
        "mean_pts_per_tree": pts_mean,
        "min_pts_per_tree": pts_min,
        "max_pts_per_tree": pts_max,
        "crown_diam_min": cd_min,
        "crown_diam_mean": cd_mean,
        "crown_diam_max": cd_max,
        "crown_eq_diam_min": ce_min,
        "crown_eq_diam_mean": ce_mean,
        "crown_eq_diam_max": ce_max,
        "height_min": h_min,
        "height_mean": h_mean,
        "height_max": h_max,
        "trunk_ratio_mean": tr_mean,
    }


def summarize_plot(
    ply_path: Path,
    tree_semantic_label: int,
    trunk_z_frac: float,
    trunk_radius_factor: float,
    trunk_radius_min: float,
    trunk_radius_max: float,
) -> Dict[str, float]:
    data = read_ply_binary(ply_path)
    required = {"x", "y", "z"}
    for r in required:
        if r not in data.dtype.names:
            raise ValueError(f"{ply_path.name}: missing field '{r}'")

    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    z = data["z"].astype(np.float64)
    n_points = int(x.shape[0])

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_range = x_max - x_min
    y_range = y_max - y_min
    area = float(max(x_range * y_range, 0.0))

    density = float(n_points / area) if area > 0 else 0.0
    mean_spacing = float(math.sqrt(area / n_points)) if area > 0 and n_points > 0 else 0.0

    tree_ids = None
    if "treeID" in data.dtype.names:
        tree_ids = data["treeID"].astype(np.int64)
    elif "instance" in data.dtype.names:
        tree_ids = data["instance"].astype(np.int64)

    if tree_ids is None and "semantic_seg" in data.dtype.names:
        # fallback: treat semantic tree points as one group (no instance stats)
        tree_ids = np.where(data["semantic_seg"] == tree_semantic_label, 1, 0).astype(np.int64)

    tree_points = int((tree_ids > 0).sum()) if tree_ids is not None else 0
    tree_ratio = float(tree_points / n_points) if n_points > 0 else 0.0

    tree_stats = compute_tree_stats(
        np.stack([x, y, z], axis=1),
        tree_ids if tree_ids is not None else np.zeros_like(x, dtype=np.int64),
        trunk_z_frac=trunk_z_frac,
        trunk_radius_factor=trunk_radius_factor,
        trunk_radius_min=trunk_radius_min,
        trunk_radius_max=trunk_radius_max,
    )

    return {
        "plot": ply_path.stem,
        "n_points": n_points,
        "x_range": x_range,
        "y_range": y_range,
        "area_m2": area,
        "density_pts_m2": density,
        "mean_spacing_m": mean_spacing,
        "tree_points": tree_points,
        "tree_points_ratio": tree_ratio,
        **tree_stats,
    }


def print_plot_summary(row: Dict[str, float]) -> None:
    print(f"\n=== {row['plot']} ===")
    print(f"points: {row['n_points']}, area: {row['area_m2']:.2f} m² "
          f"({row['x_range']:.2f} x {row['y_range']:.2f} m)")
    print(f"density: {row['density_pts_m2']:.2f} pts/m², "
          f"mean spacing ~ {row['mean_spacing_m']:.3f} m")
    print(f"tree points: {row['tree_points']} ({row['tree_points_ratio']*100:.2f}%)")
    print(f"trees: {row['num_trees']}, pts/tree: mean {row['mean_pts_per_tree']:.1f} "
          f"(min {row['min_pts_per_tree']:.0f}, max {row['max_pts_per_tree']:.0f})")
    print(f"crown diam (bbox): mean {row['crown_diam_mean']:.2f} m "
          f"(min {row['crown_diam_min']:.2f}, max {row['crown_diam_max']:.2f})")
    print(f"crown eq diam: mean {row['crown_eq_diam_mean']:.2f} m "
          f"(min {row['crown_eq_diam_min']:.2f}, max {row['crown_eq_diam_max']:.2f})")
    print(f"tree height: mean {row['height_mean']:.2f} m "
          f"(min {row['height_min']:.2f}, max {row['height_max']:.2f})")
    print(f"trunk ratio (heuristic): mean {row['trunk_ratio_mean']*100:.2f}%")


def write_csv(path: Path, rows: Iterable[Dict[str, float]], fieldnames: List[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="Folder containing .ply files")
    parser.add_argument("--out", default=None, help="Optional CSV output for per-plot stats")
    parser.add_argument("--tree-semantic-label", type=int, default=2, help="semantic_seg value for tree")
    parser.add_argument("--trunk-z-frac", type=float, default=0.2,
                        help="Lower fraction of height treated as trunk (0-1)")
    parser.add_argument("--trunk-radius-factor", type=float, default=0.1,
                        help="Trunk radius = factor * crown_diameter (clamped)")
    parser.add_argument("--trunk-radius-min", type=float, default=0.2,
                        help="Minimum trunk radius (m)")
    parser.add_argument("--trunk-radius-max", type=float, default=1.0,
                        help="Maximum trunk radius (m)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input is not a directory: {in_dir}")

    ply_files = sorted(in_dir.glob("*.ply"))
    if not ply_files:
        raise SystemExit(f"No .ply files found in {in_dir}")

    rows = []
    for p in ply_files:
        row = summarize_plot(
            p,
            tree_semantic_label=args.tree_semantic_label,
            trunk_z_frac=args.trunk_z_frac,
            trunk_radius_factor=args.trunk_radius_factor,
            trunk_radius_min=args.trunk_radius_min,
            trunk_radius_max=args.trunk_radius_max,
        )
        rows.append(row)
        print_plot_summary(row)

    # Overall summary (plots)
    def _arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=np.float64)

    print("\n=== OVERALL (plots) ===")
    for key, label in [
        ("area_m2", "area m²"),
        ("density_pts_m2", "density pts/m²"),
        ("mean_spacing_m", "mean spacing m"),
        ("num_trees", "trees per plot"),
        ("mean_pts_per_tree", "mean pts/tree"),
        ("crown_diam_mean", "crown diam mean m"),
        ("height_mean", "height mean m"),
        ("trunk_ratio_mean", "trunk ratio mean"),
    ]:
        arr = _arr(key)
        print(f"{label}: min {arr.min():.3f}, mean {arr.mean():.3f}, max {arr.max():.3f}")

    if args.out:
        out_path = Path(args.out)
        if out_path.exists() and out_path.is_dir():
            out_path = out_path / "plot_stats.csv"
        fieldnames = list(rows[0].keys())
        write_csv(out_path, rows, fieldnames)
        print(f"\nWrote per-plot CSV: {out_path}")


if __name__ == "__main__":
    main()
