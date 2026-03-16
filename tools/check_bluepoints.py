#!/usr/bin/env python3
"""
Check bluepoints/iter PLY outputs for multi-round inference.

This script does NOT parse full point data. It only reads the PLY header to get
vertex counts, so it's fast even on large files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            # PLY header is ASCII and ends with "end_header\n"
            header = b""
            while b"end_header\n" not in header:
                chunk = f.read(4096)
                if not chunk:
                    break
                header += chunk
                if len(header) > 256_000:
                    break
        text = header.decode("ascii", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
    except Exception:
        return None
    return None


def read_scan_list(path: Path) -> list[str]:
    scans: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        name = raw.strip().replace("\r", "")
        if not name:
            continue
        scans.append(name)
    return scans


@dataclass(frozen=True)
class Row:
    scan: str
    iteration: int
    pred_file: str
    pred_vertices: int | None
    blue_file: str
    blue_vertices: int | None

    @property
    def ok_pred(self) -> bool:
        return self.pred_vertices is not None and self.pred_vertices > 0

    @property
    def ok_blue(self) -> bool:
        return self.blue_vertices is not None and self.blue_vertices > 0


def iter_rows(pred_dir: Path, scans: Iterable[str], iterations: int) -> list[Row]:
    rows: list[Row] = []
    for scan in scans:
        for it in range(1, iterations + 1):
            pred = pred_dir / f"{scan}__iter{it}.ply"
            blue = pred_dir / f"{scan}__bluepoints_iter{it}.ply"
            rows.append(
                Row(
                    scan=scan,
                    iteration=it,
                    pred_file=str(pred),
                    pred_vertices=ply_vertex_count(pred),
                    blue_file=str(blue),
                    blue_vertices=ply_vertex_count(blue),
                )
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True, help="Directory containing *__iterN.ply outputs")
    ap.add_argument("--scan-list", type=Path, help="TXT list of base scan ids (one per line)")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--out-csv", type=Path, help="Optional CSV output path")
    args = ap.parse_args()

    pred_dir: Path = args.pred_dir
    if not pred_dir.exists():
        print(f"ERROR: --pred-dir not found: {pred_dir}", file=sys.stderr)
        return 2

    if args.scan_list:
        scans = read_scan_list(args.scan_list)
    else:
        # best-effort: infer from existing __iter1.ply files
        scans = sorted({p.name.split("__iter1.ply")[0] for p in pred_dir.glob("*__iter1.ply")})

    rows = iter_rows(pred_dir, scans, args.iterations)

    # Print a compact summary
    bad = 0
    for r in rows:
        ok = "OK" if (r.ok_pred and r.ok_blue) else "BAD"
        if ok == "BAD":
            bad += 1
        pv = "None" if r.pred_vertices is None else str(r.pred_vertices)
        bv = "None" if r.blue_vertices is None else str(r.blue_vertices)
        print(f"{ok}\t{r.scan}\titer{r.iteration}\tpred={pv}\tblue={bv}")

    print(f"Total rows: {len(rows)}, BAD rows: {bad}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "scan",
                    "iteration",
                    "pred_file",
                    "pred_vertices",
                    "blue_file",
                    "blue_vertices",
                    "ok_pred",
                    "ok_blue",
                ],
            )
            w.writeheader()
            for r in rows:
                d = asdict(r)
                d["ok_pred"] = r.ok_pred
                d["ok_blue"] = r.ok_blue
                w.writerow(d)
        print(f"Wrote CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

