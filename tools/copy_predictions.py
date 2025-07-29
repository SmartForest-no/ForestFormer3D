import os
import shutil
from pathlib import Path

src_main = Path("/workspace/work_dirs/bluepoint_th04fixed_03_priority_test")
src_delete = Path("/workspace/work_dirs/bluepoint_th04fixed_03_priority_test_tobedelete")

dst_root = Path("/workspace/work_dirs/bluepoint_th04fixed_03_priority_test_merged")

round_dirs = [
    "round_1",
    "round_2_after_remove_noise_200"
]

for round_dir in round_dirs:
    dst_dir = dst_root / round_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in [src_main, src_delete]:
        src_round_dir = src / round_dir
        if not src_round_dir.exists():
            print(f"Warning: {src_round_dir} does not exist, skipping.")
            continue

        for ply_file in src_round_dir.glob("*.ply"):
            dst_file = dst_dir / ply_file.name
            if dst_file.exists():
                print(f"File already exists, skipping: {dst_file}")
            else:
                shutil.copy2(ply_file, dst_file)
                print(f"Copied: {ply_file} -> {dst_file}")
