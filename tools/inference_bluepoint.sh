#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 二次推理主脚本（bluepoints 迭代）
#
# 目标：
# 1) 对 test_list_initial.txt 中的每个基础样地先做第 1 轮推理；
# 2) 将第 1 轮输出的 bluepoints 作为第 2 轮（或更多轮）输入继续推理；
# 3) 最后按基础样地名合并各轮结果（merge_prediction.py）。
#
# 关键设计点：
# - 使用“独立临时目录 TEST_DATA_DIR_TMP”承载上一轮 bluepoints，避免和原始 test_data 混用；
# - 第 1 轮 offsets 单独缓存到 OFFSETS_CACHE_DIR，避免后续轮次重建数据时覆盖/缺失；
# - 所有路径相对仓库根目录计算，避免 /workspace 与 /home/... 的路径错位问题。
# -----------------------------------------------------------------------------

set -u

# 动态解析仓库根目录：
# - SCRIPT_DIR: 当前脚本目录（.../tools）
# - WORK_DIR  : 仓库根目录（.../ForestFormer3D）
# 这样无论在容器 /workspace 还是宿主机 /home/... 都能用同一份逻辑。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ----------------------------
# 路径与运行参数
# ----------------------------
# 数据根目录（这里默认用 custom_test 软链接目录）
DATA_ROOT="$WORK_DIR/data/ForAINetV2_custom_test"
# 初始样地列表：外层循环读取它
TEST_LIST_INIT="$DATA_ROOT/meta_data/test_list_initial.txt"
# 当前轮的单样地列表：每轮覆盖写入，仅保留一个 scan stem
TEST_LIST="$DATA_ROOT/meta_data/test_list.txt"
# 原始测试输入目录（第 1 轮）
TEST_DATA_DIR_ORIG="$DATA_ROOT/test_data"
# 临时输入目录（第 2 轮及以后，只放上一轮 bluepoints）
TEST_DATA_DIR_TMP="$DATA_ROOT/test_data_bluepoints_tmp"
# 推理配置与模型
CONFIG_FILE="$WORK_DIR/configs/oneformer3d_qs_radius32_qp300_hainan.py"
MODEL_PATH="$WORK_DIR/work_dirs/run2_overfit_step2/epoch_6000_fix.pth"
# 迭代轮数：可通过环境变量覆盖，默认 2
ITERATIONS="${ITERATIONS:-2}"
# 推理输出目录（predict 会把 ply 输出到这里）
BLUEPOINTS_DIR="$WORK_DIR/work_dirs/run2_overfit_bluepoints"
# offsets 缓存目录（只缓存“基础样地”的 offsets，供 merge 使用）
OFFSETS_CACHE_DIR="$DATA_ROOT/forainetv2_instance_data_offsets_cache"

# 给模型 predict 用的环境变量：
# - FF3D_OUTPUT_PATH / BLUEPOINTS_DIR: 输出 ply 的目录
# - FF3D_USE_BLUEPOINTS=1: 启用 save_bluepoints 分支
export BLUEPOINTS_DIR
export FF3D_OUTPUT_PATH="$BLUEPOINTS_DIR"
export FF3D_USE_BLUEPOINTS=1

# 启动前基础校验：初始列表不存在则直接退出，避免空跑。
if [ ! -f "$TEST_LIST_INIT" ]; then
  echo "ERROR: TEST_LIST_INIT not found: $TEST_LIST_INIT"
  exit 1
fi

# 预创建目录：
# - TEST_DATA_DIR_TMP: 存放第 N-1 轮 bluepoints，作为第 N 轮输入
# - OFFSETS_CACHE_DIR: 存放每个基础样地的 offsets.npy
mkdir -p "$TEST_DATA_DIR_TMP" "$OFFSETS_CACHE_DIR"

# ============================================================================
# 外层循环：按基础样地逐个处理
# ============================================================================
while IFS= read -r scan_name || [ -n "$scan_name" ]; do
  # 清理 Windows 行尾与首尾空白，避免名字污染
  scan_name="$(echo "$scan_name" | tr -d '\r' | xargs)"
  if [ -z "$scan_name" ]; then
    continue
  fi

  echo "=============================="
  echo "Processing base scan: $scan_name"

  iteration=1
  # --------------------------------------------------------------------------
  # 内层循环：单个基础样地进行多轮推理（iter1, iter2, ...）
  # --------------------------------------------------------------------------
  while [ "$iteration" -le "$ITERATIONS" ]; do
    if [ "$iteration" -eq 1 ]; then
      # 第 1 轮：
      # - 直接使用基础样地名 scan_name
      # - 输入目录是原始 test_data
      input_scan_stem="$scan_name"
      input_data_dir="$TEST_DATA_DIR_ORIG"
    else
      # 第 2 轮及以后：
      # - 输入必须来自上一轮的 bluepoints 文件
      # - 文件命名：{scan_name}__bluepoints_iter{iteration-1}.ply
      prev_bluepoints_file="$BLUEPOINTS_DIR/${scan_name}__bluepoints_iter$((iteration-1)).ply"
      if [ ! -f "$prev_bluepoints_file" ]; then
        # 找不到上一轮 bluepoints，就停止当前基础样地后续迭代
        echo "WARN: missing prev bluepoints, stop iter=$iteration: $prev_bluepoints_file"
        break
      fi

      # 清空临时目录并只拷贝“当前样地上一轮”的 bluepoints：
      # 避免多个样地 bluepoints 混在同一个输入目录中导致读错。
      rm -f "$TEST_DATA_DIR_TMP"/*.ply 2>/dev/null || true
      cp "$prev_bluepoints_file" "$TEST_DATA_DIR_TMP/"

      # 本轮输入 scan stem 与上一轮 bluepoints 文件名（去 .ply）一致
      input_scan_stem="${scan_name}__bluepoints_iter$((iteration-1))"
      input_data_dir="$TEST_DATA_DIR_TMP"
    fi

    echo "Iteration $iteration (input_scan=$input_scan_stem, input_dir=$(basename "$input_data_dir"))"

    # 当前轮只处理一个 scan stem：
    # 下游 batch_load_ForAINetV2_data.py 会从 test_list.txt 读取它。
    echo "$input_scan_stem" > "$TEST_LIST"

    # 1) 将当前轮输入 ply（来自 input_data_dir）转成 forainetv2_instance_data
    # 2) 生成 oneformer3d infos
    cd "$DATA_ROOT" || exit 1
    python ../ForAINetV2/batch_load_ForAINetV2_data.py \
      --output_folder forainetv2_instance_data \
      --test_scan_names_file meta_data/test_list.txt \
      --test_forainetv2_dir "$(basename "$input_data_dir")"
    cd "$WORK_DIR" || exit 1

    python tools/create_data_forainetv2.py forainetv2 --root-path "$DATA_ROOT" --out-dir "$DATA_ROOT"

    # 修复 infos pkl（历史问题：列表结构 -> 字典结构）
    # 若生成了 *.fixed.pkl，就覆盖回原始文件名供 test.py 使用。
    python tools/fix_forainetv2_infos.py --root "$DATA_ROOT"
    for split in train val test; do
      fixed="$DATA_ROOT/forainetv2_oneformer3d_infos_${split}.pkl.fixed.pkl"
      orig="$DATA_ROOT/forainetv2_oneformer3d_infos_${split}.pkl"
      if [ -f "$fixed" ]; then
        mv "$fixed" "$orig"
      fi
    done

    # 仅在第 1 轮缓存“基础样地 offsets”：
    # merge_prediction.py 需要 {scan_name}_offsets.npy；
    # 而后续轮次会生成 bluepoints 命名的 offsets，可能覆盖或不再保留基础 offsets。
    if [ "$iteration" -eq 1 ]; then
      base_offsets="$DATA_ROOT/forainetv2_instance_data/${scan_name}_offsets.npy"
      if [ -f "$base_offsets" ]; then
        cp -f "$base_offsets" "$OFFSETS_CACHE_DIR/${scan_name}_offsets.npy"
      else
        echo "WARN: base offsets not found after iter1 preprocessing: $base_offsets"
      fi
    fi

    # 按需修改配置中的 score_th（这里固定 0.4）
    # 如需分轮策略，可改成随 iteration 变化。
    score_th=0.4
    sed -i "s/score_th = [0-9.]\+/score_th = ${score_th}/g" "$CONFIG_FILE"

    # 执行当前轮推理
    CUDA_VISIBLE_DEVICES=0 python tools/test.py "$CONFIG_FILE" "$MODEL_PATH"

    # 约定输出命名：
    # - 主输出：{scan_name}__iter{iteration}.ply
    # - 蓝点输出：{scan_name}__bluepoints_iter{iteration}.ply
    pred_file="$BLUEPOINTS_DIR/${scan_name}__iter${iteration}.ply"
    blue_file="$BLUEPOINTS_DIR/${scan_name}__bluepoints_iter${iteration}.ply"

    # 缺任一输出都停止该样地后续迭代，转入 merge 阶段
    if [ ! -f "$pred_file" ]; then
      echo "WARN: missing prediction output, stop iter=$iteration: $pred_file"
      break
    fi
    if [ ! -f "$blue_file" ]; then
      echo "WARN: missing bluepoints output, stop iter=$iteration: $blue_file"
      break
    fi

    ((iteration++))
  done

  # 单个基础样地所有可执行迭代结束后进行合并
  # 说明：merge 时使用缓存的基础 offsets，保证坐标还原稳定。
  echo "Merging results for $scan_name"
  python tools/merge_prediction.py "$scan_name" "$BLUEPOINTS_DIR" "$ITERATIONS" "$OFFSETS_CACHE_DIR"
  echo "Finished $scan_name"
done < "$TEST_LIST_INIT"

# # Evaluate each round's results
# for ((i=1; i<=ITERATIONS; i++)); do
#     ROUND_DIR="$BLUEPOINTS_DIR/round_$i"
#     echo "Evaluating results in: $ROUND_DIR"
#     python tools/final_eval.py "$ROUND_DIR"
# done

# # Evaluate results after noise removal (assuming any number suffix)
# for ((i=1; i<=ITERATIONS; i++)); do
#     for ROUND_DIR in "$BLUEPOINTS_DIR"/round_"$i"_after_remove_noise_*; do
#         if [ -d "$ROUND_DIR" ]; then
#             echo "Evaluating results in: $ROUND_DIR"
#             python tools/final_eval.py "$ROUND_DIR"
#         fi
#     done
# done

echo "All test cases processed."
