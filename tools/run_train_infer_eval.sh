#!/usr/bin/env bash
# 一键流程：训练 -> 修复 checkpoint -> 推理 -> 评测
# 覆盖 ff3d_inference_runbook.md 在“二次推理（Bluepoints）”之前的主流程。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${REPO_ROOT}/${p}"
  fi
}

print_usage() {
  cat <<'USAGE'
用法:
  bash tools/run_train_infer_eval.sh [选项]

默认流程:
  1) 训练
  2) 修复 checkpoint (spconv 兼容)
  3) 推理并输出到 pred_data
  4) 评测 final_eval

选项:
  --config PATH            配置文件，默认: configs/oneformer3d_qs_radius16_qp300_2many.py
  --run-dir PATH           运行目录，默认: work_dirs/run3_qps_se_1000+300
  --data-root PATH         数据根目录，默认: data/ForAINetV2_custom_test
  --epoch N                训练后用于修复/推理的 epoch，默认: 1300
  --train-gpu ID           训练 GPU，默认: 0
  --test-gpu ID            推理 GPU，默认: 0
  --prepare-data           先执行数据预处理(create_data + fix_infos)
  --skip-train             跳过训练
  --skip-fix               跳过 checkpoint 修复
  --skip-test              跳过推理
  --skip-eval              跳过评测
  --checkpoint PATH        原始 checkpoint 路径(可选，默认 run_dir/epoch_${epoch}.pth)
  --fixed-checkpoint PATH  修复后 checkpoint 路径(可选，默认 run_dir/epoch_${epoch}_fix.pth)
  --qps-diag               显式开启 QPS/ISA-FPS 诊断 JSON 导出（默认已开启）
  --no-qps-diag            关闭 QPS/ISA-FPS 诊断 JSON 导出
  --qps-diag-dir PATH      QPS 诊断输出目录，默认: <pred_dir>/qps_diag
  --qps-diag-small-tree-ratio R
                           小树阈值比例，按“当前样地最高树高 * R”定义，默认: 0.3333333333
  --qps-diag-region-stride N
                           每隔 N 个 region 记录一次诊断，默认: 1
  -h, --help               显示帮助

示例:
  bash tools/run_train_infer_eval.sh \
    --config configs/oneformer3d_qs_radius16_qp300_2many.py \
    --run-dir work_dirs/run3_qps_se_1000+300 \
    --data-root data/ForAINetV2_custom_test \
    --epoch 1300 \
    --train-gpu 1 \
    --test-gpu 1

仅重跑推理+评测:
  bash tools/run_train_infer_eval.sh --skip-train --skip-fix --epoch 1300
USAGE
}

CONFIG_REL="configs/oneformer3d_qs_radius16_qp300_2many.py"
RUN_DIR_REL="work_dirs/run3_qps_se_1000+300"
DATA_ROOT_REL="data/ForAINetV2_custom_test"
EPOCH="1300"
TRAIN_GPU="0"
TEST_GPU="0"
PREPARE_DATA=0
SKIP_TRAIN=0
SKIP_FIX=0
SKIP_TEST=0
SKIP_EVAL=0
CHECKPOINT_INPUT=""
CHECKPOINT_FIXED=""
QPS_DIAG=1
QPS_DIAG_DIR=""
QPS_DIAG_SMALL_TREE_RATIO="0.3333333333"
QPS_DIAG_REGION_STRIDE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_REL="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR_REL="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT_REL="$2"
      shift 2
      ;;
    --epoch)
      EPOCH="$2"
      shift 2
      ;;
    --train-gpu)
      TRAIN_GPU="$2"
      shift 2
      ;;
    --test-gpu)
      TEST_GPU="$2"
      shift 2
      ;;
    --prepare-data)
      PREPARE_DATA=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-fix)
      SKIP_FIX=1
      shift
      ;;
    --skip-test)
      SKIP_TEST=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --checkpoint)
      CHECKPOINT_INPUT="$2"
      shift 2
      ;;
    --fixed-checkpoint)
      CHECKPOINT_FIXED="$2"
      shift 2
      ;;
    --qps-diag)
      QPS_DIAG=1
      shift
      ;;
    --no-qps-diag)
      QPS_DIAG=0
      shift
      ;;
    --qps-diag-dir)
      QPS_DIAG_DIR="$2"
      shift 2
      ;;
    --qps-diag-small-tree-ratio)
      QPS_DIAG_SMALL_TREE_RATIO="$2"
      shift 2
      ;;
    --qps-diag-region-stride)
      QPS_DIAG_REGION_STRIDE="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1"
      print_usage
      exit 1
      ;;
  esac
done

CONFIG="$(resolve_path "$CONFIG_REL")"
RUN_DIR="$(resolve_path "$RUN_DIR_REL")"
DATA_ROOT="$(resolve_path "$DATA_ROOT_REL")"
PRED_DIR="${RUN_DIR}/pred_data"
if [[ -z "${QPS_DIAG_DIR}" ]]; then
  QPS_DIAG_DIR="${PRED_DIR}/qps_diag"
else
  QPS_DIAG_DIR="$(resolve_path "${QPS_DIAG_DIR}")"
fi

if [[ -z "$CHECKPOINT_INPUT" ]]; then
  CHECKPOINT_INPUT="${RUN_DIR}/epoch_${EPOCH}.pth"
else
  CHECKPOINT_INPUT="$(resolve_path "$CHECKPOINT_INPUT")"
fi

if [[ -z "$CHECKPOINT_FIXED" ]]; then
  CHECKPOINT_FIXED="${RUN_DIR}/epoch_${EPOCH}_fix.pth"
else
  CHECKPOINT_FIXED="$(resolve_path "$CHECKPOINT_FIXED")"
fi

echo "================ FF3D 一键流程 ================"
echo "REPO_ROOT          : ${REPO_ROOT}"
echo "CONFIG             : ${CONFIG}"
echo "RUN_DIR            : ${RUN_DIR}"
echo "DATA_ROOT          : ${DATA_ROOT}"
echo "EPOCH              : ${EPOCH}"
echo "TRAIN_GPU          : ${TRAIN_GPU}"
echo "TEST_GPU           : ${TEST_GPU}"
echo "CHECKPOINT_INPUT   : ${CHECKPOINT_INPUT}"
echo "CHECKPOINT_FIXED   : ${CHECKPOINT_FIXED}"
echo "PRED_DIR           : ${PRED_DIR}"
echo "QPS_DIAG           : ${QPS_DIAG}"
echo "QPS_DIAG_DIR       : ${QPS_DIAG_DIR}"
echo "QPS_DIAG_SMALL_TREE_RATIO: ${QPS_DIAG_SMALL_TREE_RATIO}"
echo "QPS_DIAG_REGION_STRIDE: ${QPS_DIAG_REGION_STRIDE}"
echo "PREPARE_DATA       : ${PREPARE_DATA}"
echo "SKIP train/fix/test/eval: ${SKIP_TRAIN}/${SKIP_FIX}/${SKIP_TEST}/${SKIP_EVAL}"
echo "==============================================="

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] 配置文件不存在: ${CONFIG}"
  exit 1
fi

mkdir -p "${RUN_DIR}" "${PRED_DIR}"
if [[ "${QPS_DIAG}" -eq 1 ]]; then
  mkdir -p "${QPS_DIAG_DIR}"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "${PREPARE_DATA}" -eq 1 ]]; then
  echo
  echo "[1/5] 数据预处理"
  BATCH_LOAD_SCRIPT="${DATA_ROOT}/../ForAINetV2/batch_load_ForAINetV2_data.py"
  if [[ ! -f "${BATCH_LOAD_SCRIPT}" ]]; then
    echo "[ERROR] 找不到 batch_load 脚本: ${BATCH_LOAD_SCRIPT}"
    exit 1
  fi
  (
    cd "${DATA_ROOT}"
    python ../ForAINetV2/batch_load_ForAINetV2_data.py
  )
  python tools/create_data_forainetv2.py forainetv2 \
    --root-path "${DATA_ROOT}" \
    --out-dir "${DATA_ROOT}" \
    --extra-tag forainetv2

  python tools/fix_forainetv2_infos.py --root "${DATA_ROOT}"
  for split in train val test; do
    fixed="${DATA_ROOT}/forainetv2_oneformer3d_infos_${split}.pkl.fixed.pkl"
    orig="${DATA_ROOT}/forainetv2_oneformer3d_infos_${split}.pkl"
    if [[ -f "${fixed}" ]]; then
      mv -f "${fixed}" "${orig}"
      echo "  [fix_infos] replaced ${orig}"
    fi
  done
fi

if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
  echo
  echo "[2/5] 训练"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python tools/train.py \
    "${CONFIG}" \
    --work-dir "${RUN_DIR}"
else
  echo
  echo "[2/5] 跳过训练"
fi

if [[ "${SKIP_FIX}" -eq 0 ]]; then
  echo
  echo "[3/5] 修复 checkpoint (spconv)"
  if [[ ! -f "${CHECKPOINT_INPUT}" ]]; then
    echo "[ERROR] 未找到训练输出 checkpoint: ${CHECKPOINT_INPUT}"
    exit 1
  fi
  python tools/fix_spconv_checkpoint.py \
    --in-path "${CHECKPOINT_INPUT}" \
    --out-path "${CHECKPOINT_FIXED}"
else
  echo
  echo "[3/5] 跳过 checkpoint 修复"
fi

if [[ "${SKIP_TEST}" -eq 0 ]]; then
  echo
  echo "[4/5] 推理"
  if [[ ! -f "${CHECKPOINT_FIXED}" ]]; then
    echo "[ERROR] 未找到修复后的 checkpoint: ${CHECKPOINT_FIXED}"
    echo "        可指定 --fixed-checkpoint 或取消 --skip-fix。"
    exit 1
  fi
  TEST_EXTRA_ARGS=()
  if [[ "${QPS_DIAG}" -eq 1 ]]; then
    TEST_EXTRA_ARGS+=(
      --qps-diag
      --qps-diag-dir "${QPS_DIAG_DIR}"
      --qps-diag-small-tree-ratio "${QPS_DIAG_SMALL_TREE_RATIO}"
      --qps-diag-region-stride "${QPS_DIAG_REGION_STRIDE}"
    )
  fi
  FF3D_OUTPUT_PATH="${PRED_DIR}" CUDA_VISIBLE_DEVICES="${TEST_GPU}" python tools/test.py \
    "${CONFIG}" \
    "${CHECKPOINT_FIXED}" \
    --work-dir "${RUN_DIR}" \
    "${TEST_EXTRA_ARGS[@]}"
  if [[ "${QPS_DIAG}" -eq 1 ]]; then
    python tools/summarize_qps_diagnostics.py "${QPS_DIAG_DIR}"
  fi
else
  echo
  echo "[4/5] 跳过推理"
fi

if [[ "${SKIP_EVAL}" -eq 0 ]]; then
  echo
  echo "[5/5] 评测"
  python tools/final_eval.py "${PRED_DIR}"
else
  echo
  echo "[5/5] 跳过评测"
fi

echo
echo "流程完成。"
