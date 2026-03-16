# FF3D 推理/训练流程（ForAINetV2 & Custom）

## 一键脚本（Bluepoints 之前）

> 适用范围：训练 -> 修复 checkpoint -> 推理 -> 评测。

sudo chown -R ubuntu22:ubuntu22 /home/ubuntu22/projects/ForestFormer3D

sudo docker start forestformer3d-container
sudo docker exec -it forestformer3d-container /bin/bash

wsl系统中，而不是镜像
export WANDB_API_KEY='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'
wandb login --relogin "$WANDB_API_KEY"
export WANDB_DISABLE_OFFLINE_ARTIFACTS=1

本地wandb最后上传
wandb sync <offline-run-path>
wandb sync /home/ubuntu22/projects/ForestFormer3D/work_dirs/run3_qps_ogrd/20260313_165709/vis_data/wandb/offline-run-20260313_165714-kpt5iv81

```bash
cd /workspace
bash tools/run_train_infer_eval.sh \
  --config configs/oneformer3d_qs_radius16_qp300_2many_ogrd.py \
  --run-dir work_dirs/run3_qps_ogrd_1000+300 \
  --data-root data/ForAINetV2 \
  --epoch 1300 \
  --train-gpu 1 \
  --test-gpu 1

bash tools/run_train_infer_eval.sh \
  --config configs/oneformer3d_qs_radius16_qp300_2many_tcfps.py \
  --run-dir work_dirs/run3_qps_tcfps_1000+300 \
  --data-root data/ForAINetV2 \
  --epoch 1300 \
  --train-gpu 0 \
  --test-gpu 0

cd /workspace
bash tools/run_train_infer_eval.sh \
  --config configs/oneformer3d_radius16_qp300.py \
  --run-dir work_dirs/run1_baseline_oneformer3d \
  --data-root data/ForAINetV2 \
  --epoch 3000 \
  --train-gpu 0 \
  --test-gpu 0

```

如需先重做数据预处理（对应第 3.2 节），添加 `--prepare-data`。

仅重跑推理+评测：

```bash
bash tools/run_train_infer_eval.sh --skip-train --skip-fix --epoch 1300
```

---

## 0) 管理员权限（宿主机）

```bash
sudo chown -R ubuntu22:ubuntu22 /home/ubuntu22/projects/ForestFormer3D
```

---

## 1) 启动 Docker（若未启动）

```bash
sudo service docker start
sudo service docker status
```

容器已存在但停止：

```bash
sudo docker start forestformer3d-container
sudo docker exec -it forestformer3d-container /bin/bash
```

首次创建容器：

```bash
cd /home/ubuntu22/projects/ForestFormer3D
sudo docker build -t forestformer3d-image .
sudo docker run --gpus all --shm-size=128g -d -p 127.0.0.1:49211:22 \
  -v /home/ubuntu22/projects/ForestFormer3D:/workspace \
  -v /home/ubuntu22/projects/ForestFormer3D/segmentator:/workspace/segmentator \
  --name forestformer3d-container forestformer3d-image
sudo docker exec -it forestformer3d-container /bin/bash
```

> 容器内仓库路径：`/workspace`

---

## 2) 原始 ForAINetV2（预训练验证）

### 2.1 数据集结构

```text
/workspace/data/ForAINetV2/
  meta_data/
    train_list.txt
    val_list.txt
    test_list.txt
  train_val_data/
    *.ply
  test_data/
    *.ply
```

### 2.2 生成中间数据（一次性）

```bash
cd /workspace/data/ForAINetV2
pip install laspy "laspy[lazrs]"
python batch_load_ForAINetV2_data.py

cd /workspace
python tools/create_data_forainetv2.py forainetv2 \
  --root-path data/ForAINetV2 \
  --out-dir  data/ForAINetV2 \
  --extra-tag forainetv2

python tools/fix_forainetv2_infos.py --root /workspace/data/ForAINetV2
```

### 2.3 推理 / 测试（预训练模型）

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/clean_forestformer/epoch_3000_fix.pth
```

---

## 3) Custom 数据集（Hainan / Run2）

### 3.1 使用 custom_test 软连接

```text
/workspace/data/ForAINetV2_custom_test -> /workspace/data/ForAINetV2_custom
```

### 3.2 重新生成中间数据（保证路径一致）

```bash
cd /workspace/data/ForAINetV2_custom_test
python ../ForAINetV2/batch_load_ForAINetV2_data.py

cd /workspace
python tools/create_data_forainetv2.py forainetv2 \
  --root-path data/ForAINetV2_custom_test \
  --out-dir  data/ForAINetV2_custom_test \
  --extra-tag forainetv2

python tools/fix_forainetv2_infos.py --root /workspace/data/ForAINetV2_custom_test
```

---

## 4) 训练（Run2 / Hainan）

```bash
cd /workspace
export PYTHONPATH=/workspace
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many_ogrd.py \
  --work-dir work_dirs/run3_qps_ogrd


cd /workspace
export PYTHONPATH=/workspace
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many_tcfps.py \
  --work-dir work_dirs/run3_qps_tcfps

```

修复 checkpoint（spconv 兼容）：

```bash
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/run3_qps_tcfps/epoch_3000.pth \
  --out-path work_dirs/run3_qps_tcfps/epoch_3000_fix.pth

```

---

## 5) 测试（Run2 / Hainan）

确保输出目录存在：

```bash
mkdir -p /workspace/work_dirs/run3_qps_ogrd/pred_data
```

现在 `tools/test.py` 默认开启 QPS 诊断。常规推理只需要手动指定 `FF3D_OUTPUT_PATH`；其余诊断环境变量会自动补齐：

```bash
FF3D_OUTPUT_PATH=/workspace/work_dirs/run3_qps_tcfps/pred_data \
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many_tcfps.py \
  work_dirs/run3_qps_tcfps/epoch_3000_fix.pth \
  --work-dir work_dirs/run3_qps_tcfps

FF3D_OUTPUT_PATH=/workspace/work_dirs/run0_pretrained_zijian/pred_data \
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --work-dir work_dirs/run0_pretrained_zijian
```

上面这条命令默认等价于显式传入下面这些诊断参数：

```bash
mkdir -p /workspace/work_dirs/run3_qps_ogrd/pred_data/qps_diag

FF3D_OUTPUT_PATH=/workspace/work_dirs/run3_qps_ogrd/pred_data \
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many_ogrd.py \
  work_dirs/run3_qps_ogrd/epoch_3000_fix.pth \
  --work-dir work_dirs/run3_qps_ogrd \
  --qps-diag \
  --qps-diag-dir /workspace/work_dirs/run3_qps_ogrd/pred_data/qps_diag \
  --qps-diag-small-tree-ratio 0.3333333333 \
  --qps-diag-region-stride 1
```

参数含义：

- `--qps-diag`：开启 ISA-FPS / QPS 诊断导出；现在默认开启，仅用于显式声明。
- `--qps-diag-dir`：诊断 JSON 输出目录；默认是 `FF3D_OUTPUT_PATH/qps_diag`。
- `--qps-diag-small-tree-ratio 0.3333333333`：把“高度不超过当前样地最高树高 1/3 的树”定义为小树。
- `--qps-diag-region-stride 1`：每个 region 都记录一次诊断；如果写成 `2`，就是每隔 1 个 region 记录一次。

如果只想关闭诊断，改成：

```bash
FF3D_OUTPUT_PATH=/workspace/work_dirs/run3_qps_ogrd/pred_data \
CUDA_VISIBLE_DEVICES=1 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many_ogrd.py \
  work_dirs/run3_qps_ogrd/epoch_3000_fix.pth \
  --work-dir work_dirs/run3_qps_ogrd \
  --no-qps-diag

FF3D_OUTPUT_PATH=/workspace/work_dirs/run1_baseline_oneformer3d/pred_data \
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_radius16_qp300.py \
  work_dirs/run1_baseline_oneformer3d/epoch_3000_fix.pth \
  --work-dir work_dirs/run1_baseline_oneformer3d \
  --no-qps-diag
```

需要汇总诊断时再执行：

```bash
python tools/summarize_qps_diagnostics.py \
  /workspace/work_dirs/run3_qps_ogrd/pred_data/qps_diag
```

历史实验 `run3_qps_ogrd_1000+300` 若要显式写全，命令如下：

```bash
FF3D_OUTPUT_PATH=/workspace/work_dirs/run3_qps_ogrd_1000+300/pred_data \
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many_ogrd.py \
  work_dirs/run3_qps_ogrd_1000+300/epoch_1300_fix.pth \
  --work-dir work_dirs/run3_qps_ogrd_1000+300 \
  --qps-diag \
  --qps-diag-dir /workspace/work_dirs/run3_qps_ogrd_1000+300/pred_data/qps_diag \
  --qps-diag-small-tree-ratio 0.3333333333 \
  --qps-diag-region-stride 1

python tools/summarize_qps_diagnostics.py \
  /workspace/work_dirs/run3_qps_ogrd_1000+300/pred_data/qps_diag
```

诊断会按 scene 输出 `*_qps_diag.json`，其中包含：

- `bi_semantic`：树体素候选的 precision / recall / instance coverage
- `query_selection`：ISA-FPS/OGRD/TCFPS 选点后的实例覆盖率、小树覆盖率、零查询实例比例、中心偏移和高度分布
  - 小树定义：`instance_height <= 当前样地最高树高 * ratio`，默认 `ratio = 1/3`
- `decoder`：decoder 内部保留下来的 query 覆盖率，以及外部 score / edge filter 后的覆盖率

一键脚本也支持同样的诊断开关：

```bash
bash tools/run_train_infer_eval.sh \
  --config configs/oneformer3d_qs_radius16_qp300_2many_ogrd.py \
  --run-dir work_dirs/run3_qps_ogrd_1000+300 \
  --epoch 1300 \
  --skip-train --skip-fix \
  --qps-diag
```

---

## 6) 评测（Run2 / Hainan）

总体评测：

```bash
python tools/final_eval.py /workspace/work_dirs/run3_qps_tcfps/pred_data

python tools/final_eval.py /workspace/work_dirs/oneformer3d_radius16_qp300_e2675_test_bm1_austrian
```

分桶评测（全局分位数 + 三指标）：

```bash
python tools/diagnose_instances.py /workspace/work_dirs/run1_baseline_1000+300/pred_data \
  --out-dir /workspace/work_dirs/run1_baseline/diag_all

```

---

## 7) 二次推理（Bluepoints）

1) 切换到 `oneformer3d_copy_bluepoints.py`：

```bash
sed -i 's/from \\.oneformer3d import/from \\.oneformer3d_copy_bluepoints import/' /workspace/oneformer3d/__init__.py
```

切回原版（完成二次推理后）：

```bash
sed -i 's/from \\.oneformer3d_copy_bluepoints import/from \\.oneformer3d import/' /workspace/oneformer3d/__init__.py
```

2) 准备测试列表：

```text
/workspace/data/ForAINetV2_custom_test/meta_data/test_list_initial.txt
```

3) 确保初始 test_data 只有原始 PLY（清理旧 bluepoints）：

```bash
rm -f /workspace/data/ForAINetV2_custom_test/test_data/*_bluepoints_*.ply
```

4) 运行二次推理脚本：

```bash
bash /workspace/tools/inference_bluepoint.sh
```

输出目录：

```text
/workspace/work_dirs/run2_overfit_bluepoints
```
