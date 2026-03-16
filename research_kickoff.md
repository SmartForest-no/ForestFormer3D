# OneFormer3D ↔ ForestFormer3D（ForInstanceV2）问题梳理与研究启动记录

> 目的：把你的“动机—现象—问题—预期产出”结构化记录下来，作为后续逐步深挖与写作的索引。  
> 注意：本文件只做**理解复述与研究计划**，不展开任何结论性回答（你明确说“还没有让我回答问题”）。

---

## 1) 我理解的背景与动机（科研视角复述）

- **对象与任务**：你关注的是点云 3D 深度学习中，面向**语义分割**与**实例/单体分割**（尤其是“单木分割”）的主流框架与其跨领域迁移能力。
- **两条技术线索**：
  - **OneFormer3D**：当前较主流的点云端到端统一分割框架（可覆盖室内/室外公开数据集上的语义/实例/全景等任务范式）。
  - **ForestFormer3D（FF3D）**：你发现的一篇专门面向遥感森林场景（机载/激光雷达树木点云）的改良框架；它是在 OneFormer3D 的思路/框架上做了针对森林点云的改进。
- **你观察到的关键现象**：
  - 原始/通用架构在你所指的“室内数据集”上表现并不理想（或至少与你预期/论文指标存在差距）。
  - 但在**不改变主要网络结构**的前提下，加入若干“重要且创新的森林处理”后，FF3D 在其提出的 **ForInstanceV2**（机载 LiDAR 树木数据集）上的精度能显著提升。
- **你关心的“更深层”问题**：
  - “从 CV（计算机视觉常见点云数据与假设）迁移到 RS（遥感/激光雷达森林点云）”，到底需要哪些**领域处理**？
  - 哪些处理是**专门作用于森林点云**的？为何能带来显著收益？
  - 你也注意到一个对比：森林树木点云往往**疏密不均、存在遮挡**等挑战；而在 CV 圈点云实例分割即便 SOTA 也常见 mAP 不算很高，但迁移到 RS 树木场景后有时却能得到很高的指标——你希望解释这种“指标与难度直觉不完全一致”的现象。

---

## 2) 你希望我后续回答的核心问题（先列清单，不作回答）

1. **机制层面的“为什么”**：为什么原始通用架构在（你关心的）室内数据集精度低，但在不改变主要网络结构的情况下，通过加入森林相关处理后，在 ForInstanceV2 上能显著提升？
2. **领域迁移的“要做什么”**：以 ForInstanceV2 的机载 LiDAR 树木点云与其语义/单木分割为例，CV→RS 迁移模型需要做哪些领域处理？这些处理分别解决了哪些“数据—任务—评价”层面的差异？
3. **森林专属处理的“有哪些”**：需要系统地找到并梳理“专门作用于森林点云”的处理工作（可能涵盖数据预处理、标注/标签体系、采样、损失/训练策略、推理与后处理等），并解释其作用机理。
4. **为何显著提升（与室内对比）**：为何这些工作在森林点云上能显著提升精度？尤其考虑你提到的疏密不均、遮挡等特点；以及“CV 领域 mAP 不高但 RS 领域可能更高”的现象应如何严谨解读（指标、任务定义、数据分布、难例构成等因素）。
5. **代码与数据集层面的“怎么做”**：
   - ForestFormer3D 仓库具体做了哪些森林处理？分别落在代码的哪些模块/文件？
   - ForInstanceV2 数据集在这个仓库中是如何处理与转化的（从原始 `.ply` 到训练所需格式）？
   - 这些处理如何借鉴到你自己的研究/工程中（可迁移的步骤、需要替换的假设、潜在风险点）。

---

## 3) 本仓库中与“数据处理/领域处理”相关的初步线索（仅做索引）

以下是我在 `ForestFormer3D/` 内初步定位到的“可能与 ForInstanceV2 数据处理、训练/评测链路相关”的入口文件（后续会逐个精读与串起来）：

- 数据与预处理脚本（ForInstanceV2/ForAINetV2）：
  - `ForestFormer3D/data/ForAINetV2/batch_load_ForAINetV2_data.py`
  - `ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py`
  - `ForestFormer3D/tools/create_data_forainetv2.py`
  - `ForestFormer3D/tools/converter_forainetv2.py`
  - `ForestFormer3D/tools/forainetv2_data_utils.py`
- 训练/测试入口与配置：
  - `ForestFormer3D/tools/train.py`
  - `ForestFormer3D/tools/test.py`
  - `ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many.py`
  - `ForestFormer3D/configs/oneformer3d_radius16_qp300.py`
- 模型与数据集定义（实现“森林版 OneFormer3D”差异的主要位置候选）：
  - `ForestFormer3D/oneformer3d/forainetv2_dataset.py`
  - `ForestFormer3D/oneformer3d/oneformer3d.py`（以及同目录的 speedup/without-speedup 版本）
  - `ForestFormer3D/oneformer3d/transforms_3d.py`
- 与“superpoint/分块/几何分割”相关的实现线索：
  - `ForestFormer3D/segmentator/`
- 环境/依赖与使用说明：
  - `ForestFormer3D/readme.md`
  - `ForestFormer3D/replace_mmdetection_files/`（对 mmengine/mmdet3d 的替换补丁，可能影响训练循环与 transforms）

---

## 4) 需要你确认/补充的信息（用于把问题限定得更精确）

为避免后续分析“答非所问”，我需要你确认以下定义与对比口径（你可以先简短回答要点）：

1. 你说的“**室内数据集精度低**”具体指哪个/哪些数据集（例如 ScanNet、S3DIS、ScanNet200 等）？对应的任务是语义、实例还是全景？评价指标是 mIoU / mAP / PQ / 其他？
2. 你希望对比的“**原始架构**”是：
   - OneFormer3D 原论文/原仓库的实现与训练配方；还是
   - ForestFormer3D 仓库里“未加入森林处理”的基线版本；还是
   - 你自己的复现实验设置？
3. ForInstanceV2 在你的研究语境里，最重要的任务优先级是：语义分割 > 单木实例分割 > 其他？是否存在“只评测树”这一类设定？
4. 你希望我在后续回答中更偏向哪类产出形式：
   - **概念与机制解释**（偏论文/综述写法）
   - **代码级溯源**（逐文件指出“森林处理”在哪里、做了什么）
   - **可复现的实验建议**（对照实验设计：去掉某处理、换采样、换后处理等）

### 4.1 你的回复（已确认）

1. **OneFormer3D（CV/室内点云）评测口径**：常测数据集包括 ScanNet / ScanNet200 / S3DIS；覆盖语义、实例、全景任务；语义指标以 mIoU 为主，实例以 mAP 为主。
2. **FF3D（RS/森林点云、单木分割）评测口径**：常用 recall / precision / F1（你希望我解释其与 mAP/mIoU 的差异与可比性）。
3. **你当前的困惑**：你在 FF3D 论文中看到作者对比了“原始 OneFormer3D”与 FF3D 在 ForInstanceV2 上的语义分割与单木分割性能，但你不清楚他们在方法层面到底差在哪里（是结构、训练配方、数据处理、还是评测口径）。
4. **任务难度优先级**：单木分割（实例）显著难于语义分割。
5. **你希望的产出类型**：概念/机制解释、代码级溯源、可复现实验建议都要，但可以分阶段逐步讨论，不要求一次性全部给出。

---

## 5) 后续工作方式（由浅入深的回答路线：先写提纲，不写答案）

为满足“由浅入深”的诉求，后续我会按如下顺序推进并持续更新新的 markdown 记录：

1. 统一术语与对齐任务/指标：明确语义/实例/全景、实例定义、评测口径与数据拆分。
2. 梳理 OneFormer3D 的关键假设与管线（以 FF3D 仓库代码为准，必要时对照原版差异）。
3. 针对 ForInstanceV2：从数据组织与预处理脚本出发，抽取“森林领域处理”的实际落点（每一步：输入/输出/改变了什么统计性质/为何可能有效）。
4. 形成“CV→RS 迁移处理”分类框架，并把 FF3D 的具体实现映射到该框架中。
5. 结合你关心的现象做机制解释与可验证的对照实验建议，最终沉淀成可引用的科研写作稿。

---

## 6) Step 1：任务与指标对齐（先把“可比性”说清楚）

你提出的现象里有一个关键“陷阱”：**室内点云论文常报 mIoU/mAP，而林业/遥感单木分割常报 Precision/Recall/F1**。如果不先对齐口径，就很容易出现“CV 里 mAP 不高，但 RS 里 F1 很高”的表面矛盾。

### 6.1 语义分割：mIoU（点级别）在两域的差异

- **语义分割**本质是点级分类；mIoU 是对每个类的 IoU 做平均（或加权平均），对“类数”和“类不平衡”敏感。
- ForInstanceV2 在本仓库里定义了 3 个语义类：`ground / wood / leaf`（见 `ForestFormer3D/oneformer3d/forainetv2_dataset.py` 与 `ForestFormer3D/configs/*.py` 的 `class_names`）。
- 因此你后续比较室内（多类、多物体）与森林（少类、强先验）时，需要把“**类数差异**”纳入解释框架；否则 mIoU 的高低并不能直接反推任务难度。

### 6.2 单木实例分割：mAP vs P/R/F1（为什么常看起来“更高”）

- **室内实例分割 mAP**（如 ScanNet AP）通常是：在固定 IoU 阈值（0.25/0.5 等）下，对不同置信度阈值扫一条 PR 曲线，再做积分得到 AP（面积）。
- **P/R/F1**通常对应 PR 曲线上的某个“操作点”（operating point）。如果这个点是“最佳 F1 点”，那么数值往往会**比 AP 更直观更高**，但它不是同一个量。
- 本仓库的实例评测实现实际上同时输出了 AP 与（最佳 F1 对应的）Precision/Recall：
  - `ForestFormer3D/oneformer3d/instance_seg_metric.py` → `ForestFormer3D/oneformer3d/instance_seg_eval.py` → `ForestFormer3D/oneformer3d/evaluate_semantic_instance.py`
  - 输出表头为：`AP_0.25 / AP_0.50 / AP / Prec_0.50 / Rec_0.50`
  - 其中 `Prec_0.50` 与 `Rec_0.50` 来自 **IoU=0.5** 下 PR 曲线上**使 F1 最大**的那个点（源码里显式做了 `f1_score.argmax()`）。
  - 因此论文/报告里若只报 P/R/F1，本质上是在“选了一个最优阈值”的口径下汇报；与“积分意义”的 mAP **不可直接横向比较**。
  - F1 可以由 `F1 = 2PR/(P+R)` 从该输出复原（P=Prec_0.50, R=Rec_0.50）。

### 6.3 ForInstanceV2 的“语义 vs 单木实例”在数据标注上的关系（一个关键点）

ForInstanceV2 的 `.ply` 数据在本仓库中读取的字段是：

- 语义：`semantic_seg`
- 实例（单木 ID）：`treeID`

对应实现见 `ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py`。

一个很重要的事实是：**同一棵树的实例 ID（treeID）下，语义标签会同时包含 wood 与 leaf**（地面为 ground）。这会导致“单木实例分割”天然更像一个**单类的树实例分割**，而不是多类实例分割。

- 评测脚本也隐式把“单木实例分割”当成“单类实例分割”在算：`ForestFormer3D/oneformer3d/instance_seg_eval.py:rename_gt` 会在每个实例内检查语义标签，如果一个实例同时出现多个语义类，则取最小语义类作为该实例的类（wood=1，leaf=2 → 取 wood）。
- 我做了一个快速 sanity-check（纯读仓库现成的二进制标注文件）：
  - `ForestFormer3D/data/ForAINetV2/semantic_mask/NIBIO_NIBIO_plot_2_annotated_val.bin` 的语义标签确有 `{0,1,2}`；
  - `ForestFormer3D/data/ForAINetV2/instance_mask/NIBIO_NIBIO_plot_2_annotated_val.bin` 的实例 ID 为 `0..40`，且同一实例覆盖大量 leaf 点；
  - 这与上述“实例=树、语义=部位”的设定一致。

### 6.4 “CV 里 mAP 不高，但 RS 里 P/R/F1 很高”应如何更严谨地解释（第一层答案）

在不引入任何网络结构细节之前，仅从“任务与评测”层面，就已经存在几条足以造成该现象的原因：

1. **指标定义不同**：AP 是对整条 PR 曲线积分；F1 通常对应“挑一个最优阈值”的点。很多情况下最优 F1 会显著大于 AP。
2. **类别数与实例定义不同**：室内实例分割往往是多类、跨物体形态；ForInstanceV2 的“单木实例”更接近单类（树）实例分割，且实例边界有强先验（以树为中心的结构）。
3. **数据组织方式不同**：森林数据常以“样地/plot”切块输入；室内数据是复杂场景的全量重建。切块策略与评价单位会显著影响难度与指标。

后续我们再把第二层、第三层原因补齐：即 FF3D 的“森林处理”如何改变点云统计特性（疏密不均、遮挡、尺度、地形起伏、垂直结构先验等），以及这些变化如何让同一套主干网络更容易学到稳定的决策边界。

---

## 7) Step 2：FF3D 论文里“原始 OneFormer3D vs FF3D”到底差在哪里？（以本仓库为准的第一版溯源）

你提到的困惑（论文里“原始 OneFormer3D”对比 FF3D，但你不清楚差异）在代码层面可以先做一个**最小可解释**的定位：这个仓库里确实同时存在“更像 OneFormer3D 基线”的实现与“FF3D 提升版”的实现，并且配了两份对照配置文件。

### 7.1 两条实现/两份配置（最直接的对照入口）

- **基线（更接近 OneFormer3D 的 query-decoder + one-to-one matching）**
  - 模型类：`ForestFormer3D/oneformer3d/oneformer3d.py` 中的 `ForAINetV2OneFormer3D`
  - 配置：`ForestFormer3D/configs/oneformer3d_radius16_qp300.py`
- **FF3D 版本（X-Aware Query + one-to-many matching 等改动）**
  - 模型类：`ForestFormer3D/oneformer3d/oneformer3d.py` 中的 `ForAINetV2OneFormer3D_XAwarequery`
  - 配置：`ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many.py`

这意味着：**论文里所谓“原始 OneFormer3D vs FF3D”很可能对应的就是这两套配置/实现**（但为了严谨，我们后续仍需要对照论文实验设置确认：是否还包含训练轮数、阈值、数据切块等差异）。

### 7.2 从配置就能看出的关键差异（不需要读完代码）

1. **实例 query 的来源不同**
   - 基线：`num_instance_queries=300`（显式的 instance queries）
   - FF3D：`num_instance_queries=0` 且引入 `query_point_num=300`（更像“从点/体素中选 query”的机制）
2. **匹配与训练信号不同**
   - 基线：`HungarianMatcher`（典型 one-to-one）
   - FF3D：`One2ManyMatcher`（one-to-many 的训练机制，通常会增加正样本匹配、改善召回/收敛）
3. **额外的“森林场景友好”辅助头/损失（用于选点与区分前景）**
   - 在 `ForAINetV2OneFormer3D_XAwarequery` 里新增了 `Embed` 与 `BiSemantic` 分支，并在 `loss()` 里显式计算：
     - `discriminative_loss`（对实例可分性/聚类友好）
     - `semantic_loss_bi`（体素级前景/背景二分类，服务于“哪里值得采样 query”）
   - 同时引入 `prepare_epoch`：训练前期/后期启用不同策略（相当于一个 staged training）。

> 直观理解：你看到的“主干结构不大改、但精度显著提升”，很可能来自这类**数据/采样/匹配/训练信号**的系统性重设计，而不只是换个 backbone。

### 7.3 需要进一步核实的点（把“论文口径差异”拆开）

为了把你的困惑彻底说清楚，我们下一步会把差异拆成三层，并分别用代码证据定位：

1. **必要的 RS 数据适配**（即便做基线也必须做）：例如大尺度点云的切块/采样/推理拼接。
2. **FF3D 的核心创新**：X-aware query、one-to-many matching、辅助分支与 staged training 等（上面已经定位到落点）。
3. **评测与阈值口径**：尤其是实例分割 P/R/F1 是不是取最优阈值点、以及 IoU 阈值与匹配策略如何设定（见第 6 节）。

如果你方便，把 FF3D 论文里那张“OneFormer3D vs FF3D”的对比表（或表格编号/截图文字）发我，我可以把“论文描述的基线”与“本仓库这两份配置”逐项一一对齐，避免我们在基线定义上走偏。

---

## 8) 下一步（Step 3 预告）：ForInstanceV2 在仓库里是怎么被处理成可训练数据的？

下一节我会按“原始数据 → 训练输入”的链路，把 ForInstanceV2/ForAINetV2 的领域处理逐步拆开，并标注每一步对应的代码位置与它解决的 RS 痛点（尺度、疏密、遮挡、地形、切块边界等）：

1. **从 `.ply` 读取语义/实例字段与坐标归一化**：`ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py`
2. **批量导出为训练用的 `points/semantic_mask/instance_mask` 二进制文件**：`ForestFormer3D/data/ForAINetV2/batch_load_ForAINetV2_data.py`
3. **生成 info pkl 与数据集索引**：`ForestFormer3D/tools/create_data_forainetv2.py`、`ForestFormer3D/tools/converter_forainetv2.py`、`ForestFormer3D/tools/forainetv2_data_utils.py`
4. **训练/验证管线里的“森林切块与采样”**：`ForestFormer3D/configs/*.py`（`CylinderCrop / GridSample / PointSample_ / PointInstClassMapping_`）
5. **推理阶段的大尺度拼接策略**：`ForestFormer3D/oneformer3d/oneformer3d.py`（test 时的滑窗圆柱区域与最近邻映射等）

你可以直接告诉我：下一节你更想先从 “数据预处理” 还是从 “训练管线/推理拼接” 开始拆。

---

## 9) Step 3：ForInstanceV2/ForAINetV2 数据预处理链路（离线 → 可训练格式）

这一节只做一件事：把 **ForInstanceV2（仓库内命名为 ForAINetV2）** 从“原始 `.ply`”变成训练/评测可直接读取的 `points/*.bin + masks/*.bin + infos.pkl`，并指出每一步背后的 RS 领域动机（先验/数值稳定性/尺度适配）。

### 9.1 数据在仓库里的组织（你后续复现/换数据时需要对齐这个结构）

以配置里的 `data_root_forainetv2 = 'data/ForAINetV2/'` 为根目录，训练/评测实际依赖的关键目录/文件是：

- `meta_data/train_list.txt`、`meta_data/val_list.txt`、`meta_data/test_list.txt`：样本列表（每行一个 sample_id，不含扩展名）
- `train_val_data/`、`test_data/`：原始 `.ply`（README 里的“把 ply 放到这两个目录”说的就是它）
- `forainetv2_instance_data/`：中间产物（从 `.ply` 解析出的 `.npy`，包含 offsets 等）
- `points/`：训练实际读入的点坐标（二进制 `.bin`）
- `semantic_mask/`、`instance_mask/`：训练/评测读入的语义/实例标注（二进制 `.bin`）
- `forainetv2_oneformer3d_infos_{train,val,test}.pkl`：mmdet3d 风格的索引文件（每个 sample 的路径与元信息）

> 你在本机路径看到的是 `ForestFormer3D/data/ForAINetV2/…`；在 Docker 里作者默认映射到 `/workspace/data/ForAINetV2/…`。

### 9.2 原始 `.ply` 里需要的字段（语义/单木实例的“定义”从这里开始）

`ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py` 里读 `.ply` 的关键字段是：

- 坐标：`x, y, z`
- 语义：`semantic_seg`（后续会做 `-1`）
- 实例（单木 ID）：`treeID`

读 ply 的底层实现来自 `ForestFormer3D/data/ForAINetV2/plyutils.py`，它**要求 ply 是 binary 格式**（不是 ascii）。

### 9.3 坐标归一化（森林 LiDAR 常见“巨大的绝对坐标/高程基准”问题）

在 `export()` 中（`ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py`）做了一个非常 RS 取向的处理：

- 对大多数文件：计算 `offsets = [mean_x, mean_y, min_z]`，然后把点云平移到局部坐标系：
  - `x ← x - mean_x`（消除大地坐标系下的巨大平移量）
  - `y ← y - mean_y`
  - `z ← z - min_z`（把最低点对齐到 0，等价于一种“地面基准”）
- 对文件名包含 `bluepoints` 的样本：`offsets = 0`（按作者逻辑不做上述归一化；这更像一个工程分支，后续我们需要结合数据集说明再确认其含义）

这一步的作用（第一性原理）：

1. **数值稳定性**：避免坐标值过大导致网络/体素化/邻域计算出现数值不稳定。
2. **平移不变性**：让模型更聚焦几何形状而非绝对地理位置。
3. **便于输出回投**：offsets 会被保存下来，推理后可以加回去恢复到原始坐标系（见 `ForestFormer3D/tools/merge_prediction.py`）。

### 9.4 标签变换（把“部位语义”与“树实例”对齐成可训练的监督信号）

`export()` 里对标签做了两件关键事：

1. **语义从 `semantic_seg` 转为 `label_ids = semantic_seg - 1`**  
   结合本仓库的 `class_names = ['ground','wood','leaf']`，可以理解为把语义映射到 `0/1/2`。
2. **实例来自 `treeID`，但对地面点强制设为背景**  
   - 先把地面（`bg_sem = [0]`）上的 `instance_ids` 设为 `-1`
   - 最终把 `-1` 再改回 `0`（所以背景实例 ID 在落盘文件里是 `0`）

另外还有一个值得你注意的小设计：作者刻意**不把实例 ID re-index 成连续编号**（代码里那段 “make them continuous” 被注释掉了）。这意味着他们相信 `treeID` 在每个样本内部已经是合理的局部编号，或者后续训练/评测不会严格依赖“连续性”。

### 9.5 中间产物（`forainetv2_instance_data/*.npy`）到底有什么

`ForestFormer3D/data/ForAINetV2/batch_load_ForAINetV2_data.py` 会对每个 sample 输出一组 `.npy`（写在 `forainetv2_instance_data/`）：

- `*_vert.npy`：归一化后的点坐标（float32）
- `*_offsets.npy`：刚才提到的 offsets（用于推理后恢复坐标）
- `*_sem_label.npy`：语义标签（int64）
- `*_ins_label.npy`：实例标签（int64，背景为 0）
- `*_aligned_bbox.npy` / `*_unaligned_bbox.npy` / `*_axis_align_matrix.npy`：bbox 与对齐矩阵（更多是为了复用 mmdet3d 的数据结构；本配置里训练并不读 bbox）

### 9.6 训练直接读取的 `.bin` 是怎么来的（以及 pkl 索引如何生成）

这一步在 `ForestFormer3D/tools/create_data_forainetv2.py` 串起来：

1. `ForestFormer3D/tools/converter_forainetv2.py:create_info_file()` 构建一个 `ForAINetV2Data`（见 `ForestFormer3D/tools/forainetv2_data_utils.py`），它会：
   - 从 `*_vert.npy` 读点，写入 `points/{sample_id}.bin`
   - 从 `*_sem_label.npy` / `*_ins_label.npy` 读标注，写入 `semantic_mask/{sample_id}.bin` 与 `instance_mask/{sample_id}.bin`
   - 生成每个 sample 的 `info` 字典（相对路径 + 元信息）
2. `update_pkl_infos()`（`ForestFormer3D/tools/update_infos_to_v2.py`）再把 infos pkl 升级到 mmdet3d v2 结构。

到这里你就得到了 config 里直接引用的三份索引文件：`forainetv2_oneformer3d_infos_{train,val,test}.pkl`。

### 9.7 数据集类如何接入 mmdet3d（为什么作者能“少写很多代码”）

`ForestFormer3D/oneformer3d/forainetv2_dataset.py` 里 `ForAINetV2SegDataset_` 直接继承 `mmdet3d.datasets.scannet_dataset.ScanNetDataset`，复用了 ScanNet 的数据加载与标注解析流程，仅通过 `METAINFO` 指定：

- 语义类：`ground / wood / leaf`
- 有效类 id：`(0,1,2)`

这也解释了为什么很多脚本注释里写着 “Modified from ScanNet / VoteNet”：作者把 ForInstanceV2 适配进了一条成熟的室内点云数据管线。

---

## 10) Step 4：在线训练管线（CylinderCrop/GridSample/PointSample_）与“森林专属监督”

离线预处理把数据“变成能读”；在线管线则把数据“变成适合学”。ForInstanceV2 的关键 RS 处理很大一部分发生在 pipeline transforms 中（见 `ForestFormer3D/configs/oneformer3d_*.py`）。

### 10.1 训练时为什么用圆柱切块（CylinderCrop），而不是室内常见的立方体 crop

`ForestFormer3D/oneformer3d/transforms_3d.py:CylinderCrop` 做的事不只是 crop：

1. **随机选中心点**（从点云里随机取一个点）
2. **按 XY 半径取圆柱**：保留满足 `(x-xc)^2 + (y-yc)^2 < r^2` 的点（Z 不裁剪）
3. **定义“树前景”**：`instance_mask = (pts_semantic_mask != 0)`（地面=背景）
4. **对背景点把实例置为 -1**（训练内部背景用 -1 更方便 one-hot）
5. **计算 `ratio_inspoint`：裁剪后该实例点数 / 原始该实例点数**

这一条 `ratio_inspoint` 非常关键：它会被用于“裁剪导致的 IoU 偏置修正”（下一小节）。

> 直觉解释：森林里的对象（树）是“竖直延展”的，圆柱 crop 在 XY 上控制尺度、在 Z 上保留完整高度，更符合树木几何与林分空间分布；也更接近机载 LiDAR 的采样方式（高度方向信息密集，水平面上分布广、尺度大）。

### 10.2 裁剪带来的“局部真值”怎么不伤害训练：IoU with crop 修正

一旦做随机 crop，很多树实例会被切成“半棵树”。如果直接用裁剪后的 mask 去算 IoU，会系统性低估目标质量，从而影响 objectness/score 分支的学习。

作者在 `ForestFormer3D/oneformer3d/instance_criterion.py:get_iou_with_crop()` 里显式修正了这个问题：

- 令 `ratio = (#points_in_crop) / (#points_in_full_instance)`
- 用 `targets.sum()/ratio` 近似“全实例点数”，从而把 union 改写为：
  - `union = targets.sum()/ratio + pred.sum - intersection`

这意味着：**即便你只看到半棵树，只要你把半棵树预测得很准，objectness 也不会被“缺失的那半棵”无端惩罚**。这类修正对“树冠被遮挡、树干缺失、密度不均导致局部缺采样”的 RS 场景尤其重要。

### 10.3 GridSample：把疏密不均“拉平”到更可学习的尺度

`ForestFormer3D/oneformer3d/transforms_3d.py:GridSample(grid_size=0.2)` 做体素级下采样（每个 voxel 随机保留一个点），主要解决：

- 同一 plot 内近地/远地、树冠/树干、遮挡区的密度差异；
- 机载 LiDAR 的条带/扫描角导致的非均匀采样；
- 让后续 sparse conv / Minkowski 的体素化更稳定（不会被极端密集区域主导）。

### 10.4 PointSample_：把输入点数固定到超大上限（640k）

`PointSample_(num_points=640000)` 用随机采样把点数裁到固定上限。这个数明显比很多室内设置更大，含义是：

- 树木实例（尤其树冠）需要更高的点覆盖才能稳定分割；
- RS 的 plot 场景在空间尺度上更大，保留更多点能减少“切块边界效应”。

### 10.5 训练 batch 里避免“空样本”：SkipEmptyScene_

`SkipEmptyScene_` 会丢弃没有有效实例的 crop（例如 crop 到纯地面区域）。这对森林尤其重要：样地里“空地/低植被/扫描空洞”更常见，直接喂给实例分割会造成大量无意义梯度。

---

## 11) Step 5：推理阶段的大尺度拼接与森林后处理（滑窗圆柱 + 投票 + 地面先验）

ForInstanceV2 的 test split 往往是“超大点云”。作者没有把滑窗写在 dataset/pipeline，而是直接写进了模型 `predict()`（见 `ForestFormer3D/oneformer3d/oneformer3d.py`）。

### 11.1 为什么 test_pipeline 里不做 crop：因为模型内部要做“滑窗+融合”

在 config 的 `test_pipeline` 里只做加载与打包；真正的大场景处理发生在：

- `ForAINetV2OneFormer3D.predict()`（基线版）
- `ForAINetV2OneFormer3D_XAwarequery.predict()`（FF3D 版）

两者共同的核心逻辑：

1. 在 XY 平面生成一系列圆柱窗口中心（`generate_cylindrical_regions`）
2. 对每个窗口：
   - 取圆柱内点 `pc1`
   - `grid_sample` 降采样到更均匀密度（`pc2`）
   - 如仍过大，再随机采样到 `num_points=640000`（`pc3`）
   - 跑一次网络得到该窗口的语义/实例预测
3. 把窗口预测回投到全局点云，并做融合（语义投票、实例 NMS/合并）

### 11.2 语义融合：从“每窗一个预测”到“全局一致的部位语义”

- 基线版用 `all_pre_sem: List[List[int]]` 收集每个点在不同窗口下被预测的语义标签，最后用多数投票 `finalize_semantic_labels()` 得到最终语义。
- FF3D 版把它优化成 `votes_counter[N_points, N_classes]` 的计数矩阵，最终 `argmax` 得到语义（更省内存、更快）。

### 11.3 实例融合：窗口之间会产生重复树，需要 score-based 合并

基线版的思路可以概括成：

- 对每个窗口的每个实例 mask（过滤低分）：
  - 用最近邻把 mask 从 `pc3 → pc1` 回投
  - 对于同一个全局点，只保留“score 更高”的实例归属（`global_instance_scores`）
- 全部窗口结束后：
  - 删除地面点上的实例（ground mask → instance=-1）
  - 删除点数过少的小实例（<10）
  - 对保留下来的候选实例集合做一次“按 overlap 比例 + score 的 NMS”（`merge_overlapping_instances_by_score*`）
  - 最后把实例 id 重编号成连续的 `0..K-1`

这相当于把“窗口重复检测”问题，转写成一个**基于点集重叠的实例级 NMS**。

### 11.4 一个很典型的森林先验：树实例应该“触地”

在 `pred_inst_sem_test()`（见 `ForestFormer3D/oneformer3d/oneformer3d.py`）里还有一个非常 RS/林业的后处理：

- 先从语义预测里找地面点，计算 `ground_z_max`
- 若某个实例 mask 内的点 `min_z > ground_z_max + 5`（高于地面太多），就把该实例的 score 置 0（等价于删除）

这个规则的直觉是：**单木实例应该包含树干并与地面相连**；只在树冠上“飘着”的实例更像是误检/碎片。它可能显著提高 precision/F1，但也可能在“树干缺失/严重遮挡”的样本上降低 recall——这正是后续对照实验值得验证的点。

### 11.5 offsets 的回投：从局部坐标恢复到原始地理坐标

推理输出（例如 `.ply/.las`）若要回到原始坐标，需要把离线预处理保存的 offsets 加回去：

- `ForestFormer3D/tools/merge_prediction.py` 中会读取 `forainetv2_instance_data/*_offsets.npy` 并把坐标加回去再导出 `.las`。

---

## 12) 把 7.3 的“三层差异”落到实际处理点（第一版映射表）

| 层级 | 你关心的本质问题 | 在本仓库里对应的“处理点” | 典型代码位置 |
|---|---|---|---|
| RS 必要适配 | “室内假设”为什么不适用于大尺度森林点云？ | 坐标归一化+offset、超大场景滑窗、密度均衡、语义/实例定义（树实例+部位语义） | `ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py`、`ForestFormer3D/oneformer3d/transforms_3d.py`、`ForestFormer3D/oneformer3d/oneformer3d.py` |
| FF3D 核心创新 | 在不改主干时，哪些机制真正带来提升？ | X-aware query（从预测的“树区域”采样 query）、one-to-many matching、discriminative + 二分类辅助头、staged training | `ForestFormer3D/oneformer3d/oneformer3d.py`、`ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many.py` |
| 评测/阈值口径 | 为什么 RS 的 P/R/F1 常看起来更高？ | AP vs 最优 F1 点；ForAINetV2 的实例被“折叠成 tree 单类”；融合/后处理阈值会强影响 P/R | `ForestFormer3D/oneformer3d/evaluate_semantic_instance.py`、`ForestFormer3D/oneformer3d/instance_seg_eval.py`、`ForestFormer3D/oneformer3d/unified_metric.py` |

下一步我建议我们就从这张表出发：你挑 1–2 个你认为“最可能解释精度跃迁”的处理点（比如 *裁剪 IoU 修正*、*触地先验过滤*、*X-aware query*），我们做“机制解释 + 可复现实验拆解”。

---

## 13) 实操：用预训练 FF3D 对“自带 ply”做仅推理（不污染原 test_data）

> 场景：你有一份 ALS 点云，已转为 `.ply`，希望仅推理；同时不改/不删除仓库原始 `data/ForAINetV2/test_data/`。

### 13.1 关键事实（先讲清楚再动手）

- 本仓库的 ForAINetV2 预处理脚本默认读取 `.ply` 中的字段：`x, y, z, semantic_seg, treeID`（见 `ForestFormer3D/data/ForAINetV2/load_forainetv2_data.py`）。
- 所以 **缺 `semantic_seg` 会导致预处理直接报错**。仅推理时它并不需要真实 GT，但我们仍要提供一个 *dummy* 的 `semantic_seg` 才能走通数据管线。
- 这个仓库的推理代码会用 `if 'test' in lidar_path` 决定是否走“滑窗圆柱 + 融合”的 test 分支；只有 test 分支会在 `output_path` 下保存最终 `.ply` 预测文件。

### 13.2 把你的 ply 补齐 `semantic_seg`（两种办法，先用最简单的）

**推荐（最快）：在 CloudCompare 里加一个常数标量场并命名为 `semantic_seg`**

- 新增 scalar field，给所有点赋常数（建议：`semantic_seg=2`，代表 wood；因为脚本内部会做 `semantic_seg - 1` 映射到 `0/1/2`）。
- 确保 `treeID` 标量场名字严格是 `treeID`（大小写敏感）。
- 导出为 **binary PLY**，并勾选导出全部 scalar fields。

> GT 语义对仅推理没意义，所以 `semantic_seg` 设常数即可；地面/树冠/树干的区分会由模型预测完成。

### 13.3 为你的自定义样本建立隔离的数据根目录（不动原 ForAINetV2）

在容器内（README 默认 `/workspace`），创建新目录结构：

```bash
/workspace/data/ForAINetV2_custom/
  meta_data/
  test_data/
```

然后：

1. 把你的 `sample.ply` 拷到：`/workspace/data/ForAINetV2_custom/test_data/`
2. 写列表文件：
   - `/workspace/data/ForAINetV2_custom/meta_data/test_list.txt`：写一行 `sample`（不带 `.ply`）
   - `/workspace/data/ForAINetV2_custom/meta_data/train_list.txt`：留空文件即可
   - `/workspace/data/ForAINetV2_custom/meta_data/val_list.txt`：留空文件即可

### 13.4 运行预处理（只针对 custom 根目录）

在容器内执行：

```bash
cd /workspace/data/ForAINetV2_custom
python /workspace/data/ForAINetV2/batch_load_ForAINetV2_data.py \
  --train_forainetv2_dir train_val_data \
  --test_forainetv2_dir test_data \
  --output_folder ./forainetv2_instance_data \
  --train_scan_names_file meta_data/train_list.txt \
  --val_scan_names_file meta_data/val_list.txt \
  --test_scan_names_file meta_data/test_list.txt
```

这一步会在 custom 根目录下生成 `forainetv2_instance_data/*_vert.npy/_sem_label.npy/_ins_label.npy/_offsets.npy` 等中间文件。

接着在 `/workspace` 下生成 `.bin` 与 infos pkl（仍写入 custom 根目录）：

```bash
cd /workspace
python tools/create_data_forainetv2.py forainetv2 \
  --root-path data/ForAINetV2_custom \
  --out-dir  data/ForAINetV2_custom \
  --extra-tag forainetv2
```

### 13.5 运行仅推理（指向 custom data_root，checkpoint 用预训练）

假设预训练模型在：`/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth`，运行：

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --cfg-options \
    test_dataloader.dataset.data_root=data/ForAINetV2_custom/ \
    val_dataloader.dataset.data_root=data/ForAINetV2_custom/ \
    train_dataloader.dataset.data_root=data/ForAINetV2_custom/
```

> 提醒：`ForAINetV2OneFormer3D_XAwarequery.predict()` 里 `output_path="/workspace/work_dirs/V3"` 是硬编码的；在不改代码的前提下，建议你保证 sample_id 唯一，避免覆盖同名输出。

### 13.6 输出在哪里看（你真正关心的“单木分割结果”）

在 XAwarequery 的 test 分支中，会把最终结果写成一个 `.ply`（包含 `semantic_pred/instance_pred/score` 字段），路径由硬编码的 `output_path` 决定（默认 `/workspace/work_dirs/V3`）。

如果你的 sample 路径不含 `test`（例如目录名/文件名里没有 `test`），就可能走“非 test 分支”，从而不会写上述融合后的 `.ply`；这种情况下可以通过把数据放进包含 `test` 字样的目录（例如 `test_data_x/`）或给文件名加 `_test` 来触发 test 分支。
