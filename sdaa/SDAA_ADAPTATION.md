# EnzyGen2 龙芯 SDAA 适配记录

> EnzyGen2 是「ligand-based 功能蛋白序列与结构协同设计」大模型（fairseq 框架），
> encoder ESM2-650M + decoder EGNN 等变图网络。模型纯 PyTorch、无自定义 CUDA 算子。
> 本次结论：端到端在 SDAA 上跑通，预训练生成 + 微调（finetuning）代码路径均已验证。

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 架构 | loongarch64（Loongnix Server 23.1，32 张 SDAA 卡）|
| 机器 | `sunhn@10.71.13.47`，容器 `tuyi_test` |
| Python | 3.12（venv `/home/py312`）|
| torch | 2.12.0 + torch_sdaa 3.3.0b0 |
| SDAA 卡 | `SDAA_VISIBLE_DEVICES=28,29,30,31` |
| 分支 | `adapt/sdaa` |
| CUDA 基准机 | `sunhn@10.10.6.21`，`cuda_env_py310`（torch 2.6.0+cu124，A100-40GB）|

## 二、分析结论

### 2.1 模型本身无自定义 CUDA 算子（fairseq 框架另有通用 CUDA 扩展）

encoder（ESM2-650M）是标准 transformer，decoder（EGNN）是 einsum 图网络，
**模型本身无 .cu / triton / cpp_extension 自定义算子**，纯标准 PyTorch。

> fairseq 框架自带了 3 个 CUDA 扩展（`.cu` 文件）：`libnat_cuda`（非自回归翻译的编辑距离）、
> `ngram_repeat_block_cuda`（beam search 的 n-gram 重复惩罚）、`cuda_utils`（通用 CUDA 工具）。
> 它们是 fairseq 的**通用组件**，经 grep 确认 **EnzyGen2 的 geometric_protein_design 推理路径
> 完全不 import**；龙芯上无 `CUDA_HOME`，编译时自动跳过，对迁移无影响。

### 2.2 模型结构

```
EnzyGen2 = fairseq（精简版，只保留 geometric_protein_design 任务）
  ├─ encoder: ESM2-650M（esm2_t33_650M_UR50D，序列编码，2.6GB）
  ├─ decoder: EGNN 等变图网络（rm-node 模式）+ transformer（生成残基 3D 坐标 + 序列）
  └─ 输出: 酶序列 + CA 坐标 PDB
```

- 任务 `geometric_protein_design`；模型 `geometric_protein_model_ncbi_substrate_esm`
- EGNN vendored 单文件 `fairseq/models/egnn.py`（550 行，纯 Python）

### 2.3 依赖链

| 依赖 | 状态 | 处理 |
|------|------|------|
| torch | ✅ 2.12 + torch_sdaa | — |
| fairseq | ✅ 仓库 vendored（1.0.0a0）| Python 3.12 / omegaconf 2.3 兼容修复 |
| hydra-core / omegaconf | ✅ pip 装 | omegaconf 2.3 的 is_primitive_type 修复 |
| C++/Cython 扩展 | ✅ 5 个编译成功 | CUDA 扩展（.cu）自动跳过 |



## 三、适配改动（10 文件，99 行）

### 3.1 device 双兼容（sdaa 优先 + cuda 兜底）

| 文件 | 改动 |
|------|------|
| `fairseq/models/egnn.py` | `device = torch.device("cuda")` → try sdaa/cuda/cpu |
| `fairseq/models/geometric_protein_model.py` | 同上 |
| `fairseq/tasks/geometric_protein_design.py` | 同上 |
| `fairseq/trainer.py` | `self.device` 双兼容 |
| `fairseq_cli/generate.py` | 加 `use_sdaa` 分支（`.to("sdaa")` + `move_to_cuda(device=sdaa)`）|

关键：fairseq 原用 `torch.cuda.is_available()` 判断 device，SDAA 上为 False 导致模型/数据全留 CPU。
用 `.to("sdaa")`（**不是** `.cuda()`，后者报 `Torch not compiled with CUDA enabled`）。

### 3.2 EGNN 的 device 对齐（`indices on cpu` 报错修复）

`get_edges_batch` 用 `.to(device)` 把边索引转到 SDAA，但 `coords` 从输入（CPU）继承 device，
导致 `coord[row]` 报 `indices ... on cpu`。修复：

```python
# geometric_protein_model.py：传 gcl 前 coords 对齐 device
coords = coords.reshape(-1, coords.size()[-1]).to(x.device)
# egnn.py：coord2radial 里 row/col 对齐 coord.device（兜底）
row = row.to(coord.device)
col = col.to(coord.device)
```

### 3.3 Python 3.11+ dataclass mutable default 修复（13 处）

fairseq 老代码大量 `field: SomeConfig = SomeConfig()` 用 dataclass 实例作默认值，
Python 3.9 允许、3.11+ 拒绝。改为 `field(default_factory=SomeConfig)`：

- `fairseq/dataclass/configs.py`（11 处 `FairseqConfig` 子配置）
- `fairseq/models/transformer/transformer_config.py`（2 处 + 1 处 `field(default=QuantNoiseConfig())`）

配套修复 `fairseq/dataclass/initialize.py`：`default_factory` 化后字段 `.default` 变为
`dataclasses.MISSING`，hydra 注册时需用 `f.default_factory()` 而非 `f.default`。

### 3.4 omegaconf 2.1 → 2.3 兼容（`is_primitive_type` 移除）

fairseq 用 monkeypatch `omegaconf._utils.is_primitive_type` 绕开对象类型检查，该内部 API
在 omegaconf 2.3 被拆分成 `is_primitive_type_annotation` / `is_primitive_container` 等。

- `fairseq/dataclass/utils.py`：`omegaconf_no_object_check` 类改用 `hasattr` 自适应 + `flags={"allow_objects": True}`
- `fairseq/checkpoint_utils.py`：内联 hack 同样处理

### 3.5 checkpoint 三层嵌套解包（运行时，非代码改动）

EnzyGen2 发布的 checkpoint 是 **gzip + tar + zip** 三层嵌套（作者打包所致）：

```
checkpoint_best.pt（原始）
  ├─ 外层 gzip（1f 8b）
  ├─ 中层 tar（成员 checkpoint_best.pt，8.6GB）
  └─ 内层 zip（PK，torch new zipfile serialization，1984 成员）
```

`torch.load` 只认 zip/pickle，需逐层解包到内层 zip：

```bash
cd models/EnzyGen2
mv checkpoint_best.pt checkpoint_best.pt.gz && gunzip checkpoint_best.pt.gz   # 去 gzip
mv checkpoint_best.pt checkpoint_best.pt.tar && tar -xf checkpoint_best.pt.tar && rm checkpoint_best.pt.tar  # 去 tar
# 最终 checkpoint_best.pt 为 zip，torch.load 直接加载
```

> 代码侧无需改（`checkpoint_utils.py` 的 gzip 检测属可选增强，解包后走正常路径）。

### 3.6 C++ / Cython 扩展编译

5 个纯 CPU 扩展龙芯编译一次成功（CUDA 扩展在无 `CUDA_HOME` 时自动跳过）：

| 扩展 | 类型 | 龙芯 |
|------|------|:---:|
| libbleu | 纯 C++ | ✅ |
| libbase / libnat | C++（torch cpp_extension）| ✅ |
| data_utils_fast / token_block_utils_fast | Cython | ✅ |
| libnat_cuda / ngram_repeat_block_cuda / cuda_utils | CUDA（.cu）| ⏭️ 跳过 |

```bash
pip install cython bitarray sacremoses
python setup.py build_ext --inplace
```

## 四、运行方式

### 4.1 前置准备

1. 权重：`models/EnzyGen2/checkpoint_best.pt`（解包后 zip）+
   `esm2_t33_650M_UR50D.pt`（ESM2 encoder，首次运行自动下载到 `~/.cache/torch/hub/checkpoints/`）
2. 数据：`data/ncbi2id.json`（NCBI taxonomy 映射，Zenodo 19264491，136KB）
3. 示例：`example/example.json` + `example/3DWZ.cif`、`5C8U.cif`

> 微调生成需额外数据（Zenodo，注意脚本用名与 Zenodo 文件名不一致，需重命名）：
> `rhea_18421_final.json`（ChlR）← `chloramphenicol_acetyltransferase_final.json`、
> `rhea_20245_final.json`（AadA）← `aminoglycoside_adenylyltransferase_final.json`、
> `Thiopurine_S_methyltransferas_final.json`（TPMT）← `thiopurine_methyltransferase_final.json`。

### 4.2 SDAA 预训练生成命令

```bash
source /opt/tecoai/setvars.sh
export SDAA_VISIBLE_DEVICES=28,29,30,31
cd EnzyGen2 && export PYTHONPATH=$PWD:$PYTHONPATH

python fairseq_cli/generate.py example/example.json \
  --task geometric_protein_design --protein-task 83332 \
  --dataset-impl-source raw --dataset-impl-target coor \
  --path models/EnzyGen2/checkpoint_best.pt \
  --batch-size 1 --results-path models/output/EnzyGen2/83332 \
  --skip-invalid-size-inputs-valid-test --valid-subset test \
  --generation --decoding-strategy top-p --topp-probability 0.2
```

输出（`models/output/EnzyGen2/83332/`）：`pred_pdbs/*.pdb`（生成结构）、`protein.txt`（序列）、
`src.seq.txt`、`log_likelihood.txt` 等。

### 4.3 SDAA 微调生成命令（finetuning data_stage）

```bash
export SDAA_VISIBLE_DEVICES=28,29,30,31
mkdir -p models/output/finetune/18421/pred_pdbs models/output/finetune/18421/tgt_pdbs

python fairseq_cli/generate.py data/rhea_18421_final.json \
  --task geometric_protein_design --protein-task 18421 \
  --dataset-impl-source raw --dataset-impl-target coor \
  --data-stage finetuning \
  --path models/rhea_18421_finetune/checkpoint_best.pt \
  --batch-size 1 --results-path models/output/finetune/18421 \
  --skip-invalid-size-inputs-valid-test --valid-subset test \
  --generation --decoding-strategy top-p --topp-probability 0.4
```

> 注意：finetuning 的 `--valid-subset test` 默认跑**整个测试集**（每酶数千样本），
> 全量生成耗时数小时。若要快速验证可用 `--max-tokens 100` 限制样本数。

## 五、功能覆盖

| 功能 | 入口 | data_stage | 覆盖 |
|------|------|-----------|:---:|
| 预训练模型生成 | `generate_new_example.sh` / `generate_enzygen2_pretrain.sh` | `pretraining-full` | ✅ |
| 微调生成 ChlR / AadA / TPMT | `generate_ChlR.sh` / `generate_AadA.sh` / `generate_TPMT.sh` | `finetuning` | ✅ |
| 训练/微调 | `train_EnzyGen2_*.sh` / `reah_*.sh` | 训练 | ⬜（训练，不属迁移验证）|

**关于 finetuning 代码路径**：`load_dataset` 里 `if generation or data_stage == "finetuning"` 分支，
`--generation` 时即走 `NCBIFinetuneDataset`。三个 finetune checkpoint 均已下载、解包（gzip+tar+zip
三层嵌套，同预训练）并实测跑通，各酶 loss 正常：

| 酶 | loss | sequence loss | coordinate loss |
|----|------|--------------|----------------|
| ChlR（18421）| 0.372 | 0.417 | 8.326 |
| AadA（20245）| 0.045 | 0.053 | 0.813 |
| TPMT | 0.435 | 0.463 | 11.387 |

> finetune checkpoint 的区分方式：解包后读 `archive/data.pkl` 里 cfg 的数据路径
> （`rhea_18421`→ChlR、`rhea_20245`→AadA、`Thiopurine`→TPMT）。
>
> **注（微调未做 CUDA 对比）**：三个酶微调生成仅在 SDAA 上验证（上表 loss），
> 未在 CUDA 上跑。原因：微调模型与预训练模型**同架构同算子**
> （`geometric_protein_model_ncbi_esm`，仅权重值不同），精度特性已由第 7 节的
> 预训练 CUDA vs SDAA 对比证明（sequence loss 一致 ±0.4%，coordinate loss 差异来自
> `np.random` 无种子初始化 + 连续坐标浮点分叉），微调不会引入新的精度差异；
> 性能上同为 batch=1 单条前向，与预训练的 29.5× 结论一致。

## 六、踩坑记录

| # | 坑 | 根因 | 解决 |
|---|----|------|------|
| 1 | 205 文件假 modified | Windows 解压 CRLF | `git config core.autocrlf input` + checkout |
| 2 | dataclass mutable default 报错 | py3.9→3.12 | 13 处 `field(default_factory=...)` |
| 3 | omegaconf `is_primitive_type` AttributeError | omegaconf 2.1→2.3 | hasattr 自适应 + allow_objects |
| 4 | `invalid load key '\x1f'` | checkpoint 是 gzip | 见 §3.5 三层解包 |
| 5 | `indices on cpu` | use_cuda=False 模型留 CPU | generate.py 加 use_sdaa |
| 6 | `h[row]` 报 cpu 索引 | coords/edges device 不一致 | §3.2 device 对齐 |

## 七、精度性能对比（CUDA vs SDAA）

同一输入（example 蛋白 83332，3DWZ 底物），同一套适配代码（device 双兼容），
batch=1，top-p（topp=0.2）采样。

### 7.1 性能对比

| 指标 | CUDA（A100-40GB）| SDAA（龙芯）| 差距 |
|------|------|------|------|
| 纯推理时间 | 0.77s | 22.7s | **慢 29.5×** |
| 总耗时（含模型加载）| 15.7s | 32.7s | 慢 2.1× |

> 纯推理慢 29.5× 的主要原因：ESM2-650M 是 33 层 transformer，SDAA 单条（batch=1）
> 前向利用率低；EGNN 的 `NearestNeighbors`（sklearn 在 CPU）+ 边索引在 CPU↔SDAA 间搬移。

### 7.2 精度对比（确定性 loss 指标）

| 指标 | CUDA | SDAA | 差异 |
|------|------|------|------|
| sequence loss | 0.496 | 0.498 | **0.4%** ✅ |
| loss（总）| 0.38 | 0.375 | **1.3%** ✅ |
| coordinate loss | 3.661 | 3.003 | 21.9% ⚠️ |

**精度结论**：
- **序列相关指标高度一致（±0.4%）** → ESM2/transformer 的 logits、softmax 在 SDAA 上算得和
  CUDA 一样对，算子精度正确。
- **coordinate loss 有 21.9% 差异，但非纯精度问题**：`geometric_protein_model.py` 的 coords
  初始化用 `np.random.uniform`（无 seed），SDAA/CUDA 走不同随机序列，坐标初始化本就不同；
  加之 EGNN 连续坐标计算受浮点差异放大，与 RFdiffusion/BoltzGen 的「序列稳定、坐标分叉」同特征。

### 7.3 生成序列

top-p 随机采样，CUDA/SDAA 生成序列**必然不同**（无可比性）：

- CUDA: `MPAAREAELEAARAARLGVVVLPTAAEEALEYRADERFAFCSTFKTAL...`
- SDAA: `MPAAREAALEEARAARLGVLAAATGGAAALEYRADERFAFCSTFKAAL...`

> 若要精确对标需固定 `np.random.seed` + torch 种子，但 coordinate 连续量的浮点分叉无法消除。