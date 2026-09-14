# INSTRUCTION.md — 这个项目怎么读、怎么跑、怎么讲

这份文档面向"接手这个项目的你"。目标有三个：

1. **能跑起来**：从零到出一条 mesh，命令级可复制。
2. **能读懂**：每个模块为什么长这样、对应论文哪一句。
3. **能讲清楚**：包括哪些做成了、哪些没做成、为什么没做成——这一条在面试里比"跑通了"更值钱。

---

## 0. 一分钟版本

这个仓库是 **Seed3D 1.0（arXiv 2510.19944）形状生成部分的独立复现**，外加一个能在
4×RTX3090 上训练的 mini 版本。

* 代码全部自己写，**与 Hunyuan3D-2.1 的官方 VAE 做过逐层数值对齐（56 项全部 bit 级一致）**；
* 数据管线在 **1005 个真实 ModelNet40 资产**上跑通，重网格化 100% 水密；
* 训练在 4 卡上端到端可复现；
* **mini 模型本身没有训练成功**（VAE 收敛到"平均场"），根因已定位到解码器结构，
  并给出并验证了第一个修复（query-skip，相关性 −0.06 → +0.37）。

最后一条是这份文档的重点：**怎么把一个"没完全成功"的项目讲成一个高质量项目**。

---

## 1. 环境

```bash
cd /home/xy/seed3d-1.0-reproduce

# Python 环境（已建好，torch 2.5.1+cu121，适配 RTX3090 sm_86）
envs/seed3d/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 网络：直连 huggingface.co / github.com 不通，必须走镜像
export HF_ENDPOINT=https://hf-mirror.com      # 或 source scripts/env_hf.sh
```

关键依赖与"为什么需要它"：

| 包 | 用途 | 备注 |
|---|---|---|
| torch 2.5.1+cu121 | 全部模型与训练 | RTX3090 = sm_86，cu121 可用 |
| transformers 5.x | DINOv2 与 RADIO | RADIO 需要 `trust_remote_code=True` |
| timm | RADIO 的 remote code 依赖 | 否则只能用 `dinov2_pair` 兜底 |
| trimesh + rtree | 网格 IO、射线内外判断 | **rtree 必须装**，否则 `contains()` 报错 |
| fast-simplification | 网格简化（渲染提速 5×） | 不装会退化到全分辨率渲染 |
| scikit-image | marching cubes | 提取等值面 |
| scipy | cKDTree / 连通域洪泛 | 重网格化核心 |

---

## 2. 目录结构（按阅读顺序）

```
src/seed3d_mini/
  modules/attention.py        Fourier 位置编码、注意力、Transformer（与参考实现同构）
  modules/point_sampling.py   FPS、球查询、编码器 query/data 构造
  models/vae.py               ★ Seed3D-VAE：encoder/decoder/TSDF 头/损失
  models/dit.py               ★ Seed3D-DiT：FLUX 式双流+单流主干
  models/conditioner.py       ★ 双图像编码器（DINOv2 ‖ RADIO，通道拼接）
  flow_matching.py            rectified flow：logit-normal 时间步 + 长度感知平移
  mesh_ops.py                 ★ 重网格化、表面/锐边采样、TSDF 查询
  volume_decode.py            稠密/分层体解码 + marching cubes
  render/rasterizer.py        PyTorch Z-buffer 渲染（替代 Blender）
  data/dataset.py             shard 数据集（分层查询采样）
  train_utils.py              DDP/日志/checkpoint + StepProfiler
  parallel.py                 flat all-reduce 数据并行（见 §6）

scripts/   prepare_data.py / train_vae.py / train_dit.py / pack_latents.py
           infer.py / evaluate.py / add_renders.py / reweight_shards.py / run_mini.sh
tests/     test_vae_parity.py（数值对齐）smoke_pipeline.py test_mesh_ops.py
           test_integration.py overfit_one.py probe_latent_usage.py diagnose_latents.py
configs/   vae_mini.yaml / dit_mini.yaml
docs/      REPRODUCTION_NOTES.md（★论文→代码映射+全部结论）
           TRAINING_RECIPE.md（★实测开销+速度排查）DATA_PIPELINE.md MODELS.md
```

打 ★ 的六个文件是核心，其余是支撑。

---

## 3. 跑起来：四个阶段

### 阶段 0：先确认代码正确（不需要数据，2 分钟）

```bash
envs/seed3d/bin/python tests/smoke_pipeline.py                 # CPU 也能跑，全过
envs/seed3d/bin/python tests/test_mesh_ops.py                  # 8 项几何不变量
envs/seed3d/bin/python tests/test_vae_parity.py                # 与 Hunyuan3D 逐层对齐(56 PASS)
```

`test_vae_parity.py` 是这个项目最有说服力的一个测试：它把官方 Hunyuan3D-2.1 的
VAE 权重（410 个张量）映射进我们自己的实现，逐层比对，**全部 max|Δ| = 0.0**。
它同时把两条不平凡的权重映射规则固定下来了（`c_kv` 不需要重排；`c_qkv` 需要
`[heads,3,d] → [3,heads,d]`）。面试时这一条最值得讲。

### 阶段 1：数据

```bash
# 下载（约 1278 个 .off，走镜像，有断点续传）
envs/seed3d/bin/python scripts/prepare_data.py download \
    --categories airplane,car,guitar --file-list data/raw/file_list_cats.json --workers 4

# 处理：归一化 → 重网格化(水密) → 采样 → TSDF 查询（CPU，12~16 进程，约 1.5~2h）
envs/seed3d/bin/python scripts/prepare_data.py process \
    --categories airplane,car,guitar --workers 14 --remesh-resolution 96

# 打包成 shard（类别平衡，每类 250）
envs/seed3d/bin/python scripts/prepare_data.py shards --per-category 250
envs/seed3d/bin/python scripts/prepare_data.py stats
```

产出：`data/shards/{train,val}-*.npz`（750 样本）。
参考图渲染是单独一步（放 shard 里）：

```bash
envs/seed3d/bin/python scripts/add_renders.py --workers 40 --resolution 256
```

### 阶段 2：训练 VAE

```bash
# 用空闲的那一组卡！(见 §6)
SEED3D_PROFILE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
envs/seed3d/bin/python -m torch.distributed.run --nproc_per_node=4 \
    scripts/train_vae.py --config configs/vae_mini.yaml --steps 4000
```

### 阶段 3：打包 latent + 训练 DiT

```bash
envs/seed3d/bin/python scripts/pack_latents.py \
    --vae checkpoints/vae_mini/vae_final.pt --config configs/vae_mini.yaml \
    --out data/latents_mini --batch-size 16 --device cuda

CUDA_VISIBLE_DEVICES=0,1,2,3 \
envs/seed3d/bin/python -m torch.distributed.run --nproc_per_node=4 \
    scripts/train_dit.py --config configs/dit_mini.yaml --steps 3000
```

### 阶段 4：出 mesh + 评测

```bash
# 图片 → mesh（--image auto 会用 shard 里的参考图，方便对照）
envs/seed3d/bin/python scripts/infer.py \
    --vae checkpoints/vae_mini/vae_final.pt --dit checkpoints/dit_mini/dit_final.pt \
    --image auto --mesh-from-shard data/shards/val-00000.npz --index 0 \
    --steps 40 --guidance 4.5 --resolution 192 --out artifacts/meshes/demo.glb

# 量化指标：Chamfer / F-score / IoU / 法向一致性
envs/seed3d/bin/python scripts/evaluate.py --level vae --n 20 \
    --vae checkpoints/vae_mini/vae_final.pt --out artifacts/eval --save-meshes
```

---

## 4. 怎么读代码：每个模块"为什么长这样"

| 模块 | 论文依据 | 关键实现点 |
|---|---|---|
| `VecSetEncoder` | §2.1.1 Eq.1：`Z0=CrossAttn(PE(P),n_P)`，`Zi=SelfAttn` | query 是 FPS 选出的点，key/value 是随机子集；**latent 无位置编码**（这是多尺度 token 的前提） |
| `TSDFDecoder` | §2.1.1 Eq.2 | 查询点 Fourier → 自注意力 → 对 `Z` 交叉注意力 → MLP；**输出的是 TSDF 不是 occupancy** |
| `Seed3DVAE.forward` | Eq.3：`L=L_recon+λ_KL·L_KL` | KL warm-up；多尺度 `M∈{256,384,512}`；**目标归一化**（见 §5） |
| `FlatDataParallel` | 报告用 HSDP，这里等价单机版 | 一次扁平 all-reduce，避开 DDP 分桶（见 §6） |
| `Seed3DDiT` | §2.1.2 FLUX 式 | 双流块（形状/图像各自 norm+QKV+MLP，联合注意力）+ 单流块；条件 token 在最后被丢掉 |
| `DualImageEncoder` | §2.1.2 双编码器 | DINOv2 与 RADIO 的 patch 网格不同（37² vs 32²），用 `PixelShuffleAlign` 对齐后**按通道拼接** |
| `flow_matching.py` | §2.1.2 | logit-normal 时间步 + **长度感知平移** `μ`；`mu` 必须 clamp，否则短序列会把 t 全推到 1（踩过） |
| `mesh_ops.remesh_watertight` | §3.1 | 体素化 → 从边界洪泛出外部 → 形态学清理 → marching cubes；**半个体素偏移**是必须的（踩过） |
| `volume_decode.hierarchical_volume_decode` | §6 | 粗网格定位 → 只细化活动带 → 远场填饱和值；与稠密解码差 2.4e-7 |

**两个"论文没说清、我做了决定"的地方**（详见 `docs/REPRODUCTION_NOTES.md`）：

1. Eq.2 的解码器自注意力放在 **query** 侧；但它引用的参考实现（Dora/Hunyuan）放在
   **latent** 侧。两者数值差 rel 0.52，**不可互换**。代码里做成了开关
   `decoder_self_attn: query|latent|none`。
2. 报告说 TSDF 监督，参考实现是 occupancy + 极小 level。这决定了提取时
   `mc_level=0` 还是 `-1/512`，也直接影响本节 §5 的那个坑。

---

## 5. 核心失败与 insight（面试就讲这些）

### 5.1 VAE 收敛到"平均场"——训练曲线完全看不出来

**现象**：weighted recon 一路降到 `1.3e-4`，看起来非常健康；但解码器输出与输入
**几乎无关**：两个不同形状解码出的场差 `mean|Δ| = 0.00003`，训练 loss 实测
`1.918e-4`，而"直接输出一个常数"就能拿到 `1.843e-4`。

**为什么曲线骗人**：目标是 *截断* TSDF，动态范围只有 ±0.1；形状信息在带内只有
std `0.0135`（物体尺寸是 2，即包围盒的 0.7%），而"输出平均场"已经值 96% 的可降损失。

**排查路径**（每一步都留了脚本，可复现）：

| 步骤 | 脚本 | 结论 |
|---|---|---|
| ① 训练曲线正常吗 | `artifacts/logs/*.jsonl` | 正常，且 recon 优于常数 |
| ② latent 被用了吗 | `tests/probe_latent_usage.py` | **没有**：`DECODER IGNORES THE LATENT` |
| ③ 换个目标函数会好吗 | plain MSE vs weighted | 都一样，排除 loss 归一化 |
| ④ 加步数会好吗 | 2000 步 @0.6s/步 | **不会**，20× 步数结果不变 → 排除训练预算 |
| ⑤ 能过拟合单个形状吗 | `tests/overfit_one.py` | 不能，平台正好等于最优常数 |
| ⑥ 梯度传到 encoder 了吗 | 手写 backward + grad norm | 传了（1.86），且有区分度的 latent（‖z_i−z_0‖/‖z_0‖≈1.2） |
| ⑦ **解码器本身能拟合吗** | 冻结 latent，只训 decoder | **不能** |
| ⑧ 对照：MLP 能拟合吗 | Fourier → MLP(256) | **能，corr +0.86** |

**根因**：第 ⑦⑧ 步对比把问题钉死在**解码器结构**上——同一组 Fourier 特征，
两层 MLP 能拟合（corr 0.86），而"交叉注意力解码器"连坐标函数都学不出来
（corr −0.05，且关掉自注意力、换 query 侧都一样）。也就是说，这个解码器的
输出通路太窄，latent 里再好的信息也传不到输出。

**已验证的第一个修复**：给输出头加一条**从原始 Fourier 特征直连的 skip**
（`query_skip=True`）：同一实验下 corr 从 **−0.06 → +0.37**，loss 从 1.94e-4 →
1.58e-4。方向对了，但还没到 MLP 的 0.86，说明还需要（a）加宽 skip 头，
（b）把 latent 作为**残差调制**而不是唯一信息源。

**还剩的候选修复**（`docs/REPRODUCTION_NOTES.md` §4b）：加宽截断带并把带内目标
重标到 `[-1,1]`（把信号幅度提高约 5×）、用 occupancy 做粗监督 + TSDF 做细监督。

### 5.2 训练比硬件慢 20 倍——是两个环境问题

**现象**：单卡实测 0.30 s/步，4 卡 DDP 却要 13–27 s/步。

**排查**：加 `SEED3D_PROFILE=1` 打印分阶段耗时，定位到 `fwd_bwd` 占 98%（数据加载
只占 1.6%），再用 A/B 对比同一模型/DDP 开关：

| GPU 组 | torch DDP | flat all-reduce | 不同步 |
|---|---|---|---|
| 0,1,2,3（空闲） | **0.340 s/步** | 0.318 | 0.277 |
| 4,5,6,7（GPU 6 被占） | **7.3–7.9 s/步** | 4.6–9.8 | — |

**原因 1**：GPU 6 上有别人的进程（99% 占用）。NCCL 集合通信需要所有 rank 参与，
**一块卡被占就把整个作业拖死**。
**原因 2**：`nvidia-smi` 显示 8 张卡里 7 张协商到 **PCIe Gen 1**（上限是 Gen 4），
实测 all-reduce 只有 4–5.3 GB/s（健康 Gen4 x16 约 25 GB/s）。这是主机
ASPM/转接卡层面的问题，脚本修不了。

**可复用的经验**：共享服务器上做多卡实验，先看 `nvidia-smi` 有没有别的进程、
再用 profiler 分清"数据/计算/通信"，不要靠猜。

### 5.3 一路上修掉的其它真 bug（都是量出来的）

| 现象 | 根因 | 修复 |
|---|---|---|
| 单卡能过拟合、全量不能 | 远场 MSE(7.2e-3) 比整条带内信号(2.9e-3) 还大 | 远场权重 0.25 → 0.002 |
| 重网格化后不是水密 | 轴对齐平面正好落在样本点上，marching cubes 出开边 | 网格**偏移半个体素** |
| 数据集被撑坏 | `binary_fill_holes` 把开放壳体的内部全填了（体积 0.20→0.42） | 去掉该步，只做 6 连通闭/开运算 |
| 时刻全被推到 1 | 长度感知平移在 `num_tokens < base_tokens` 时外推出 μ<1，反而反向平移 | `mu` clamp 到 `[base_shift, base_shift+max_shift]` |
| 采样不可复现 | `sample_timesteps` 用了全局 RNG | 传入 `generator` |
| 训练脚本直接崩 | `torch.timedelta` 不存在；NCCL all_reduce 收到 CPU 张量 | `datetime.timedelta`；metrics 先 `.to(device)` |
| eikonal 让训练爆掉 | 随机初始化时 |∇d|≈150 → 项值 2.7e4 | warm-up + 限制在未截断区域 + 子采样 512 点 |

---

## 6. 在这台机器上必须知道的四件事

1. **选空闲的卡**：`nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv`。
   只要组里有一张卡被占，多卡训练就会被拖慢 10–20×。
2. **PCIe Gen 1**：`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max --format=csv`。
   全机 7/8 张是 Gen 1，多卡通信带宽约 5 GB/s，别指望线性加速。
3. **用 flat 数据并行**：配置里 `training.data_parallel: flat`（默认），比 DDP 稳。
4. **慢就跑 profiler**：`SEED3D_PROFILE=1` 会打印 data/h2d/fwd_bwd/optim/log 各阶段耗时。

---

## 7. 怎么验收（给面试官看的顺序）

```bash
# ① 代码正确性：与官方权重逐层一致
envs/seed3d/bin/python tests/test_vae_parity.py | tail -20     # 56 PASS
# ② 数据管线在真实数据上成立
envs/seed3d/bin/python scripts/prepare_data.py stats            # 3 类 x 250，全水密
# ③ 全链路可跑（CLI 级）
envs/seed3d/bin/python tests/test_integration.py                # 5 阶段全绿
# ④ 模型质量：现状如何、为什么
envs/seed3d/bin/python tests/probe_latent_usage.py --vae checkpoints/vae_mini/vae_final.pt
```

**当前诚实的结论**：①②③ 全部达成；④ 未达成（VAE 平均场），根因已定位到解码器
结构，第一个修复已验证有效但还不够。

---

## 8. 建议的下一步（按性价比排序）

1. **加宽监督带**：`truncation 0.1 → 0.3`，把带内目标重标到 `[-1,1]`，形状信号幅度
   提高约 5×。改动集中在 `mesh_ops.build_query_cloud` + `vae.forward`，重跑数据
   与 VAE（约 1h）。
2. **加宽 query-skip 头**并让 latent 走**残差调制**（FiLM/adaLN），而不是唯一信息源。
3. **粗监督 + 细监督**：低分辨率 occupancy 给全局形状，TSDF 只负责表面细化。
4. 上述都通了以后，再谈"多训一会儿"——顺序反了会一直在平均场附近打转。

如果要把项目再抬一档：把 §5.1 的 ①→⑧ 排查过程整理成 4~6 页技术笔记（图 + 数字），
它证明的是"能定位一个开放问题"，而这正是复现类项目最难展示的能力。
