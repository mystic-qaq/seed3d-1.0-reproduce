# 我的研究领域

下面几个领域，与我研究的相关性由高到低排序，我会分别按个人思路给出一些看过的工作：

## 3D Generation (最熟悉的领域)：

- Native 3D Generation：一种是从 Latent Space 解码出 SDF 后使用 Marching Cubes等曲面提取算法提取 Geometry，隐空间形态类似 Vecset 的有 Hunyuan3D-2.1,TripoSG,Seed3D-1.0 等，形态类似 Voxel 的有 Direct3D,TRELLIS系列；另一种是生成顶点和连接关系，将 Topology 也考虑进去，AR 式有的 MeshGPT，Diffusion 式的有 Nexus，FlowMatching 式的有 LATO.2 等；还有一些尝试提高精度和效率的（比如 Block3D 等），与 Reconstruction 结合以提高指令遵循度的（比如 Pixal3D 等），原生实现绑骨的，以及分割场景中的物体或者分部件生成以尝试原生可交互的（比如 SAM3D,Partcrafter 等）。
  
- Agentic 3D Generation：近期我看到 GPT-6 Astra 的 3D 能力后，意识到 3D Generation 与 Agent 结合是未来重要的方向（优点是可以更好地实现关节交互、仿真等，缺点是精度暂时不及 Native Generation）：我看过的工作有 PhysX-anything,SAGE,Procedure,Lucida，它们结合的方式各不相同，但都很有意思；Github 上也有 Articraft,img2threejs 等有趣且热度上升的项目；
  
- 3D Editing：我认为 3D 编辑也是不可与 3D Generation 分开讲的部分，比如近期有 Hunyuan-Buffalo 等论文；

- Others：无论 Generation 的核心是 AR Model,Flow Model,Diffusion Model,VLM, RL 都很重要，比如 DreamDPO,FlowGRPO,DiffusionNFT,Diffusion-OPSD 等工作和 Omni-veRL 的工作都可以迁移到 3D Generation 和 Video Generation 中，且可能与目前火热的 World Model 相结合；相应的 Infra 搭建也很重要（比如 TRELLIS 就专门适配了 Infra；RL Infra 目前也人才稀缺）。

## 3D Reconstruction：

- 在 Feed Forward 3D 之前，主流做法是逐场景优化：比如ColMap (本质是 SfM + MVS )及其变体 （比如 MVSNet等）， NeRF 及其变体（Triplane NeRF等），3DGS 及其变体（比如 ProxyGS）；

- Feed Forward Reconstruction: 先前有 LRM,TripoSR 等工作主要集中在对物体的 3D Reconstruction，有 DUSt3R,MASt3R,VGGT（这个比较特殊，是集大成者） 等工作集中在对场景的 3D Reconstruction。还有很多尝试将 Feed Forward Reconstruction 与 GS 结合的工作；
  
- 近期我还看到一些相关工作，比如尝试提高 Efficiency 的，尝试探索 3D Scaling Law 的 VGGT-Omega，尝试将 3D 升维为 4D 以重建动态场景的 D4RT(2026 CVPR Best Paper)，这些我认为也都属于与 3D Reconstruction 不可分割的部分。

## Video Generation：

- 我对此不太熟悉但是很感兴趣，知道 Wan-2.1,Minimax-H3 等基座模型。我认为这一领域与 3D Generation 有许多相似之处 ：除了基座模型外，Video Editing, RL in Video Generation, Benchmark Generation Quality 等方面也是不可分割的。

- 我将 Image 理解为 Video 的某种降维形态，因此如果有重要或有意思的相关工作也可以给我！
  
## Embodied AI：

- 我不太熟悉但是略感兴趣，只了解过 OpenVLA,pi_0,pi_0.5 等重要工作。这是一个非常广泛、路线繁多的领域。如果在前面几个领域中，你实在找不到认为有价值、值得读的工作，再到这个领域里寻找。