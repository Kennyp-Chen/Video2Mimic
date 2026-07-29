# AGENTS.md — Video2Mimic

## 项目性质

**Video2Mimic**是将单目视频转换为宇树G1等机器人动作数据的Pipeline。主仓库在 `https://github.com/Kennyp-Chen/Video2Mimic`，本地克隆名为 `Video2Track`。

它是一个**元仓库**（含3个子模块），不构成独立Python包——核心逻辑在三方子模块内，本仓库提供集成脚本和改进。

## 仓库结构

```
Video2Track/
├── GEM-X/          # NVIDIA GEM — 77关节SOMA人体姿态估计（Apache 2.0）
├── GMR/            # 通用动作重定向 — SMPL-X/BVH/FBX → 机器人DOF（MIT）
├── GVHMR/          # 单目视频 → SMPL-X人体运动恢复（SIGGRAPH Asia 2024）
├── Robot_videos/   # 测试视频素材（未跟踪）
├── video_editing.py      # 视频裁剪工具（ffmpeg）
├── audio_video_merge.py  # 音频视频融合工具（ffmpeg）
├── GMR/scripts/gvhmr2robot_fix.py    # ★ 核心改进脚本（地面接触修正、默认姿态填充、SLERP平滑）
└── GMR/scripts/batch_gmr_pkl_to_csv.py # ★ PKL批量转CSV（供BeyondMimic使用）
```

## 子模块关系

| 子模块 | 上游来源 | 备注 |
|--------|----------|------|
| GEM-X | https://github.com/NVlabs/GEM-X | 完整克隆，本仓库不改动 |
| GMR | https://github.com/YanjieZe/GMR | **本仓库fork**，`scripts/gvhmr2robot_fix.py` 和 `scripts/batch_gmr_pkl_to_csv.py` 是自定义新增 |
| GVHMR | https://github.com/zju3dv/GVHMR | **本仓库fork**，未发现改动 |

子模块**不是git submodule**，是独立git仓库的完整克隆。各子模块有独立的remote指向各自fork。

## 核心Pipeline

### 完整流程（3步）

```bash
# Step 1: 视频 → SMPLX人体动作（GVHMR）
cd GVHMR
conda activate gvhmr  # 独立环境，PyTorch 2.3 + CUDA 12.1
python tools/demo/demo.py --video=path/to/video.mp4 -s
# 输出: GVHMR/outputs/demo/<video_name>/hmr4d_results.pt

# Step 2: SMPLX → 机器人DOF轨迹（GMR增强版）
cd ../GMR
conda activate gmr  # 独立环境，Python 3.10+，mink+mujoco
python scripts/gvhmr2robot_fix.py \
  --gvhmr_pred_file ../GVHMR/outputs/demo/<video_name>/hmr4d_results.pt \
  --robot unitree_g1 \
  --ground_contact \
  --pad_default_pose \
  --pad_frames 30 \
  --default_facing_angle 90,0 \
  --record_video
# 输出: GMR/motions/G1/<video_name>.pkl

# Step 3: PKL → CSV（供BeyondMimic/RL训练）
python scripts/batch_gmr_pkl_to_csv.py --folder motions/G1/
```

### GEM-X替代方案

```bash
cd GEM-X
source .venv/bin/activate  # uv + Python 3.12
python scripts/demo/demo_soma.py --video path/to/video.mp4 --ckpt inputs/pretrained/gem_soma.ckpt --retarget
```

## gvhmr2robot_fix.py 关键参数

| 参数 | 作用 |
|------|------|
| `--ground_contact` | 脚部接地修正（防止悬空/穿地） |
| `--pad_default_pose` | 动作前后插入默认站立姿态并平滑过渡 |
| `--pad_frames N` | 过渡帧数（默认30） |
| `--default_facing_angle A,B` | 起始/结束面朝角度（度），如 `90,0` |
| `--robot` | 支持 `unitree_g1`（29DOF）、`unitree_g1_23dof` 等 |

内部使用SLERP四元数插值进行平滑过渡，含安全根部高度检测防止蹲姿穿地。

## 工具脚本

```bash
# 视频裁剪（移除前后指定秒数）
python video_editing.py --input in.mp4 --output out.mp4 --start_trim 2.0 --end_trim 3.0

# 音视频融合（支持音频延迟）
python audio_video_merge.py --video in.mp4 --audio in.mp3 --output out.mp4 --audio_start 0.0
```

## 环境隔离

三个子模块需要**独立的conda环境**，不可混用：

| 环境 | Python | 关键依赖 | 用途 |
|------|--------|----------|------|
| `gvhmr` | 3.10 | PyTorch 2.3+cu121, lightning, hydra, smplx | 视频→SMPLX |
| `gmr` | ≥3.10 | mink, mujoco, qpsolvers, smplx | SMPLX→机器人 |
| `gem-x` | 3.12 | PyTorch 2.10+, uv, soma | 视频→SOMA→机器人 |

GMR安装后需修改 `smplx/body_models.py` 中 `ext` 从 `npz` 改为 `pkl`（使用pkl格式SMPL-X时）。

## 关键约束

- **无CI**：根仓库无GitHub Actions，GEM-X子模块内有lint workflow（ruff+black），与本仓库无关
- **无测试框架**：不运行pytest/unittest
- **无linter/formatter配置**：根目录无pyproject.toml/setup.cfg
- **Git单分支**：仅`main`分支，无tag，3次提交历史
- **gitignore**：忽略 `*.pt` `*.pkl` `*.pth` `*.onnx` `wandb/` `logs/` `videos/` `outputs/`；不忽略 `.csv`
- **Large files**：Robot_videos/ 包含mp4/mov/mp3文件，未被git跟踪（未git add）
- **中文提交信息**：本仓库commit message使用中文

## 开发注意事项

1. **子模块代码修改**：修改GMR/GVHMR内的文件时，确认是在本仓库fork上工作（remote指向 `Kennyp-Chen/Video2Mimic`），而非上游
2. **各子模块独立发布**：GMR和GVHMR各自有独立的setup.py，可 `pip install -e .` 安装
3. **新增机器人支持**：在GMR中需要添加robot XML/URDF到 `assets/`，在 `general_motion_retargeting/params.py` 注册，创建ik_config JSON文件
4. **可视化**：MuJoCo窗口播放时 `[`/`]` 切上一段/下一段，`space` 暂停/播放
5. **视频录制**：重定向脚本加 `--record_video` 即可输出mp4，不需额外工具
