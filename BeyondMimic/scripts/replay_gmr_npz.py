"""播放篮球动作NPZ文件的脚本

使用方法:
    python scripts/play_basketball_npz.py --motion_file g1/Basketball-NPZ/shoot_m.npz
"""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="播放篮球动作NPZ文件")
parser.add_argument(
    "--motion_file",
    type=str,
    default="g1/Basketball-NPZ/shoot_m.npz",
    help="NPZ动作文件路径"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下是Isaac Sim启动后的代码"""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class BasketballSceneCfg(InteractiveSceneCfg):
    """篮球动作场景配置"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class NPZMotionPlayer:
    """NPZ动作播放器"""

    def __init__(self, motion_file: str, device: torch.device):
        self.motion_file = motion_file
        self.device = device
        self._load_motion()

    def _load_motion(self):
        """加载NPZ文件"""
        data = np.load(self.motion_file)

        # 加载所有数据到GPU
        self.fps = int(data['fps'][0])
        self.dt = 1.0 / self.fps
        self.joint_pos = torch.from_numpy(data['joint_pos']).float().to(self.device)
        self.joint_vel = torch.from_numpy(data['joint_vel']).float().to(self.device)
        self.body_pos_w = torch.from_numpy(data['body_pos_w']).float().to(self.device)
        self.body_quat_w = torch.from_numpy(data['body_quat_w']).float().to(self.device)
        self.body_lin_vel_w = torch.from_numpy(data['body_lin_vel_w']).float().to(self.device)
        self.body_ang_vel_w = torch.from_numpy(data['body_ang_vel_w']).float().to(self.device)

        self.num_frames = self.joint_pos.shape[0]

        duration = self.num_frames / self.fps
        print(f"加载动作文件: {self.motion_file}")
        print(f"  FPS: {self.fps}")
        print(f"  总帧数: {self.num_frames}")
        print(f"  时长: {duration:.2f}秒")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """运行模拟器"""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # 加载动作
    motion = NPZMotionPlayer(args_cli.motion_file, sim.device)

    # 当前帧索引
    frame_idx = 0

    print("\n开始播放动作...")
    print("按 ESC 退出\n")

    # 模拟循环
    while simulation_app.is_running():
        # 获取当前帧的状态
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[frame_idx, 0] + scene.env_origins[:, :]
        root_states[:, 3:7] = motion.body_quat_w[frame_idx, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[frame_idx, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[frame_idx, 0]

        # 写入机器人状态
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(
            motion.joint_pos[frame_idx:frame_idx + 1],
            motion.joint_vel[frame_idx:frame_idx + 1]
        )

        # 更新场景
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        # 更新相机位置（跟随机器人）
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([3.0, 3.0, 1.5]), pos_lookat)

        # 下一帧
        frame_idx += 1
        if frame_idx >= motion.num_frames:
            frame_idx = 0
            print(f"动作循环播放 (帧 {frame_idx}/{motion.num_frames})")


def main():
    """主函数"""
    # 配置模拟器
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02  # 50 FPS
    sim = SimulationContext(sim_cfg)

    # 创建场景
    scene_cfg = BasketballSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 重置模拟器
    sim.reset()

    # 运行模拟器
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
