from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

'''
用于IsaacLab仿真环境的运动命令生成器，它通过加载预录制的运动数据（如关节位置、身体姿态等）来为机器人提供参考轨迹，并计算机器人与参考轨迹之间的误差作为强化学习任务的奖励信号或评估指标。
'''
class MotionLoader:
    """运动数据加载器，负责从文件中加载预录制的运动数据"""
    
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        # 验证运动文件是否存在
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        # 加载运动数据文件
        data = np.load(motion_file)
        # 帧率（每秒帧数）
        self.fps = data["fps"]
        # 关节位置数据 [时间步数, 关节数]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        # 关节速度数据 [时间步数, 关节数]
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        # 身体部位在世界坐标系中的位置 [时间步数, 身体部位数, 3]
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        # 身体部位在世界坐标系中的四元数姿态 [时间步数, 身体部位数, 4]
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        # 身体部位在世界坐标系中的线速度 [时间步数, 身体部位数, 3]
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        # 身体部位在世界坐标系中的角速度 [时间步数, 身体部位数, 3]
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        # 需要跟踪的身体部位的索引
        self._body_indexes = body_indexes
        # 运动数据的总时间步数
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """获取选定身体部位在世界坐标系中的位置"""
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """获取选定身体部位在世界坐标系中的四元数姿态"""
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """获取选定身体部位在世界坐标系中的线速度"""
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """获取选定身体部位在世界坐标系中的角速度"""
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    """基于运动数据的命令生成器，用于模仿学习任务"""
    
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 获取机器人对象
        self.robot: Articulation = env.scene[cfg.asset_name]
        # 获取机器人锚点身体的索引
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        # 获取运动数据中锚点身体的索引
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        # 获取所有需要跟踪的身体部位在机器人中的索引
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # 加载运动数据
        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        # 当前每个环境对应的时间步
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 身体部位相对于锚点的位置
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        # 身体部位相对于锚点的姿态
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0  # 初始化为单位四元数

        # 自适应采样相关参数
        # 将运动数据分成多个时间区间（bin）
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        # 记录每个时间区间失败的概率
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        # 当前episode中每个时间区间的失败计数
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        # 自适应采样的卷积核（用于平滑失败概率）
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()  # 归一化

        # ========== 指标定义和解释 ==========
        self.metrics = {
            # 锚点身体位置误差：机器人锚点身体与参考运动锚点身体之间的位置距离
            "error_anchor_pos": torch.zeros(self.num_envs, device=self.device),
            
            # 锚点身体旋转误差：机器人锚点身体与参考运动锚点身体之间的旋转差异（四元数误差）
            "error_anchor_rot": torch.zeros(self.num_envs, device=self.device),
            
            # 锚点身体线速度误差：机器人锚点身体与参考运动锚点身体之间的线速度差异
            "error_anchor_lin_vel": torch.zeros(self.num_envs, device=self.device),
            
            # 锚点身体角速度误差：机器人锚点身体与参考运动锚点身体之间的角速度差异
            "error_anchor_ang_vel": torch.zeros(self.num_envs, device=self.device),
            
            # 身体位置误差：所有跟踪身体部位与参考运动对应部位之间的平均位置距离
            "error_body_pos": torch.zeros(self.num_envs, device=self.device),
            
            # 身体旋转误差：所有跟踪身体部位与参考运动对应部位之间的平均旋转差异
            "error_body_rot": torch.zeros(self.num_envs, device=self.device),
            
            # 关节位置误差：机器人关节位置与参考运动关节位置之间的差异
            "error_joint_pos": torch.zeros(self.num_envs, device=self.device),
            
            # 关节速度误差：机器人关节速度与参考运动关节速度之间的差异
            "error_joint_vel": torch.zeros(self.num_envs, device=self.device),
            
            # 采样熵：自适应采样分布的熵值，反映采样策略的随机性（值越高表示越随机）
            "sampling_entropy": torch.zeros(self.num_envs, device=self.device),
            
            # 采样top1概率：自适应采样中最可能被选中的时间区间的概率
            "sampling_top1_prob": torch.zeros(self.num_envs, device=self.device),
            
            # 采样top1区间：自适应采样中最可能被选中的时间区间（归一化到[0,1]）
            "sampling_top1_bin": torch.zeros(self.num_envs, device=self.device),
        }

    @property
    def command(self) -> torch.Tensor:
        """返回当前命令（观察值），包含关节位置和速度"""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """从运动数据中获取当前时间步的关节位置"""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """从运动数据中获取当前时间步的关节速度"""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """获取身体部位在世界坐标系中的位置（考虑环境原点偏移）"""
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """获取身体部位在世界坐标系中的四元数姿态"""
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """获取身体部位在世界坐标系中的线速度"""
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """获取身体部位在世界坐标系中的角速度"""
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """获取锚点身体在世界坐标系中的位置"""
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """获取锚点身体的四元数姿态"""
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """获取锚点身体的线速度"""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """获取锚点身体的角速度"""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """获取机器人的实际关节位置"""
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """获取机器人的实际关节速度"""
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """获取机器人身体部位的实际位置"""
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """获取机器人身体部位的实际姿态"""
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """获取机器人身体部位的实际线速度"""
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """获取机器人身体部位的实际角速度"""
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """获取机器人锚点身体的实际位置"""
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """获取机器人锚点身体的实际姿态"""
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """获取机器人锚点身体的实际线速度"""
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """获取机器人锚点身体的实际角速度"""
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        """更新所有性能指标"""
        # 锚点位置误差：欧几里得距离
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        # 锚点旋转误差：四元数差异度量
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        # 锚点线速度误差
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        # 锚点角速度误差
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        # 身体部位位置误差：所有身体部位位置误差的平均值
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        # 身体部位旋转误差：所有身体部位旋转误差的平均值
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        # 关节位置误差：所有关节位置差异的范数
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        # 关节速度误差：所有关节速度差异的范数
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """自适应采样策略：根据失败历史调整采样概率"""
        # 检查哪些环境在当前episode失败了
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # 计算失败时对应的时间区间
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            # 统计每个时间区间的失败次数
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # 计算采样概率：结合失败历史和均匀分布
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        # 对概率分布进行卷积平滑
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # 非因果卷积核
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # 根据概率分布采样时间区间
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        # 在选定的时间区间内随机采样具体时间步
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # 计算采样相关的指标
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()  # 熵
        H_norm = H / math.log(self.bin_count)  # 归一化熵
        pmax, imax = sampling_probabilities.max(dim=0)  # 最大概率及其索引
        
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        """为指定环境重新采样运动命令"""
        if len(env_ids) == 0:
            return
        # 使用自适应采样策略
        self._adaptive_sampling(env_ids)

        # 获取根身体（第一个身体部位）的状态
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # 对位置和姿态添加随机扰动
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        
        # 对速度添加随机扰动
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # 对关节位置添加随机扰动，并限制在合理范围内
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        
        # 将采样的状态设置到仿真环境中
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """更新运动命令：推进时间步并处理运动序列结束的情况"""
        # 时间步前进
        self.time_steps += 1
        # 找出运动序列已结束的环境
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        # 为结束的环境重新采样命令
        self._resample_command(env_ids)

        # 计算身体部位相对于锚点的相对位姿
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        # 计算位置和姿态的变换
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]  # 保持高度一致
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        # 应用变换得到相对位姿
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # 更新失败统计（指数移动平均）
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()  # 重置当前计数

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化"""
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                # 创建锚点身体的可视化器
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                # 创建各身体部位的可视化器
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            # 显示所有可视化器
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            # 隐藏所有可视化器
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        """调试可视化回调函数"""
        if not self.robot.is_initialized:
            return

        # 可视化当前和目标的锚点身体
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        # 可视化当前和目标的身体部位
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """运动命令的配置类"""

    class_type: type = MotionCommand

    asset_name: str = MISSING  # 机器人资源名称
    motion_file: str = MISSING  # 运动数据文件路径
    anchor_body_name: str = MISSING  # 锚点身体名称
    body_names: list[str] = MISSING  # 需要跟踪的身体部位名称列表

    pose_range: dict[str, tuple[float, float]] = {}  # 位姿扰动范围
    velocity_range: dict[str, tuple[float, float]] = {}  # 速度扰动范围

    joint_position_range: tuple[float, float] = (-0.52, 0.52)  # 关节位置扰动范围

    # 自适应采样参数
    adaptive_kernel_size: int = 1  # 卷积核大小
    adaptive_lambda: float = 0.8  # 衰减因子
    adaptive_uniform_ratio: float = 0.1  # 均匀分布比例
    adaptive_alpha: float = 0.001  # 指数移动平均系数

    # 可视化配置
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)  # 锚点可视化尺度

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # 身体部位可视化尺度