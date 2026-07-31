"""
G1 环境配置 - 无状态估计 + 综合稳定性优化
结合 WoStateEstimation 观测和稳定性改进
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import G1FlatWoStateEstimationEnvCfg
from whole_body_tracking.tasks.tracking import mdp


@configclass
class G1FlatWoStateStableEnvCfg(G1FlatWoStateEstimationEnvCfg):
    """无状态估计 + 综合稳定性优化配置"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # === Reward 调整 ===
        # 增加方向跟踪权重（对于无状态估计更重要）
        self.rewards.motion_global_anchor_ori.weight = 1.2
        self.rewards.motion_body_ori.weight = 1.3
        
        # 增加位置跟踪权重
        self.rewards.motion_global_anchor_pos.weight = 0.8
        
        # 增加动作平滑性（无状态估计时更需要平滑）
        self.rewards.action_rate_l2.weight = -0.2
        
        # === Domain 随机化调整 ===
        # 适度增加质心随机化
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {
                    "x": (-0.04, 0.04),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05)
                },
            },
        )
        
        # 适度增加推力
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.8, 2.5),
            params={
                "velocity_range": {
                    "x": (-0.6, 0.6),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.6, 0.6),  # 增加俯仰扰动
                    "yaw": (-0.78, 0.78),
                }
            },
        )


@configclass
class G1FlatWoStateForwardBiasEnvCfg(G1FlatWoStateEstimationEnvCfg):
    """无状态估计 + 前倾偏置优化配置"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 只调整 reward 权重
        self.rewards.motion_global_anchor_ori.weight = 1.0
        self.rewards.motion_body_ori.weight = 1.5
        self.rewards.action_rate_l2.weight = -0.15


@configclass
class G1FlatWoStateRobustEnvCfg(G1FlatWoStateEstimationEnvCfg):
    """无状态估计 + 鲁棒性增强配置"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 增强质心随机化
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05)
                },
            },
        )
        
        # 增加推力频率和强度
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.5, 2.0),
            params={
                "velocity_range": {
                    "x": (-0.8, 0.8),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                }
            },
        )
