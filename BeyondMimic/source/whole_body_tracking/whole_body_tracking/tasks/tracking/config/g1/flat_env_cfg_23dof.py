from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.robots.g1_23dof import G1_23DOF_ACTION_SCALE, G1_23DOF_CYLINDER_CFG
 
@configclass
class G1_23dof_FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_23DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_23DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            # "left_wrist_yaw_link",
            "left_wrist_roll_rubber_hand",
            "right_shoulder_roll_link",
            "right_elbow_link",
            # "right_wrist_yaw_link",
            "right_wrist_roll_rubber_hand",
        ]


@configclass
class G1_23dof_FlatWoStateEstimationEnvCfg(G1_23dof_FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1_23dof_FlatLowFreqEnvCfg(G1_23dof_FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
