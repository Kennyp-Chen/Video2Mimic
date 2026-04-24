"""
GVHMR to Robot Motion Retargeting Script

This script retargets human motion from GVHMR (Global Vision-based Human Motion Reconstruction)
to various robot models. It processes SMPLX motion data and converts it to robot-specific
joint trajectories with ground contact correction and optional padding.

Features:
- Retargets SMPLX human motion to robot DOF trajectories
- Ground contact correction to ensure feet stay on the ground
- Smooth ground alignment with configurable smoothing window
- Optional padding with default pose at start/end of motion
- Customizable default facing angle for the default pose
- Video recording for visualization

Usage:
    python scripts/gvhmr2robot_fix.py \\
        --gvhmr_pred_file <path_to_hmr4d_results.pt> \\
        --robot <robot_type> \\
        --ground_contact \\
        --pad_default_pose \\
        --pad_frames <transition_frames> \\
        --pad_hold_frames <hold_frames> \\
        --default_facing_angle <start_angle,end_angle> \\
        --record_video \\
        --save_path <output_path.pkl>

Arguments:
    --gvhmr_pred_file: Path to GVHMR prediction file (.pt)
    --robot: Robot type (e.g., unitree_g1, unitree_g1_23dof)
    --ground_contact: Enable ground contact correction
    --ground_contact_mode: per_frame or global ground correction
    --ground_contact_bodies: Comma-separated body names for ground contact
    --ground_contact_smooth_window: Smoothing window for ground offsets
    --smooth_ground_alignment: Enable smooth transitions for ground alignment
    --pad_default_pose: Pad motion with default pose at start/end
    --pad_frames: Number of transition frames for padding
    --pad_hold_frames: Number of hold frames for default pose
    --default_facing_angle: Default facing angles in degrees for start and end poses (rotation around Z), e.g., 90 0
    --record_video: Record video of the retargeted motion
    --save_path: Path to save the output robot motion (.pkl)
    --loop: Loop the motion
    --rate_limit: Limit rate to match human motion FPS

Example:
    # Basic retargeting with ground contact
    python scripts/gvhmr2robot_fix.py \\
        --gvhmr_pred_file GVHMR/outputs/demo/gangster_dance/hmr4d_results.pt \\
        --robot unitree_g1 \\
        --ground_contact \\
        --save_path motions/G1/gangster_dance.pkl

    # With padding and custom facing angles (start=90deg, end=0deg)
    python scripts/gvhmr2robot_fix.py \\
        --gvhmr_pred_file GVHMR/outputs/demo/gangster_dance/hmr4d_results.pt \\
        --robot unitree_g1 \\
        --ground_contact \\
        --pad_default_pose \\
        --pad_frames 15 \\
        --pad_hold_frames 30 \\
        --default_facing_angle 90 0 \\
        --record_video \\
        --save_path motions/G1/gangster_dance.pkl

中文说明：
    GVHMR到机器人动作重定向脚本

    该脚本将GVHMR（全局视觉人体运动重建）的人体动作重定向到各种机器人模型。
    它处理SMPLX运动数据并将其转换为机器人特定的关节轨迹，支持地面接触修正和可选的填充。

    功能特性：
    - 将SMPLX人体运动重定向为机器人DOF轨迹
    - 地面接触修正，确保足部保持在地面
    - 可配置平滑窗口的平滑地面对齐
    - 可选的在动作开始/结束时填充默认姿态
    - 可自定义默认姿态的面朝方向角度
    - 支持视频录制用于可视化

    使用方法：
        python scripts/gvhmr2robot_fix.py \\
            --gvhmr_pred_file <hmr4d_results.pt路径> \\
            --robot <机器人类型> \\
            --ground_contact \\
            --pad_default_pose \\
            --pad_frames <过渡帧数> \\
            --pad_hold_frames <保持帧数> \\
            --default_facing_angle <起始角度,结束角度> \\
            --record_video \\
            --save_path <输出路径.pkl>

    参数说明：
        --gvhmr_pred_file: GVHMR预测文件路径 (.pt)
        --robot: 机器人类型 (如 unitree_g1, unitree_g1_23dof)
        --ground_contact: 启用地面接触修正
        --ground_contact_mode: per_frame 或 global 地面修正模式
        --ground_contact_bodies: 用于地面接触的body名称，逗号分隔
        --ground_contact_smooth_window: 地面偏移的平滑窗口
        --smooth_ground_alignment: 启用地面对齐的平滑过渡
        --pad_default_pose: 在动作开始/结束时填充默认姿态
        --pad_frames: 填充的过渡帧数
        --pad_hold_frames: 默认姿态的保持帧数
        --default_facing_angle: 默认面朝角度（度，绕Z轴旋转），两个值分别对应起始和结束姿态，例如 90 0
        --record_video: 录制重定向动作的视频
        --save_path: 保存输出机器人动作的路径 (.pkl)
        --loop: 循环播放动作
        --rate_limit: 限制速率以匹配人体运动FPS

    示例：
        # 基础重定向，带地面接触修正
        python scripts/gvhmr2robot_fix.py \\
            --gvhmr_pred_file GVHMR/outputs/demo/gangster_dance/hmr4d_results.pt \\
            --robot unitree_g1 \\
            --ground_contact \\
            --save_path motions/G1/gangster_dance.pkl

        # 带填充和自定义面朝角度（起始90度，结束0度）
        python scripts/gvhmr2robot_fix.py \\
            --gvhmr_pred_file GVHMR/outputs/demo/gangster_dance/hmr4d_results.pt \\
            --robot unitree_g1 \\
            --ground_contact \\
            --pad_default_pose \\
            --pad_frames 15 \\
            --pad_hold_frames 30 \\
            --default_facing_angle 90 0 \\
            --record_video \\
            --save_path motions/G1/gangster_dance.pkl
"""

import argparse
import pathlib
import os
import time

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print

arti_offset = -0.04
# arti_offset = 0.1

# 安全的根部高度阈值，低于此值认为是蹲下状态
SAFE_ROOT_HEIGHT = 0.81
CROUCH_THRESHOLD = 0.75  # 低于此值使用安全高度


def slerp_quat(q1, q2, t):
    """
    球面线性插值四元数
    
    Args:
        q1: 起始四元数 (w, x, y, z)
        q2: 目标四元数 (w, x, y, z)
        t: 插值参数 [0, 1]
    
    Returns:
        插值后的四元数
    """
    # 确保四元数归一化
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算点积
    dot = np.dot(q1, q2)
    
    # 如果点积为负，取反其中一个四元数以确保最短路径
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # 计算插值角度
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2


def get_safe_root_height(current_z):
    """
    获取安全的根部高度，防止蹲下状态时穿过地面

    Args:
        current_z: 当前根部Z轴高度

    Returns:
        安全的Z轴高度
    """
    if current_z < CROUCH_THRESHOLD:
        print(f"Warning: Current root height {current_z:.3f}m is below "
              f"threshold {CROUCH_THRESHOLD}m, using safe height "
              f"{SAFE_ROOT_HEIGHT}m to prevent ground penetration")
        return SAFE_ROOT_HEIGHT
    return current_z


def yaw_angle_to_quat(yaw_angle_degrees):
    """
    将偏航角（绕Z轴旋转）转换为四元数

    Args:
        yaw_angle_degrees: 偏航角（度）

    Returns:
        四元数 (x, y, z, w) 格式
    """
    # 转换为弧度
    yaw = np.radians(yaw_angle_degrees)
    # 计算四元数 (绕Z轴旋转)
    # q = [sin(yaw/2)*0, sin(yaw/2)*0, sin(yaw/2)*1, cos(yaw/2)]
    #   = [0, 0, sin(yaw/2), cos(yaw/2)]
    half_yaw = yaw / 2.0
    return np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)])


def smooth_transition_to_default(last_dof, last_root_pos, last_root_rot,
                                  default_pos, default_root_pos_end,
                                  default_root_rot, n_frames):
    """
    平滑过渡到默认姿态，避免足部离地

    Args:
        last_dof: 最后一帧的DOF位置
        last_root_pos: 最后一帧的根部位置
        last_root_rot: 最后一帧的根部旋转 (xyzw格式)
        default_pos: 默认DOF位置
        default_root_pos_end: 默认根部位置
        default_root_rot: 默认根部旋转 (xyzw格式)
        n_frames: 过渡帧数

    Returns:
        trans_dof: 过渡DOF位置
        trans_root_pos: 过渡根部位置
        trans_root_rot: 过渡根部旋转
    """
    # 转换四元数格式 xyzw -> (w, x, y, z) 用于scipy
    last_root_rot_wxyz = np.array([last_root_rot[3], last_root_rot[0], last_root_rot[1], last_root_rot[2]])
    default_root_rot_wxyz = np.array([default_root_rot[3], default_root_rot[0], default_root_rot[1], default_root_rot[2]])
    
    # 时间参数
    ts = np.linspace(0.0, 1.0, n_frames + 1)[1:]  # 从0到1，不包括0
    
    # DOF插值 - 使用更平滑的插值函数
    trans_dof = []
    for t in ts:
        # 使用sin函数进行更平滑的过渡
        smooth_t = (1 - np.cos(t * np.pi)) / 2  # smoothstep
        frame_dof = last_dof * (1 - smooth_t) + default_pos * smooth_t
        trans_dof.append(frame_dof)
    trans_dof = np.array(trans_dof)
    
    # 根部位置插值 - Z轴保持相对稳定
    trans_root_pos = []
    last_z = last_root_pos[2]
    default_z = default_root_pos_end[2]

    for t in ts:
        smooth_t = (1 - np.cos(t * np.pi)) / 2
        # X, Y线性插值，Z轴使用更保守的插值
        x = last_root_pos[0] * (1 - smooth_t) + default_root_pos_end[0] * smooth_t
        y = last_root_pos[1] * (1 - smooth_t) + default_root_pos_end[1] * smooth_t
        # Z轴插值：如果最后一帧已经接近地面，保持稳定；否则缓慢变化
        if abs(last_z - default_z) < 0.05:  # 如果已经很接近
            z = last_z  # 保持稳定
        else:
            z = last_z * (1 - smooth_t) + default_z * smooth_t
        trans_root_pos.append([x, y, z])
    trans_root_pos = np.array(trans_root_pos)
    
    # 根部旋转插值 - 使用SLERP
    trans_root_rot = []
    for t in ts:
        smooth_t = (1 - np.cos(t * np.pi)) / 2
        interp_rot = slerp_quat(last_root_rot_wxyz, default_root_rot_wxyz, smooth_t)
        # 转换回 xyzw 格式
        interp_rot_xyzw = np.array([interp_rot[1], interp_rot[2], interp_rot[3], interp_rot[0]])
        trans_root_rot.append(interp_rot_xyzw)
    trans_root_rot = np.array(trans_root_rot)
    
    return trans_dof, trans_root_pos, trans_root_rot


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_g1_23dof",],
                 #"unitree_h1", "unitree_h1_2",
                #  "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                # "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                # "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--ground_contact",
        default=False,
        action="store_true",
        help=(
            "Shift root z so at least one foot/toe lowest point touches the "
            "ground (z=0)."
        ),
    )

    parser.add_argument(
        "--ground_contact_mode",
        choices=["per_frame", "global"],
        default="per_frame",
        help=(
            "Ground contact correction mode: per_frame (each frame) or global "
            "(single offset)."
        ),
    )

    parser.add_argument(
        "--ground_contact_bodies",
        type=str,
        default=None,
        help=(
            "Comma-separated MuJoCo body names to use as feet/toes for ground "
            "contact (optional)."
        ),
    )

    parser.add_argument(
        "--ground_contact_smooth_window",
        type=int,
        default=1,
        help=(
            "Optional smoothing window (frames) for per-frame ground offsets. "
            "1 disables smoothing."
        ),
    )

    parser.add_argument(
        "--smooth_ground_alignment",
        default=False,
        action="store_true",
        help="Enable smooth transitions for ground alignment.",
    )

    parser.add_argument(
        "--pad_default_pose",
        default=False,
        action="store_true",
        help="Pad motion with default pose at start/end when saving.",
    )

    parser.add_argument(
        "--pad_hold_frames",
        type=int,
        default=0,
        help="Number of hold frames for --pad_default_pose.",
    )

    parser.add_argument(
        "--pad_frames",
        type=int,
        default=10,
        help="Number of transition frames for --pad_default_pose.",
    )

    parser.add_argument(
        "--default_facing_angle",
        type=float,
        nargs=2,
        default=None,
        help=(
            "Default facing angles in degrees for start and end poses "
            "(rotation around Z axis). Provide two values: start_angle end_angle. "
            "If not specified, uses identity quaternion (no rotation)."
        ),
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=(
            f"videos/{args.robot}_"
            f"{args.gvhmr_pred_file.split('/')[-1].split('.')[0]}.mp4"
        ),
    )

    ground_model = None
    ground_data = None
    ground_body_ids = None
    global_ground_offset = None
    raw_qpos_list = []
    per_frame_dz_list = []

    if args.ground_contact:
        ground_model = mj.MjModel.from_xml_path(retarget.xml_file)
        ground_data = mj.MjData(ground_model)

        if (
            args.ground_contact_bodies is not None
            and args.ground_contact_bodies.strip() != ""
        ):
            selected_names = [
                n.strip()
                for n in args.ground_contact_bodies.split(",")
                if n.strip()
            ]
        else:
            all_body_names = [
                mj.mj_id2name(ground_model, mj.mjtObj.mjOBJ_BODY, i)
                for i in range(ground_model.nbody)
            ]
            toe_candidates = [
                n
                for n in all_body_names
                if n is not None and "toe" in n.lower()
            ]
            foot_candidates = [
                n
                for n in all_body_names
                if n is not None and "foot" in n.lower()
            ]
            ankle_candidates = [
                n
                for n in all_body_names
                if n is not None and "ankle" in n.lower()
            ]
            if len(toe_candidates) > 0:
                selected_names = toe_candidates
            elif len(foot_candidates) > 0:
                selected_names = foot_candidates
            else:
                selected_names = ankle_candidates

        ground_body_ids = []
        for name in selected_names:
            try:
                body_id = ground_model.body(name).id
            except Exception:
                continue
            ground_body_ids.append(body_id)

        if len(ground_body_ids) == 0:
            raise ValueError(
                "ground_contact enabled but no valid foot/toe bodies found. "
                "Please pass --ground_contact_bodies with valid MuJoCo body "
                "names."
            )


    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    # If pad_default_pose is enabled, we need to render the padded frames
    if args.pad_default_pose and args.save_path is not None:
        # First, collect all the data
        all_raw_qpos_list = []
        all_per_frame_dz_list = []
        
        while True:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break
            
            # Update task targets.
            smplx_data = smplx_data_frames[i]

            # retarget
            qpos = retarget.retarget(smplx_data)
            # all_raw_qpos_list.append(qpos.copy())

            if args.ground_contact:
                ground_data.qpos[:] = qpos
                mj.mj_forward(ground_model, ground_data)
                min_z = float(np.min(ground_data.xpos[ground_body_ids, 2])) + arti_offset
                # print(f"Min z: {min_z}")
                
                if args.ground_contact_mode == "global":
                    if global_ground_offset is None:
                        global_ground_offset = -min_z
                    qpos = qpos.copy()
                    qpos[2] = qpos[2] + global_ground_offset
                else:
                    dz = -min_z
                    all_per_frame_dz_list.append(dz)
                    qpos = qpos.copy()
                    qpos[2] = qpos[2] + dz
            all_raw_qpos_list.append(qpos.copy())
        
        # Apply padding logic to create the complete sequence
        raw_qpos_list = all_raw_qpos_list
        per_frame_dz_list = all_per_frame_dz_list
        
        # Process ground contact smoothing if needed
        if (
            args.ground_contact
            and args.ground_contact_mode == "per_frame"
            and args.ground_contact_smooth_window > 1
        ):
            if len(per_frame_dz_list) != len(raw_qpos_list):
                raise RuntimeError(
                    "ground_contact smoothing internal error: dz list length "
                    "mismatch"
                )

            dz = np.asarray(per_frame_dz_list, dtype=float)
            w = int(args.ground_contact_smooth_window)
            kernel = np.ones(w, dtype=float) / float(w)
            padded = np.pad(dz, (w // 2, w - 1 - w // 2), mode="edge")
            smooth_dz = np.convolve(padded, kernel, mode="valid")

            qpos_arr = np.stack(raw_qpos_list, axis=0)
            qpos_arr[:, 2] = qpos_arr[:, 2] + smooth_dz
            qpos_list = [qpos_arr[i].copy() for i in range(qpos_arr.shape[0])]
        else:
            qpos_list = raw_qpos_list

        # Extract root and dof positions
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        # Apply padding
        if args.robot not in ["unitree_g1", "unitree_g1_23dof"]:
            raise ValueError(
                "--pad_default_pose is currently only supported for "
                "unitree_g1 and unitree_g1_23dof"
            )

        if args.pad_frames <= 0:
            raise ValueError("--pad_frames must be > 0")

        # Default pose for different robot types
        if args.robot == "unitree_g1_23dof":
            # 23dof version: remove waist_roll, waist_pitch,
            # left_wrist_pitch, left_wrist_yaw,
            # right_wrist_pitch, right_wrist_yaw
            default_pos = np.array(
                [
                    # Left leg (6 joints)
                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    # Right leg (6 joints)
                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    # Waist (1 joint: only waist_yaw)
                    0.0,
                    # Left arm (5 joints: shoulder_pitch, shoulder_roll,
                    # shoulder_yaw, elbow, wrist_roll)
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    # Right arm (5 joints: shoulder_pitch, shoulder_roll,
                    # shoulder_yaw, elbow, wrist_roll)
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                dtype=float,
            )
        else:  # unitree_g1 (29dof)
            default_pos = np.array(
                [
                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                dtype=float,
            )

        if dof_pos.ndim != 2:
            raise ValueError(
                f"Expected dof_pos shape (T, D), got {dof_pos.shape}"
            )

        if default_pos.shape[0] != dof_pos.shape[1]:
            raise ValueError(
                "default_pos length does not match dof_pos dimension: "
                f"{default_pos.shape[0]} != {dof_pos.shape[1]}"
            )

        n = args.pad_frames
        hold_frames = args.pad_hold_frames
        first_dof = dof_pos[0]
        last_dof = dof_pos[-1]

        # Create padding sequences with hold frames
        pre_dof_list = []
        post_dof_list = []
        pre_root_pos_list = []
        post_root_pos_list = []
        pre_root_rot_list = []
        post_root_rot_list = []

        # Default root position and rotation
        first_frame_safe_z = get_safe_root_height(root_pos[0][2])
        last_frame_safe_z = get_safe_root_height(root_pos[-1][2])
        default_root_pos_start = np.array([root_pos[0][0], root_pos[0][1], first_frame_safe_z])
        default_root_pos_end = np.array([root_pos[-1][0], root_pos[-1][1], last_frame_safe_z])

        # Set default root rotation based on user input or current frame
        if args.default_facing_angle is not None:
            start_angle, end_angle = args.default_facing_angle
            default_root_rot_start = yaw_angle_to_quat(start_angle)
            default_root_rot_end = yaw_angle_to_quat(end_angle)
            print(f"Using custom facing angles: start={start_angle} degrees, "
                  f"end={end_angle} degrees")
        else:
            # Use identity quaternion (no rotation)
            default_root_rot_start = np.array([0.0, 0.0, 0.0, 1.0])
            default_root_rot_end = np.array([0.0, 0.0, 0.0, 1.0])

        # Prepend sequence: hold default -> transition to first frame
        if hold_frames > 0:
            # Hold default pose for specified frames
            for _ in range(hold_frames):
                pre_dof_list.append(default_pos.copy())
                pre_root_pos_list.append(default_root_pos_start.copy())
                pre_root_rot_list.append(default_root_rot_start.copy())

        # Transition frames from default to first frame (always n frames)
        pre_trans_dof, pre_trans_root_pos, pre_trans_root_rot = \
            smooth_transition_to_default(
                default_pos, default_root_pos_start, default_root_rot_start,
                first_dof, root_pos[0], root_rot[0], n
            )
        pre_dof_list.extend(pre_trans_dof)
        pre_root_pos_list.extend(pre_trans_root_pos)
        pre_root_rot_list.extend(pre_trans_root_rot)

        # Append sequence: transition from last frame -> hold default
        post_trans_dof, post_trans_root_pos, post_trans_root_rot = \
            smooth_transition_to_default(
                last_dof, root_pos[-1], root_rot[-1],
                default_pos, default_root_pos_end, default_root_rot_end, n
            )
        post_dof_list.extend(post_trans_dof)
        post_root_pos_list.extend(post_trans_root_pos)
        post_root_rot_list.extend(post_trans_root_rot)

        if hold_frames > 0:
            # Hold default pose for specified frames
            for _ in range(hold_frames):
                post_dof_list.append(default_pos.copy())
                post_root_pos_list.append(default_root_pos_end.copy())
                post_root_rot_list.append(default_root_rot_end.copy())

        # Concatenate all sequences
        if pre_dof_list:
            pre_dof = np.array(pre_dof_list)
            pre_root_pos = np.array(pre_root_pos_list)
            pre_root_rot = np.array(pre_root_rot_list)
        else:
            pre_dof = np.empty((0, len(default_pos)))
            pre_root_pos = np.empty((0, 3))
            pre_root_rot = np.empty((0, 4))
            
        if post_dof_list:
            post_dof = np.array(post_dof_list)
            post_root_pos = np.array(post_root_pos_list)
            post_root_rot = np.array(post_root_rot_list)
        else:
            post_dof = np.empty((0, len(default_pos)))
            post_root_pos = np.empty((0, 3))
            post_root_rot = np.empty((0, 4))

        dof_pos = np.concatenate([pre_dof, dof_pos, post_dof], axis=0)
        root_pos = np.concatenate([pre_root_pos, root_pos, post_root_pos], axis=0)
        root_rot = np.concatenate([pre_root_rot, root_rot, post_root_rot], axis=0)

        # Now render the padded sequence
        for i in range(len(root_pos)):
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time
            
            # Reconstruct qpos for this frame
            qpos = np.concatenate([
                root_pos[i],
                root_rot[i][[3, 0, 1, 2]],  # Convert back from xyzw to wxyz
                dof_pos[i]
            ])
            
            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=None,  # Don't show human during padded frames
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
            )
    else:
        # Original rendering logic for non-padded case
        while True:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break
            
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time
            
            # Update task targets.
            smplx_data = smplx_data_frames[i]

            # retarget
            qpos = retarget.retarget(smplx_data)

            if args.save_path is not None:
                raw_qpos_list.append(qpos.copy())

            if args.ground_contact:
                ground_data.qpos[:] = qpos
                mj.mj_forward(ground_model, ground_data)
                min_z = float(np.min(ground_data.xpos[ground_body_ids, 2])) + arti_offset
                print(f"Min z: {min_z}")
                
                if args.ground_contact_mode == "global":
                    if global_ground_offset is None:
                        global_ground_offset = -min_z
                    qpos = qpos.copy()
                    qpos[2] = qpos[2] + global_ground_offset
                else:
                    dz = -min_z
                    if args.save_path is not None:
                        per_frame_dz_list.append(dz)
                    qpos = qpos.copy()
                    qpos[2] = qpos[2] + dz

            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                # human_motion_data=smplx_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
            )
            if args.save_path is not None:
                qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle

        if (
            args.ground_contact
            and args.ground_contact_mode == "per_frame"
            and args.ground_contact_smooth_window > 1
        ):
            if len(per_frame_dz_list) != len(raw_qpos_list):
                raise RuntimeError(
                    "ground_contact smoothing internal error: dz list length "
                    "mismatch"
                )

            dz = np.asarray(per_frame_dz_list, dtype=float)
            w = int(args.ground_contact_smooth_window)
            kernel = np.ones(w, dtype=float) / float(w)
            padded = np.pad(dz, (w // 2, w - 1 - w // 2), mode="edge")
            smooth_dz = np.convolve(padded, kernel, mode="valid")

            qpos_arr = np.stack(raw_qpos_list, axis=0)
            qpos_arr[:, 2] = qpos_arr[:, 2] + smooth_dz
            qpos_list = [qpos_arr[i].copy() for i in range(qpos_arr.shape[0])]
        else:
            qpos_list = raw_qpos_list

        # Extract root and dof positions
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        if args.pad_default_pose:
            if args.robot not in ["unitree_g1", "unitree_g1_23dof"]:
                raise ValueError(
                    "--pad_default_pose is currently only supported for "
                    "unitree_g1 and unitree_g1_23dof"
                )

            if args.pad_frames <= 0:
                raise ValueError("--pad_frames must be > 0")

            # Default pose for different robot types
            if args.robot == "unitree_g1_23dof":
                # 23dof version: remove waist_roll, waist_pitch,
                # left_wrist_pitch, left_wrist_yaw,
                # right_wrist_pitch, right_wrist_yaw
                default_pos = np.array(
                    [
                        # Left leg (6 joints)
                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                        # Right leg (6 joints)
                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                        # Waist (1 joint: only waist_yaw)
                        0.0,
                        # Left arm (5 joints: shoulder_pitch, shoulder_roll,
                        # shoulder_yaw, elbow, wrist_roll)
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        # Right arm (5 joints: shoulder_pitch, shoulder_roll,
                        # shoulder_yaw, elbow, wrist_roll)
                        0.0, 0.0, 0.0, 0.0, 0.0,
                    ],
                    dtype=float,
                )
            else:  # unitree_g1 (29dof)
                default_pos = np.array(
                    [
                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ],
                    dtype=float,
                )

            if dof_pos.ndim != 2:
                raise ValueError(
                    f"Expected dof_pos shape (T, D), got {dof_pos.shape}"
                )

            if default_pos.shape[0] != dof_pos.shape[1]:
                raise ValueError(
                    "default_pos length does not match dof_pos dimension: "
                    f"{default_pos.shape[0]} != {dof_pos.shape[1]}"
                )

            n = args.pad_frames
            hold_frames = args.pad_hold_frames
            first_dof = dof_pos[0]
            last_dof = dof_pos[-1]

            # Create padding sequences with hold frames
            pre_dof_list = []
            post_dof_list = []
            pre_root_pos_list = []
            post_root_pos_list = []
            pre_root_rot_list = []
            post_root_rot_list = []

            # Default root position and rotation
            first_frame_safe_z = get_safe_root_height(root_pos[0][2])
            last_frame_safe_z = get_safe_root_height(root_pos[-1][2])
            default_root_pos_start = np.array([root_pos[0][0], root_pos[0][1], first_frame_safe_z])
            default_root_pos_end = np.array([root_pos[-1][0], root_pos[-1][1], last_frame_safe_z])

            # Set default root rotation based on user input or current frame
            if args.default_facing_angle is not None:
                start_angle, end_angle = args.default_facing_angle
                default_root_rot_start = yaw_angle_to_quat(start_angle)
                default_root_rot_end = yaw_angle_to_quat(end_angle)
                print(f"Using custom facing angles: start={start_angle} degrees, "
                      f"end={end_angle} degrees")
            else:
                # Use identity quaternion (no rotation)
                default_root_rot_start = np.array([0.0, 0.0, 0.0, 1.0])
                default_root_rot_end = np.array([0.0, 0.0, 0.0, 1.0])

            # Prepend sequence: hold default -> transition to first frame
            if hold_frames > 0:
                # Hold default pose for specified frames
                for _ in range(hold_frames):
                    pre_dof_list.append(default_pos.copy())
                    pre_root_pos_list.append(default_root_pos_start.copy())
                    pre_root_rot_list.append(default_root_rot_start.copy())

            # Transition frames from default to first frame (always n frames)
            pre_trans_dof, pre_trans_root_pos, pre_trans_root_rot = \
                smooth_transition_to_default(
                    default_pos, default_root_pos_start, default_root_rot_start,
                    first_dof, root_pos[0], root_rot[0], n
                )
            pre_dof_list.extend(pre_trans_dof)
            pre_root_pos_list.extend(pre_trans_root_pos)
            pre_root_rot_list.extend(pre_trans_root_rot)

            # Append sequence: transition from last frame -> hold default
            post_trans_dof, post_trans_root_pos, post_trans_root_rot = \
                smooth_transition_to_default(
                    last_dof, root_pos[-1], root_rot[-1],
                    default_pos, default_root_pos_end, default_root_rot_end, n
                )
            post_dof_list.extend(post_trans_dof)
            post_root_pos_list.extend(post_trans_root_pos)
            post_root_rot_list.extend(post_trans_root_rot)

            # Hold default pose for specified frames
            if hold_frames > 0:
                for _ in range(hold_frames):
                    post_dof_list.append(default_pos.copy())
                    post_root_pos_list.append(default_root_pos_end.copy())
                    post_root_rot_list.append(default_root_rot_end.copy())

            # Concatenate all sequences
            if pre_dof_list:
                pre_dof = np.array(pre_dof_list)
                pre_root_pos = np.array(pre_root_pos_list)
                pre_root_rot = np.array(pre_root_rot_list)
            else:
                pre_dof = np.empty((0, len(default_pos)))
                pre_root_pos = np.empty((0, 3))
                pre_root_rot = np.empty((0, 4))
                
            if post_dof_list:
                post_dof = np.array(post_dof_list)
                post_root_pos = np.array(post_root_pos_list)
                post_root_rot = np.array(post_root_rot_list)
            else:
                post_dof = np.empty((0, len(default_pos)))
                post_root_pos = np.empty((0, 3))
                post_root_rot = np.empty((0, 4))

            dof_pos = np.concatenate([pre_dof, dof_pos, post_dof], axis=0)
            root_pos = np.concatenate([pre_root_pos, root_pos, post_root_pos], axis=0)
            root_rot = np.concatenate([pre_root_rot, root_rot, post_root_rot], axis=0)

        local_body_pos = None
        body_names = None

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    robot_motion_viewer.close()
