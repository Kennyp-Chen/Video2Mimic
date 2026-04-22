import argparse
import pathlib
import os
import time

import numpy as np
import mujoco as mj

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print

arti_offset = -0.04
# arti_offset = 0.1


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
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "unitree_g1_23dof",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
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
        "--pad_frames",
        type=int,
        default=10,
        help="Number of padding frames for --pad_default_pose.",
    )

    parser.add_argument(
        "--pad_hold_frames",
        type=int,
        default=0,
        help="Number of frames to hold default pose at start/end "
             "(before/after transition).",
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
                print(f"Min z: {min_z}")
                
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
        default_root_pos_start = np.array([root_pos[0][0], root_pos[0][1], 0.81])  # X,Y from first frame, Z=0.81
        default_root_pos_end = np.array([root_pos[-1][0], root_pos[-1][1], 0.81])  # X,Y from last frame, Z=0.81
        default_root_rot = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion (w=1)

        # Prepend sequence: hold default -> transition to first frame
        if hold_frames > 0:
            # Hold default pose for specified frames
            for _ in range(hold_frames):
                pre_dof_list.append(default_pos.copy())
                pre_root_pos_list.append(default_root_pos_start.copy())
                pre_root_rot_list.append(default_root_rot.copy())
        
        # Transition frames from default to first frame (always n frames)
        pre_ts = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        pre_trans_dof = (
            default_pos[None, :] * (1.0 - pre_ts)
            + first_dof[None, :] * pre_ts
        )
        pre_dof_list.extend(pre_trans_dof)
        
        # Interpolate root position and rotation
        pre_trans_root_pos = (
            default_root_pos_start[None, :] * (1.0 - pre_ts)
            + root_pos[0:1] * pre_ts
        )
        pre_root_pos_list.extend(pre_trans_root_pos)
        
        pre_trans_root_rot = (
            default_root_rot[None, :] * (1.0 - pre_ts)
            + root_rot[0:1] * pre_ts
        )
        pre_root_rot_list.extend(pre_trans_root_rot)

        # Append sequence: transition from last frame -> hold default
        post_ts = (np.linspace(0.0, 1.0, n + 1)[1:])[:, None]
        post_trans_dof = (
            last_dof[None, :] * (1.0 - post_ts)
            + default_pos[None, :] * post_ts
        )
        post_dof_list.extend(post_trans_dof)
        
        # Interpolate root position and rotation
        post_trans_root_pos = (
            root_pos[-1:] * (1.0 - post_ts) + default_root_pos_end[None, :] * post_ts
        )
        post_root_pos_list.extend(post_trans_root_pos)
        
        post_trans_root_rot = (
            root_rot[-1:] * (1.0 - post_ts) + default_root_rot[None, :] * post_ts
        )
        post_root_rot_list.extend(post_trans_root_rot)
        
        # Hold default pose for specified frames
        if hold_frames > 0:
            for _ in range(hold_frames):
                post_dof_list.append(default_pos.copy())
                post_root_pos_list.append(default_root_pos_end.copy())
                post_root_rot_list.append(default_root_rot.copy())

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
            default_root_pos_start = np.array([root_pos[0][0], root_pos[0][1], 0.81])  # X,Y from first frame, Z=0.81
            default_root_pos_end = np.array([root_pos[-1][0], root_pos[-1][1], 0.81])  # X,Y from last frame, Z=0.81
            default_root_rot = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion

            # Prepend sequence: hold default -> transition to first frame
            if hold_frames > 0:
                # Hold default pose for specified frames
                for _ in range(hold_frames):
                    pre_dof_list.append(default_pos.copy())
                    pre_root_pos_list.append(default_root_pos_start.copy())
                    pre_root_rot_list.append(default_root_rot.copy())
            
            # Transition frames from default to first frame (always n frames)
            pre_ts = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
            pre_trans_dof = (
                default_pos[None, :] * (1.0 - pre_ts)
                + first_dof[None, :] * pre_ts
            )
            pre_dof_list.extend(pre_trans_dof)
            
            # Interpolate root position and rotation
            pre_trans_root_pos = (
                default_root_pos_start[None, :] * (1.0 - pre_ts)
                + root_pos[0:1] * pre_ts
            )
            pre_root_pos_list.extend(pre_trans_root_pos)
            
            pre_trans_root_rot = (
                default_root_rot[None, :] * (1.0 - pre_ts)
                + root_rot[0:1] * pre_ts
            )
            pre_root_rot_list.extend(pre_trans_root_rot)

            # Append sequence: transition from last frame -> hold default
            post_ts = (np.linspace(0.0, 1.0, n + 1)[1:])[:, None]
            post_trans_dof = (
                last_dof[None, :] * (1.0 - post_ts)
                + default_pos[None, :] * post_ts
            )
            post_dof_list.extend(post_trans_dof)
            
            # Interpolate root position and rotation
            post_trans_root_pos = (
                root_pos[-1:] * (1.0 - post_ts) + default_root_pos_end[None, :] * post_ts
            )
            post_root_pos_list.extend(post_trans_root_pos)
            
            post_trans_root_rot = (
                root_rot[-1:] * (1.0 - post_ts) + default_root_rot[None, :] * post_ts
            )
            post_root_rot_list.extend(post_trans_root_rot)
            
            # Hold default pose for specified frames
            if hold_frames > 0:
                for _ in range(hold_frames):
                    post_dof_list.append(default_pos.copy())
                    post_root_pos_list.append(default_root_pos_end.copy())
                    post_root_rot_list.append(default_root_rot.copy())

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
