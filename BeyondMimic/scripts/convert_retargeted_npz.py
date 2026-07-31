"""Convert retargeted G1 NPZ data to project format.

This script converts NPZ files with retargeted motion data (root_pos, root_rot, dof_pos)
to the project's format which includes full body state information (joint_pos, joint_vel,
body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w).

Input format:
    - fps: scalar (e.g., 50)
    - root_pos: (N, 3) - root position
    - root_rot: (N, 4) - root rotation quaternion (xyzw or wxyz)
    - dof_pos: (N, 29) - joint positions

Output format (project format):
    - fps: [50]
    - joint_pos: (N, 29) - joint positions
    - joint_vel: (N, 29) - joint velocities (computed)
    - body_pos_w: (N, num_bodies, 3) - body positions in world frame
    - body_quat_w: (N, num_bodies, 4) - body quaternions in world frame
    - body_lin_vel_w: (N, num_bodies, 3) - body linear velocities
    - body_ang_vel_w: (N, num_bodies, 3) - body angular velocities

Usage:
    # Single file
    python convert_retargeted_npz.py --input shoot_m.npz --output g1/Basketball-NPZ/shoot_m.npz --headless

    # Single file with CSV output
    python convert_retargeted_npz.py --input shoot_m.npz --output g1/Basketball-NPZ/shoot_m.npz --output_csv --headless

    # Batch conversion
    python convert_retargeted_npz.py --input_dir retargeted_data --output_dir g1/Basketball-NPZ --headless

    # Batch conversion with CSV output
    python convert_retargeted_npz.py --input_dir retargeted_data --output_dir g1/Basketball-NPZ --output_csv --headless
"""

import argparse
import sys
from pathlib import Path

# Parse arguments before launching Isaac Sim
parser = argparse.ArgumentParser(description="Convert retargeted NPZ to project format")
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Input NPZ file to convert",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output NPZ file path",
)
parser.add_argument(
    "--input_dir",
    type=str,
    default=None,
    help="Input directory containing NPZ files (for batch processing)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory for converted NPZ files (for batch processing)",
)
parser.add_argument(
    "--quat_format",
    type=str,
    default="xyzw",
    choices=["xyzw", "wxyz"],
    help="Quaternion format in input file (default: xyzw)",
)
parser.add_argument(
    "--output_fps",
    type=int,
    default=None,
    help="Output FPS (default: use input FPS from file)",
)
parser.add_argument(
    "--output_csv",
    action="store_true",
    help="Also output CSV file in addition to NPZ",
)

# Add Isaac Lab launcher arguments
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate arguments
if args_cli.input is None and args_cli.input_dir is None:
    print("Error: Either --input or --input_dir must be specified")
    sys.exit(1)

if args_cli.input is not None and args_cli.output is None:
    print("Error: --output must be specified when using --input")
    sys.exit(1)

if args_cli.input_dir is not None and args_cli.output_dir is None:
    print("Error: --output_dir must be specified when using --input_dir")
    sys.exit(1)

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is launched."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ConversionSceneCfg(InteractiveSceneCfg):
    """Configuration for conversion scene."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def compute_velocities(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocities from positions using finite differences."""
    velocities = np.zeros_like(positions)
    # Forward difference for first frame
    velocities[0] = (positions[1] - positions[0]) / dt
    # Central difference for middle frames
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    # Backward difference for last frame
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    return velocities


def compute_angular_velocities(quaternions: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocities from quaternion sequence."""
    # Convert to torch for quaternion operations
    quats = torch.from_numpy(quaternions).float()
    
    # Compute relative rotations
    q_prev = quats[:-2]
    q_next = quats[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    
    # Convert to axis-angle and divide by time
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    
    # Pad with first and last values
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    
    return omega.numpy()


def convert_npz_file(
    sim: SimulationContext,
    scene: InteractiveScene,
    input_file: str,
    output_file: str,
    quat_format: str,
    joint_names: list[str],
    output_fps: int = None,
    output_csv: bool = False,
):
    """Convert a single retargeted NPZ file to project format."""
    # Load input data
    print(f"Loading: {input_file}")
    data = np.load(input_file)
    
    # Detect input FPS
    input_fps = int(data['fps']) if 'fps' in data else 50
    input_dt = 1.0 / input_fps
    
    # Use output FPS if specified, otherwise use input FPS
    fps = output_fps if output_fps is not None else input_fps
    dt = 1.0 / fps
    
    root_pos = data['root_pos']  # (N, 3)
    root_rot = data['root_rot']  # (N, 4)
    dof_pos = data['dof_pos']    # (N, 29)
    
    num_frames = root_pos.shape[0]
    print(f"  Input: {num_frames} frames @ {input_fps} FPS")
    print(f"  Output: {fps} FPS")
    
    # Convert quaternion format if needed (xyzw -> wxyz)
    if quat_format == "xyzw":
        root_rot_wxyz = np.concatenate([root_rot[:, 3:4], root_rot[:, :3]], axis=1)
    else:
        root_rot_wxyz = root_rot
    
    # Interpolate if output FPS differs from input FPS
    if fps != input_fps:
        print(f"  Interpolating from {input_fps} FPS to {fps} FPS...")
        duration = (num_frames - 1) * input_dt
        
        # Create new time points
        new_times = np.arange(0, duration, dt)
        old_times = np.arange(0, duration + input_dt/2, input_dt)[:num_frames]
        
        # Interpolate positions
        root_pos_interp = np.zeros((len(new_times), 3))
        dof_pos_interp = np.zeros((len(new_times), dof_pos.shape[1]))
        
        for i in range(3):
            root_pos_interp[:, i] = np.interp(new_times, old_times, root_pos[:, i])
        
        for i in range(dof_pos.shape[1]):
            dof_pos_interp[:, i] = np.interp(new_times, old_times, dof_pos[:, i])
        
        # SLERP for quaternions
        rotations = Rotation.from_quat(root_rot_wxyz[:, [1, 2, 3, 0]])  # wxyz -> xyzw for scipy
        slerp = Slerp(old_times, rotations)
        root_rot_interp = slerp(new_times).as_quat()  # returns xyzw
        root_rot_wxyz = np.concatenate([root_rot_interp[:, 3:4], root_rot_interp[:, :3]], axis=1)  # xyzw -> wxyz
        
        # Update data
        root_pos = root_pos_interp
        dof_pos = dof_pos_interp
        num_frames = len(new_times)
        print(f"  Interpolated to {num_frames} frames")
    
    # Compute velocities
    print("  Computing velocities...")
    root_lin_vel = compute_velocities(root_pos, dt)
    root_ang_vel = compute_angular_velocities(root_rot_wxyz, dt)
    dof_vel = compute_velocities(dof_pos, dt)
    
    # Convert to torch tensors
    root_pos_t = torch.from_numpy(root_pos).float().to(sim.device)
    root_rot_t = torch.from_numpy(root_rot_wxyz).float().to(sim.device)
    root_lin_vel_t = torch.from_numpy(root_lin_vel).float().to(sim.device)
    root_ang_vel_t = torch.from_numpy(root_ang_vel).float().to(sim.device)
    dof_pos_t = torch.from_numpy(dof_pos).float().to(sim.device)
    dof_vel_t = torch.from_numpy(dof_vel).float().to(sim.device)
    
    # Get robot and joint indices
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    
    # Data logger
    log = {
        "fps": [fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    
    # Simulate and record
    print("  Simulating and recording...")
    for i in range(num_frames):
        # Set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos_t[i:i+1]
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = root_rot_t[i:i+1]
        root_states[:, 7:10] = root_lin_vel_t[i:i+1]
        root_states[:, 10:] = root_ang_vel_t[i:i+1]
        robot.write_root_state_to_sim(root_states)
        
        # Set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos_t[i:i+1]
        joint_vel[:, robot_joint_indexes] = dof_vel_t[i:i+1]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        sim.render()
        scene.update(sim.get_physics_dt())
        
        # Record data
        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())
    
    # Stack arrays
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)
    
    # Save to NPZ
    print(f"  Saving to: {output_file}")
    np.savez(output_file, **log)
    
    # Save to CSV if requested
    if output_csv:
        csv_file = str(Path(output_file).with_suffix('.csv'))
        print(f"  Saving CSV to: {csv_file}")
        
        # Extract base pose and joint positions
        # Format: [base_pos (3), base_quat_xyzw (4), joint_pos (29)]
        base_pos = log["body_pos_w"][:, 0, :]  # (N, 3) - pelvis position
        base_quat = log["body_quat_w"][:, 0, :]  # (N, 4) - pelvis quaternion (wxyz)
        joint_pos = log["joint_pos"]  # (N, 29)
        
        # Convert quaternion from wxyz to xyzw for CSV
        base_quat_xyzw = np.concatenate([base_quat[:, 1:4], base_quat[:, 0:1]], axis=1)
        
        # Combine into CSV format: [base_pos, base_quat_xyzw, joint_pos]
        csv_data = np.concatenate([base_pos, base_quat_xyzw, joint_pos], axis=1)
        
        # Save CSV
        np.savetxt(csv_file, csv_data, delimiter=",", fmt="%.6f")
        print(f"  ✓ CSV saved ({csv_data.shape[0]} frames, {csv_data.shape[1]} columns)")
    
    return log


def main():
    """Main conversion function."""
    # Determine simulation dt based on output FPS
    output_fps = args_cli.output_fps if args_cli.output_fps is not None else 50
    sim_dt = 1.0 / output_fps
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = sim_dt
    sim = SimulationContext(sim_cfg)
    
    # Design scene
    scene_cfg = ConversionSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    
    # Joint names for G1 robot
    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    # Process files
    if args_cli.input is not None:
        # Single file conversion
        input_path = Path(args_cli.input)
        output_path = Path(args_cli.output)
        
        if not input_path.exists():
            print(f"Error: Input file does not exist: {input_path}")
            sys.exit(1)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("NPZ Format Converter")
        print("=" * 80)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Quaternion format: {args_cli.quat_format}")
        if args_cli.output_fps is not None:
            print(f"Output FPS: {args_cli.output_fps}")
        else:
            print("Output FPS: Auto (use input FPS)")
        print("=" * 80)
        
        try:
            convert_npz_file(
                sim=sim,
                scene=scene,
                input_file=str(input_path),
                output_file=str(output_path),
                quat_format=args_cli.quat_format,
                joint_names=joint_names,
                output_fps=args_cli.output_fps,
                output_csv=args_cli.output_csv,
            )
            print("\n✅ Conversion completed successfully!")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Batch conversion
        input_dir = Path(args_cli.input_dir)
        output_dir = Path(args_cli.output_dir)
        
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all NPZ files
        npz_files = sorted([f for f in input_dir.glob("*.npz")])
        
        if not npz_files:
            print(f"No NPZ files found in {input_dir}")
            sys.exit(0)
        
        print("=" * 80)
        print("Batch NPZ Format Converter")
        print("=" * 80)
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Quaternion format: {args_cli.quat_format}")
        if args_cli.output_fps is not None:
            print(f"Output FPS: {args_cli.output_fps}")
        else:
            print("Output FPS: Auto (use input FPS)")
        print(f"Files to process: {len(npz_files)}")
        print("=" * 80)
        
        success_count = 0
        error_count = 0
        
        for idx, npz_file in enumerate(npz_files, 1):
            output_file = output_dir / npz_file.name
            
            print(f"\n[{idx}/{len(npz_files)}] Processing: {npz_file.name}")
            
            try:
                convert_npz_file(
                    sim=sim,
                    scene=scene,
                    input_file=str(npz_file),
                    output_file=str(output_file),
                    quat_format=args_cli.quat_format,
                    joint_names=joint_names,
                    output_fps=args_cli.output_fps,
                    output_csv=args_cli.output_csv,
                )
                print(f"  ✓ Success")
                success_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
        
        # Print summary
        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  Success: {success_count}")
        print(f"  Errors:  {error_count}")
        print(f"  Total:   {len(npz_files)}")
        print("=" * 80)
        
        if error_count > 0:
            print("\n⚠️  Some files failed to process.")
        else:
            print("\n✅ All files converted successfully!")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
