"""Batch convert CSV files to NPZ format for Isaac Lab.
Output FPS: 越高，总时常越大，动作执行越慢（与视频相反）
This script processes all g1 29dof motion CSV files in Input directory and converts them to NPZ format.
It uses the same interpolation and simulation methods as csv_to_npz.py but implements batch processing with local file saving.

Default behavior:
    - Input: g1/LAFAN/*.csv (30fps)
    - Output: g1/29dof/LAFAN-30fp/*.npz (30fps) for 29DOF
    - Output: g1/23dof/LAFAN-30fp/*.npz (30fps) for 23DOF
    - WandB: Disabled (use --upload_wandb to enable)

Usage:
    # Basic usage (default settings)
    python batch_csv_to_npz.py --headless

    # Enable wandb upload
    python batch_csv_to_npz.py --upload_wandb --headless

    # Custom directories
    python batch_csv_to_npz.py --input_dir g1/LAFAN --output_dir g1/LAFAN-60fps-NPZ --output_fps 60 --headless
    python batch_csv_to_npz.py --input_dir g1/LAFAN --output_dir g1/29dof/LAFAN-30fps-NPZ --output_fps 30 --headless
    python batch_csv_to_npz.py --dof 23 --input_dir g1/LAFAN --output_dir g1/23dof/LAFAN-30fps-NPZ --output_fps 30 --headless
    
    python scripts/batch_csv_to_npz.py --dof 23 --input_dir g1/LAFAN --output_dir g1/23dof/LAFAN-50fps-NPZ --output_fps 50 --headless
    python scripts/batch_csv_to_npz.py --dof 23 --input_dir g1/LAFAN --output_dir g1/23dof/LAFAN-80fps-NPZ --output_fps 80 --headless
    python scripts/batch_csv_to_npz.py --input_dir g1/LAFAN --output_dir g1/29dof/LAFAN-80fps-NPZ --output_fps 80 --headless

"""

import argparse
import os
import sys
from pathlib import Path

# Parse arguments before launching Isaac Sim
parser = argparse.ArgumentParser(description="Batch convert LAFAN CSV files to NPZ format")
parser.add_argument(
    "--input_dir",
    type=str,
    default="g1/LAFAN",
    help="Input directory containing CSV files (default: g1/LAFAN)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,  # Will be set based on DOF
    help="Output directory for NPZ files (default: g1/29dof/LAFAN-30fp or g1/23dof/LAFAN-30fp)",
)
parser.add_argument(
    "--input_fps",
    type=int,
    default=30,
    help="FPS of input CSV files (default: 30)",
)
parser.add_argument(
    "--output_fps",
    type=int,
    default=30,
    help="FPS of output NPZ files (default: 30)",
)
parser.add_argument(
    "--upload_wandb",
    action="store_true",
    help="Upload to wandb registry (default: False)",
)
parser.add_argument(
    "--skip_existing",
    action="store_true",
    help="Skip files if output NPZ already exists",
)
parser.add_argument(
    "--max_files",
    type=int,
    default=None,
    help="Maximum number of files to process (for testing)",
)
parser.add_argument(
    "--dof",
    type=int,
    choices=[23, 29],
    default=29,
    help="Robot DOF configuration: 23 or 29 (default: 29)",
)

# Add Isaac Lab launcher arguments
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set default output directory based on DOF if not specified
if args_cli.output_dir is None:
    if args_cli.dof == 23:
        args_cli.output_dir = "g1/23dof/LAFAN-30fp"
    else:
        args_cli.output_dir = "g1/29dof/LAFAN-30fp"

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim is launched."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.g1_23dof import G1_23DOF_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    """Load and interpolate motion from CSV file."""
    
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None = None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def process_motion_file(
    sim: SimulationContext,
    scene: InteractiveScene,
    input_file: str,
    output_file: str,
    input_fps: int,
    output_fps: int,
    joint_names: list[str],
    dof: int,
):
    """Process a single motion file and save to NPZ."""
    # Load motion
    motion = MotionLoader(
        motion_file=input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
        frame_range=None,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # Create mapping from 29DOF to 23DOF if needed
    if dof == 23:
        # Map 29DOF joint indices to 23DOF joint indices
        # CSV file has 29 joints, we need to select the 23 joints that exist in 23DOF robot
        full_29dof_joint_names = [
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
        
        # Find indices of 23DOF joints in the 29DOF CSV data
        csv_joint_indices = []
        for joint_name in joint_names:
            if joint_name in full_29dof_joint_names:
                csv_joint_indices.append(full_29dof_joint_names.index(joint_name))
            else:
                print(f"Warning: Joint {joint_name} not found in CSV data")
                csv_joint_indices.append(0)  # Default to first joint if not found
    else:
        csv_joint_indices = list(range(len(joint_names)))  # Use all joints for 29DOF

    # Data logger
    log = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    # Simulate and record
    for _ in range(motion.output_frames):
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # Set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # Set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        
        # Map motion data to robot joints
        if dof == 23:
            # Select only the joints that exist in 23DOF robot
            motion_dof_pos_23 = motion_dof_pos[:, csv_joint_indices]
            motion_dof_vel_23 = motion_dof_vel[:, csv_joint_indices]
            joint_pos[:, robot_joint_indexes] = motion_dof_pos_23
            joint_vel[:, robot_joint_indexes] = motion_dof_vel_23
        else:
            # Use all joints for 29DOF
            joint_pos[:, robot_joint_indexes] = motion_dof_pos
            joint_vel[:, robot_joint_indexes] = motion_dof_vel
            
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

        if reset_flag:
            break

    # Stack arrays
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    # Save to NPZ
    np.savez(output_file, **log)
    
    return log


def main():
    """Main batch processing function."""
    # Resolve paths
    input_dir = Path(args_cli.input_dir).resolve()
    output_dir = Path(args_cli.output_dir).resolve()

    # Check input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted([f for f in input_dir.glob("*.csv")])
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(0)

    # Limit files if specified
    if args_cli.max_files is not None:
        csv_files = csv_files[:args_cli.max_files]

    # Print configuration
    print("=" * 80)
    print("Batch LAFAN to NPZ Converter")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Robot DOF:       {args_cli.dof}")
    print(f"Input FPS:        {args_cli.input_fps}")
    print(f"Output FPS:       {args_cli.output_fps}")
    print(f"Upload to wandb:  {'Yes' if args_cli.upload_wandb else 'No'}")
    print(f"Files to process: {len(csv_files)}")
    print("=" * 80)

    # Initialize simulation
    # sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(device="cpu")

    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    
    # Design scene
    if args_cli.dof == 23:
        robot_cfg = G1_23DOF_CYLINDER_CFG
    else:
        robot_cfg = G1_CYLINDER_CFG
    
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()

    # Joint names based on DOF configuration
    if args_cli.dof == 23:
        # 23DOF configuration - based on URDF file
        joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint",
        ]
    else:
        # 29DOF configuration - full joint set
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
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # Initialize wandb if needed
    wandb_run = None
    if args_cli.upload_wandb:
        import wandb
        wandb_run = wandb.init(project="batch_csv_to_npz", name=f"batch_{len(csv_files)}_files")

    for idx, csv_file in enumerate(csv_files, 1):
        motion_name = csv_file.stem
        output_file = output_dir / f"{motion_name}.npz"

        # Skip if exists
        if args_cli.skip_existing and output_file.exists():
            print(f"[{idx}/{len(csv_files)}] [SKIP] {csv_file.name} (already exists)")
            skip_count += 1
            continue

        print(f"\n[{idx}/{len(csv_files)}] Processing: {csv_file.name}")
        print(f"  Motion name: {motion_name}")
        print(f"  Output: {output_file.name}")

        try:
            # Process the file
            log = process_motion_file(
                sim=sim,
                scene=scene,
                input_file=str(csv_file),
                output_file=str(output_file),
                input_fps=args_cli.input_fps,
                output_fps=args_cli.output_fps,
                joint_names=joint_names,
                dof=args_cli.dof,
            )

            print(f"  ✓ Success - Saved to {output_file}")
            success_count += 1

            # Upload to wandb if enabled
            if args_cli.upload_wandb and wandb_run is not None:
                artifact = wandb.Artifact(motion_name, type="motions")
                artifact.add_file(str(output_file))
                wandb_run.log_artifact(artifact)
                print(f"  ✓ Uploaded to wandb")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1

    # Finish wandb
    if wandb_run is not None:
        wandb_run.finish()

    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Success:  {success_count}")
    print(f"  Skipped:  {skip_count}")
    print(f"  Errors:   {error_count}")
    print(f"  Total:    {len(csv_files)}")
    print("=" * 80)

    if error_count > 0:
        print("\n⚠️  Some files failed to process.")
    else:
        print("\n✅ All files processed successfully!")


if __name__ == "__main__":
    # try:
    main()
    # finally:
    #     simulation_app.close()

