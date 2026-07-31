# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL using TensorBoard."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL using TensorBoard.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to the motion .npz file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml#,dump_pickle
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import pickle
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class TensorBoardMotionOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with TensorBoard support for exporting ONNX models."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.log_dir = log_dir

    def get_normalizer(self):
        """
        解决rsl-rl-lib3.1.2不存在:
        self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
        """
        # 安全地获取归一化器
        normalizer = None
        # 先检查新版结构
        if hasattr(self, 'alg') and hasattr(self.alg, 'policy'):
            # 检查策略中是否有归一化器
            policy_obj = self.alg.policy
            if hasattr(policy_obj, 'actor_obs_normalizer'):
                normalizer = policy_obj.actor_obs_normalizer
            elif hasattr(policy_obj, 'obs_normalizer'):
                normalizer = policy_obj.obs_normalizer
        # 再检查旧版结构
        elif hasattr(self, 'obs_normalizer'):
            normalizer = self.obs_normalizer
        # 最后检查算法中是否有归一化器
        elif hasattr(self, 'alg') and hasattr(self.alg, 'actor_critic'):
            actor_critic = self.alg.actor_critic
            if hasattr(actor_critic, 'obs_normalizer'):
                normalizer = actor_critic.obs_normalizer
        return normalizer

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)

        # Export ONNX model for TensorBoard logger
        # if self.logger_type in ["tensorboard"]:
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        normalizer = self.get_normalizer()
        export_motion_policy_as_onnx(
            self.env.unwrapped, self.alg.policy, normalizer=normalizer, path=policy_path, filename=filename
        )
        # Use log_dir or run_name as run_path for metadata
        run_path = self.log_dir if self.log_dir else "tensorboard_run"
        attach_onnx_metadata(self.env.unwrapped, run_path, path=policy_path, filename=filename)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent using TensorBoard."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set logger to tensorboard if not specified
    if args_cli.logger is None:
        agent_cfg.logger = "tensorboard"
    elif args_cli.logger != "tensorboard":
        print(f"[WARNING] This script is designed for TensorBoard logging. You specified: {args_cli.logger}")

    # load the motion file from the specified path
    motion_file = pathlib.Path(args_cli.motion_file)
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    env_cfg.commands.motion.motion_file = str(motion_file.absolute())
    print(f"[INFO] Using motion file: {env_cfg.commands.motion.motion_file}")

    # extract motion file name (without extension) for logging
    motion_name = motion_file.stem  # e.g., "dance2_subject4" from "dance2_subject4.npz"

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{motion_name}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir += f"_{motion_name}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl with TensorBoard support
    runner = TensorBoardMotionOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # Lab 2.1.0
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    # Lab2.3.x
    # Save pickle files for exact object serialization
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

