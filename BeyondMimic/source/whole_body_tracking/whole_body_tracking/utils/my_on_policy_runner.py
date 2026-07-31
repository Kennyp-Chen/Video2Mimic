import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def get_normalizer(self,):
        """
        解决rsl-rl-lib3.1.2不存在            
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
        
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            normalizer = self.get_normalizer()
            # export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            export_policy_as_onnx(self.alg.policy, normalizer=normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

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
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            normalizer = self.get_normalizer()
            # export_motion_policy_as_onnx(
            #     self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            # )
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
