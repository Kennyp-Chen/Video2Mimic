from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """获取指定身体部位的索引
    
    Args:
        command: 运动命令对象
        body_names: 身体部位名称列表，如果为None则返回所有身体部位的索引
        
    Returns:
        身体部位索引列表
    """
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """全局锚点位置误差奖励函数
    
    数学公式: exp(-‖p_anchor_ref - p_anchor_robot‖² / std²)
    
    作用: 鼓励机器人的锚点身体位置与参考运动的锚点位置尽可能接近
         当位置误差越小，奖励值越接近1；误差越大，奖励值越接近0
         
    Args:
        env: 强化学习环境
        command_name: 运动命令名称
        std: 标准差参数，控制奖励函数的宽度（灵敏度）
        
    Returns:
        基于锚点位置误差的指数奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 计算锚点位置的平方误差
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    # 应用指数衰减：误差越小，奖励越接近1
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """全局锚点方向误差奖励函数
    
    数学公式: exp(-(quat_error(q_anchor_ref, q_anchor_robot))² / std²)
    
    作用: 鼓励机器人的锚点身体方向与参考运动的锚点方向尽可能一致
         使用四元数误差度量旋转差异，误差越小奖励越高
         
    Args:
        env: 强化学习环境
        command_name: 运动命令名称
        std: 标准差参数，控制奖励函数的宽度
        
    Returns:
        基于锚点方向误差的指数奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 计算四元数旋转误差的平方
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    # 应用指数衰减
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """相对身体位置误差奖励函数
    
    数学公式: exp(-mean(‖p_body_relative_ref - p_body_robot‖²) / std²)
    
    作用: 鼓励所有指定身体部位相对于锚点的位置与参考运动一致
         计算多个身体部位位置误差的平均值，提供整体的位置跟踪奖励
         
    Args:
        env: 强化学习环境
        command_name: 运动命令名称
        std: 标准差参数
        body_names: 指定的身体部位名称列表，None表示使用所有身体部位
        
    Returns:
        基于相对身体位置误差的指数奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 获取指定身体部位的索引
    body_indexes = _get_body_indexes(command, body_names)
    # 计算相对位置平方误差并求平均
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """相对身体方向误差奖励函数
    
    数学公式: exp(-mean(quat_error(q_body_relative_ref, q_body_robot)²) / std²)
    
    作用: 鼓励所有指定身体部位相对于锚点的方向与参考运动一致
         提供整体的姿态跟踪奖励，确保身体各部位的相对姿态正确
         
    Args:
        env: 强化学习环境
        command_name: 运动命令名称
        std: 标准差参数
        body_names: 指定的身体部位名称列表
        
    Returns:
        基于相对身体方向误差的指数奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # 计算四元数旋转误差的平方并求平均
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """全局身体线速度误差奖励函数
    
    数学公式: exp(-mean(‖v_lin_body_ref - v_lin_body_robot‖²) / std²)
    
    作用: 鼓励身体部位的线速度与参考运动的速度匹配
         确保运动的速度特性（如移动快慢）与示范一致
         
    Args:
        env: 强化学习环境
        command_name: 运动命令名称
        std: 标准差参数
        body_names: 指定的身体部位名称列表
        
    Returns:
        基于身体线速度误差的指数奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # 计算线速度平方误差并求平均
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """全局身体角速度误差奖励函数
    
    数学公式: exp(-mean(‖ω_body_ref - ω_body_robot‖²) / std²)
    
    作用: 鼓励身体部位的角速度与参考运动的速度匹配
         确保旋转运动的速度特性与示范一致
         
    Args:
        env: 强化学习环境
        command_name: 运动命令名称
        std: 标准差参数
        body_names: 指定的身体部位名称列表
        
    Returns:
        基于身体角速度误差的指数奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # 计算角速度平方误差并求平均
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """足部接触时间奖励函数
    
    数学公式: Σ[(last_contact_time < threshold) * first_air]
    
    作用: 鼓励足部在合适的时间与地面接触
         - 对接触时间短于阈值的足部给予奖励
         - 只对刚离开地面的足部进行奖励（避免重复奖励）
         - 用于实现步态时序的匹配
         
    Args:
        env: 强化学习环境
        sensor_cfg: 接触传感器配置
        threshold: 接触时间阈值，短于此阈值的接触会获得奖励
        
    Returns:
        基于足部接触时序的奖励
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取刚离开地面的足部标志
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    # 获取上次接触时间
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    # 对接触时间短于阈值且刚离开地面的足部给予奖励
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward