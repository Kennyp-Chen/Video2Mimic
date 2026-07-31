from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取机器人锚定身体在世界坐标系中的方向（旋转矩阵的前两列）
    
    Args:
        env: 环境实例
        command_name: 运动命令名称
        
    Returns:
        锚定身体在世界坐标系中的方向矩阵前两列（6维向量）
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取机器人锚定身体在世界坐标系中的线速度
    
    Args:
        env: 环境实例
        command_name: 运动命令名称
        
    Returns:
        锚定身体在世界坐标系中的线速度
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取机器人锚定身体在世界坐标系中的角速度
    
    Args:
        env: 环境实例
        command_name: 运动命令名称
        
    Returns:
        锚定身体在世界坐标系中的角速度
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取所有身体部位在锚定身体坐标系中的相对位置
    
    Args:
        env: 环境实例
        command_name: 运动命令名称
        
    Returns:
        所有身体部位相对于锚定身体的位置（在锚定身体坐标系中）
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取所有身体部位在锚定身体坐标系中的相对方向
    
    Args:
        env: 环境实例
        command_name: 运动命令名称
        
    Returns:
        所有身体部位相对于锚定身体的方向矩阵前两列
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取参考运动锚点在机器人锚定身体坐标系中的相对位置
    
    Args:
        env: 环境实例
        command_name: 运动命令名称
        
    Returns:
        参考运动锚点相对于机器人锚定身体的位置
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """获取参考运动锚点在机器人锚定身体坐标系中的相对方向
    
    Args:
        env: 环境实例
        command_name: 运动命令名称
        
    Returns:
        参考运动锚点相对于机器人锚定身体的方向矩阵前两列（6维向量）
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)