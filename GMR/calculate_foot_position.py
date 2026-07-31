#!/usr/bin/env python3
"""
根据G1机器人29DOF的MuJoCo XML文件，计算足部位置
"""

import numpy as np
import xml.etree.ElementTree as ET
try:
    import mujoco as mj
except ImportError:
    print("MuJoCo not installed. This script requires mujoco.")
    exit(1)

def load_g1_model():
    """加载G1机器人模型"""
    xml_path = '/home/hero/Projects/Robotics/RL/Video2Track/GMR/assets/unitree_g1/g1_mocap_29dof.xml'
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    return model, data

def get_foot_positions(model, data, root_pos, root_rot, dof_pos):
    """
    计算足部位置
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        root_pos: 根部位置 (x, y, z)
        root_rot: 根部旋转四元数 (w, x, y, z)
        dof_pos: 关节角度数组
    
    Returns:
        dict: 足部位置信息
    """
    # 设置机器人姿态
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot
    data.qpos[7:] = dof_pos
    
    # 前向运动学计算
    mj.mj_forward(model, data)
    
    # 足部相关的body名称
    foot_bodies = {
        'left_toe': 'left_toe_link',
        'left_ankle_pitch': 'left_ankle_pitch_link', 
        'left_ankle_roll': 'left_ankle_roll_link',
        'right_toe': 'right_toe_link',
        'right_ankle_pitch': 'right_ankle_pitch_link',
        'right_ankle_roll': 'right_ankle_roll_link'
    }
    
    # 获取足部位置
    foot_positions = {}
    for foot_name, body_name in foot_bodies.items():
        try:
            body_id = model.body(body_name).id
            position = data.xpos[body_id]  # 世界坐标系中的位置
            foot_positions[foot_name] = position
        except:
            print(f"Warning: Could not find body '{body_name}'")
            foot_positions[foot_name] = None
    
    return foot_positions

def analyze_gangster_dance_foot_positions(last_frames=35):
    """分析gangster_dance动作最后35帧的足部位置"""
    # 加载机器人模型
    model, data = load_g1_model()
    
    # 加载动作数据
    motion_data = np.loadtxt('/home/hero/Projects/Robotics/RL/Video2Track/GMR/motions/G1/csv/gangster_dance.csv', delimiter=',')
    
    # 提取最后35帧
    frames = motion_data[-last_frames:]
    
    print(f'G1 Gangster Dance - 最后{last_frames}帧足部位置分析')
    print('=' * 80)
    print()
    
    print('足部body映射:')
    foot_bodies = ['left_toe_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
                   'right_toe_link', 'right_ankle_pitch_link', 'right_ankle_roll_link']
    for body in foot_bodies:
        body_id = model.body(body).id
        print(f'  {body}: ID {body_id}')
    print()
    
    # 分析每一帧
    for i in range(len(frames)):
        frame_idx = 1415 - len(frames) + i  # 实际帧索引
        frame_data = frames[i]
        
        # 提取位置和姿态数据
        root_pos = frame_data[0:3]
        root_rot = frame_data[3:7]
        dof_pos = frame_data[7:]
        
        # 计算足部位置
        foot_positions = get_foot_positions(model, data, root_pos, root_rot, dof_pos)
        
        if i in [0, 17, 30, 31, 32, 33, 34]:  # 显示关键帧
            print(f'帧 {frame_idx}:')
            print(f'  根部位置: ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})')
            
            for foot_name, position in foot_positions.items():
                if position is not None:
                    print(f'  {foot_name:15s}: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})')
            print()
    
    # 分析足部高度变化
    print('足部高度变化分析:')
    print('-' * 50)
    
    left_toe_heights = []
    right_toe_heights = []
    left_ankle_heights = []
    right_ankle_heights = []
    
    for i in range(len(frames)):
        frame_data = frames[i]
        root_pos = frame_data[0:3]
        root_rot = frame_data[3:7]
        dof_pos = frame_data[7:]
        
        foot_positions = get_foot_positions(model, data, root_pos, root_rot, dof_pos)
        
        left_toe_heights.append(foot_positions['left_toe'][2] if foot_positions['left_toe'] is not None else np.nan)
        right_toe_heights.append(foot_positions['right_toe'][2] if foot_positions['right_toe'] is not None else np.nan)
        left_ankle_heights.append(foot_positions['left_ankle_pitch'][2] if foot_positions['left_ankle_pitch'] is not None else np.nan)
        right_ankle_heights.append(foot_positions['right_ankle_pitch'][2] if foot_positions['right_ankle_pitch'] is not None else np.nan)
    
    print(f'左脚趾高度范围: [{np.nanmin(left_toe_heights):.3f}, {np.nanmax(left_toe_heights):.3f}] 米')
    print(f'右脚趾高度范围: [{np.nanmin(right_toe_heights):.3f}, {np.nanmax(right_toe_heights):.3f}] 米')
    print(f'左踝高度范围:   [{np.nanmin(left_ankle_heights):.3f}, {np.nanmax(left_ankle_heights):.3f}] 米')
    print(f'右踝高度范围:   [{np.nanmin(right_ankle_heights):.3f}, {np.nanmax(right_ankle_heights):.3f}] 米')
    print()
    
    # 检查最后几帧是否着地
    print(f'着地状态检查 (最后{last_frames}帧):')
    print('-' * 40)
    
    ground_height = 0.0  # 假设地面高度为0
    
    for i in range(0, len(frames)):
        frame_idx = 1415 - len(frames) + i
        frame_data = frames[i]
        root_pos = frame_data[0:3]
        root_rot = frame_data[3:7]
        dof_pos = frame_data[7:]
        
        foot_positions = get_foot_positions(model, data, root_pos, root_rot, dof_pos)
        
        left_toe_z = foot_positions['left_toe'][2] if foot_positions['left_toe'] is not None else np.nan
        right_toe_z = foot_positions['right_toe'][2] if foot_positions['right_toe'] is not None else np.nan
        
        left_on_ground = abs(left_toe_z - ground_height) < 0.05  # 5cm容差
        right_on_ground = abs(right_toe_z - ground_height) < 0.05
        
        left_too_high = abs(left_toe_z - ground_height) > 0.4
        right_too_high = abs(right_toe_z - ground_height) > 0.4

        print(f'倒数第{last_frames-i}帧/帧 {frame_idx}: 左脚趾高度: {left_toe_z:.3f}m, 右脚趾高度: {right_toe_z:.3f}m')
        if left_on_ground and right_on_ground:
            print(f'           ooo左右脚同时着地')
        if left_too_high and right_too_high:
            print(f'           !!!左脚、右脚趾过高')
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--last_frames", type=int, default=35)
    args = parser.parse_args()
    analyze_gangster_dance_foot_positions(args.last_frames)
    