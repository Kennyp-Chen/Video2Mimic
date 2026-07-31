"""调整NPZ文件的整体高度

这个脚本可以调整NPZ文件的整体高度，支持两种格式：
1. 重定向格式（root_pos, root_rot, dof_pos）
2. 转换后格式（body_pos_w, joint_pos等）

使用方法:
    # 手动指定偏移量
    python scripts/adjust_root_height.py g1/Basketball-NPZ/shoot_m.npz g1/Basketball-NPZ/shoot_m_adjusted.npz -0.15 --output_csv

    # 自动调整（根据前几帧的站立姿态计算）
    python scripts/adjust_root_height.py g1/Basketball-NPZ/shoot_m.npz g1/Basketball-NPZ/shoot_m_adjusted.npz --auto --output_csv

    # 指定用于计算的初始帧数
    python scripts/adjust_root_height.py g1/Basketball-NPZ/shoot_m.npz g1/Basketball-NPZ/shoot_m_adjusted.npz --auto --init_frames 10 --output_csv
"""

import argparse
import numpy as np
from pathlib import Path


def detect_file_type(data):
    """检测NPZ文件类型"""
    if 'root_pos' in data and 'root_rot' in data and 'dof_pos' in data:
        return 'retargeted'
    elif 'body_pos_w' in data and 'joint_pos' in data:
        return 'converted'
    else:
        return 'unknown'


def adjust_height_retargeted(input_file, output_file, z_offset):
    """调整重定向格式文件的高度"""
    data = np.load(input_file)
    
    # 读取数据
    fps = data['fps']
    root_pos = data['root_pos'].copy()  # (N, 3)
    root_rot = data['root_rot'].copy()  # (N, 4)
    dof_pos = data['dof_pos'].copy()    # (N, 29)
    
    num_frames = root_pos.shape[0]
    
    # 显示原始统计
    print(f"\n原始数据:")
    print(f"  帧数: {num_frames}")
    print(f"  Root Z轴: 最小={root_pos[:, 2].min():.4f}, 最大={root_pos[:, 2].max():.4f}, 平均={root_pos[:, 2].mean():.4f}")
    
    # 调整Z轴
    root_pos[:, 2] += z_offset
    
    # 显示调整后统计
    print(f"\n调整后数据:")
    print(f"  Z轴偏移: {z_offset:+.4f} m")
    print(f"  Root Z轴: 最小={root_pos[:, 2].min():.4f}, 最大={root_pos[:, 2].max():.4f}, 平均={root_pos[:, 2].mean():.4f}")
    
    # 保存新文件
    print(f"\n保存到: {output_file}")
    np.savez(
        output_file,
        fps=fps,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos
    )
    
    print("✓ 完成！")
    return True


def adjust_height_converted(input_file, output_file, z_offset, output_csv=False):
    """调整转换后格式文件的高度"""
    data = np.load(input_file)
    
    # 读取数据
    fps = data['fps']
    joint_pos = data['joint_pos'].copy()
    joint_vel = data['joint_vel'].copy()
    body_pos_w = data['body_pos_w'].copy()  # (N, num_bodies, 3)
    body_quat_w = data['body_quat_w'].copy()
    body_lin_vel_w = data['body_lin_vel_w'].copy()
    body_ang_vel_w = data['body_ang_vel_w'].copy()
    
    num_frames = body_pos_w.shape[0]
    num_bodies = body_pos_w.shape[1]
    
    # 显示原始统计
    pelvis_z = body_pos_w[:, 0, 2]
    all_body_z = body_pos_w[:, :, 2]
    min_z_per_frame = all_body_z.min(axis=1)
    
    print(f"\n原始数据:")
    print(f"  帧数: {num_frames}, Body数: {num_bodies}")
    print(f"  Pelvis Z轴: 最小={pelvis_z.min():.4f}, 最大={pelvis_z.max():.4f}, 平均={pelvis_z.mean():.4f}")
    print(f"  全局最低点: 最小={min_z_per_frame.min():.4f}, 平均={min_z_per_frame.mean():.4f}")
    
    # 调整所有body的Z轴（统一偏移）
    body_pos_w[:, :, 2] += z_offset
    
    # 显示调整后统计
    pelvis_z_new = body_pos_w[:, 0, 2]
    all_body_z_new = body_pos_w[:, :, 2]
    min_z_per_frame_new = all_body_z_new.min(axis=1)
    
    print(f"\n调整后数据:")
    print(f"  Z轴偏移: {z_offset:+.4f} m")
    print(f"  Pelvis Z轴: 最小={pelvis_z_new.min():.4f}, 最大={pelvis_z_new.max():.4f}, 平均={pelvis_z_new.mean():.4f}")
    print(f"  全局最低点: 最小={min_z_per_frame_new.min():.4f}, 平均={min_z_per_frame_new.mean():.4f}")
    
    # 保存NPZ文件
    print(f"\n保存NPZ到: {output_file}")
    np.savez(
        output_file,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w
    )
    
    # 保存CSV文件（如果需要）
    if output_csv:
        csv_file = str(Path(output_file).with_suffix('.csv'))
        print(f"保存CSV到: {csv_file}")
        
        # CSV格式: base_pos (3) + base_quat_xyzw (4) + joint_pos (29)
        base_pos = body_pos_w[:, 0, :]  # (N, 3) - pelvis position
        base_quat_wxyz = body_quat_w[:, 0, :]  # (N, 4) - pelvis quaternion (wxyz)
        
        # 转换四元数从wxyz到xyzw（CSV格式）
        base_quat_xyzw = np.concatenate([base_quat_wxyz[:, 1:4], base_quat_wxyz[:, 0:1]], axis=1)
        
        # 组合: base_pos + base_quat_xyzw + joint_pos
        csv_data = np.concatenate([base_pos, base_quat_xyzw, joint_pos], axis=1)
        
        # 保存CSV
        np.savetxt(csv_file, csv_data, delimiter=",", fmt="%.6f")
        print(f"  ✓ CSV saved ({csv_data.shape[0]} frames, {csv_data.shape[1]} columns)")
    
    print("✓ 完成！")
    return True


def auto_adjust(input_file, output_file, init_frames=5, foot_thickness=0.03, output_csv=False):
    """自动计算并调整高度
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        init_frames: 用于计算offset的初始帧数（默认5帧）
        foot_thickness: 脚掌厚度（默认0.03m）
        output_csv: 是否同时输出CSV文件
    """
    print(f"自动分析文件: {input_file}")
    data = np.load(input_file)
    
    file_type = detect_file_type(data)
    
    if file_type == 'retargeted':
        print("文件类型: 重定向格式")
        root_pos = data['root_pos']
        
        # 只使用前几帧计算
        init_root_pos = root_pos[:init_frames]
        z_min = init_root_pos[:, 2].min()
        z_mean = init_root_pos[:, 2].mean()
        
        print(f"\n初始帧分析 (前{init_frames}帧):")
        print(f"  Root Z轴最低点: {z_min:.4f} m")
        print(f"  Root Z轴平均值: {z_mean:.4f} m")
        
        # 假设理想的骨盆高度（站立时）
        # 对于G1机器人，站立时骨盆高度约0.65-0.75m
        target_min = 0.65
        z_offset = target_min - z_min
        
    elif file_type == 'converted':
        print("文件类型: 转换后格式")
        body_pos_w = data['body_pos_w']
        
        # 只使用前几帧计算
        init_body_pos = body_pos_w[:init_frames]
        
        # 找到脚掌最低点
        init_min_z = init_body_pos[:, :, 2].min()
        init_mean_min_z = init_body_pos[:, :, 2].min(axis=1).mean()
        
        print(f"\n初始帧分析 (前{init_frames}帧):")
        print(f"  脚掌最低点: {init_min_z:.4f} m")
        print(f"  平均最低点: {init_mean_min_z:.4f} m")
        
        # 目标：脚掌最低点应该在foot_thickness高度
        target_foot_height = foot_thickness
        z_offset = target_foot_height - init_min_z
        
    else:
        print("错误: 无法识别的文件格式")
        return False
    
    print(f"\n建议调整:")
    print(f"  目标脚掌高度: {foot_thickness:.4f} m ({foot_thickness*100:.1f} cm)")
    print(f"  需要偏移: {z_offset:+.4f} m")
    
    if abs(z_offset) < 0.005:
        print("\n高度已经合适，无需调整")
        return False
    
    # 执行调整
    if file_type == 'retargeted':
        return adjust_height_retargeted(input_file, output_file, z_offset)
    else:
        return adjust_height_converted(input_file, output_file, z_offset, output_csv)


def main():
    parser = argparse.ArgumentParser(description="调整NPZ文件的整体高度")
    parser.add_argument("input", type=str, help="输入NPZ文件")
    parser.add_argument("output", type=str, help="输出NPZ文件")
    parser.add_argument("offset", type=str, nargs='?', default=None,
                       help="Z轴偏移量（米），正数上升，负数下降，或使用--auto自动计算")
    parser.add_argument("--auto", action="store_true", help="自动计算合适的偏移量")
    parser.add_argument("--init_frames", type=int, default=5,
                       help="用于计算offset的初始帧数（默认5帧）")
    parser.add_argument("--foot_thickness", type=float, default=0.03,
                       help="脚掌厚度（默认0.03米，即3cm）")
    parser.add_argument("--output_csv", action="store_true",
                       help="同时输出CSV格式文件")
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    output_path = Path(args.output)
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("高度调整工具")
    print("=" * 80)
    
    # 检测文件类型
    data = np.load(input_path)
    file_type = detect_file_type(data)
    
    # 执行调整
    if args.auto or args.offset == '--auto':
        success = auto_adjust(
            str(input_path),
            str(output_path),
            init_frames=args.init_frames,
            foot_thickness=args.foot_thickness,
            output_csv=args.output_csv
        )
    elif args.offset is None:
        print("错误: 请指定偏移量或使用--auto")
        return
    else:
        try:
            z_offset = float(args.offset)
            if file_type == 'retargeted':
                success = adjust_height_retargeted(str(input_path), str(output_path), z_offset)
            elif file_type == 'converted':
                success = adjust_height_converted(str(input_path), str(output_path), z_offset, args.output_csv)
            else:
                print("错误: 无法识别的文件格式")
                return
        except ValueError:
            print(f"错误: 无效的偏移量: {args.offset}")
            return
    
    if success:
        print("\n" + "=" * 80)
        print("建议:")
        print(" 使用 relapy_npz.py 播放查看效果")
        print("=" * 80)


if __name__ == "__main__":
    main()
