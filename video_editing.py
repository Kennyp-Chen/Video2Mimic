#!/usr/bin/env python3
"""
用于裁减视频，支持自定义前后裁减时间（可以不同）

Video trimming script.
Trims video by removing custom durations from start and end.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting video duration: {result.stderr}")
        return None
    return float(result.stdout.strip())


def trim_video(input_path, output_path, start_trim=2.0, end_trim=3.0):
    """
    Trim video by removing custom durations from start and end.

    Args:
        input_path: Input video file path
        output_path: Output video file path
        start_trim: Seconds to remove from beginning (default: 2.0)
        end_trim: Seconds to remove from end (default: 3.0)
    """

    # Get original video duration
    original_duration = get_video_duration(input_path)
    if original_duration is None:
        print("Failed to get video duration")
        return False

    print(f"Original video duration: {original_duration:.2f} seconds")
    print(f"Removing {start_trim:.2f} seconds from beginning")
    print(f"Removing {end_trim:.2f} seconds from end")

    total_remove = start_trim + end_trim
    target_duration = original_duration - total_remove

    if target_duration <= 0:
        error_msg = (f"Error: Cannot remove {total_remove:.2f} seconds "
                f"from {original_duration:.2f} second video")
        print(error_msg)
        return False

    start_time = start_trim
    end_time = original_duration - end_trim

    print(f"Keeping segment from {start_time:.2f}s to {end_time:.2f}s")
    print(f"Target duration: {target_duration:.2f} seconds")

    # Use ffmpeg to trim the video
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ss', str(start_time),
        '-t', str(target_duration),
        '-c:v', 'libx264',  # Re-encode video for accurate seeking
        '-c:a', 'aac',      # Re-encode audio
        '-y',  # Overwrite output file
        output_path
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Video successfully trimmed and saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Trim video by removing custom durations from start/end.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python video_editing.py input.mp4                     # Use default trim
  python video_editing.py input.mp4 --start 1.5 --end 2.0
  python video_editing.py input.mp4 -s 0.5 -e 1.0
        '''
    )

    parser.add_argument('input_video',
                    help='Input video file path')
    parser.add_argument('-s', '--start', type=float, default=2.0,
                    help='Seconds to remove from beginning (default: 2.0)')
    parser.add_argument('-e', '--end', type=float, default=3.0,
                    help='Seconds to remove from end (default: 3.0)')
    parser.add_argument('-o', '--output',
                    help='Output video file path (auto-generated)')

    args = parser.parse_args()

    input_video = args.input_video
    start_trim_seconds = args.start
    end_trim_seconds = args.end

    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Input video not found: {input_video}")
        sys.exit(1)

    # Generate output filename if not specified
    input_path = Path(input_video)
    if args.output:
        output_video = args.output
    else:
        output_name = (f"{input_path.stem}_trimmed_start{start_trim_seconds}s"
                       f"_end{end_trim_seconds}s.mp4")
        output_video = input_path.parent / output_name

    print(f"Input video: {input_video}")
    print(f"Output video: {output_video}")
    print(f"Start trim: {start_trim_seconds} seconds")
    print(f"End trim: {end_trim_seconds} seconds")
    print("-" * 50)

    # Trim video
    success = trim_video(input_video, str(output_video),
                        start_trim_seconds, end_trim_seconds)

    if success:
        print("-" * 50)
        print("Video trimming completed successfully!")

        # Verify the output
        output_duration = get_video_duration(str(output_video))
        if output_duration:
            print(f"Output video duration: {output_duration:.2f} seconds")
    else:
        print("Video trimming failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
