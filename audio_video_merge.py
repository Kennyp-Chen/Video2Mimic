#!/usr/bin/env python3
"""
视频音频融合脚本
将音频与视频合并，支持自定义音频开始播放时间

Video audio merge script.
Merges audio with video, supports custom audio start time.
"""

import argparse
import os
import subprocess
import sys


def merge_audio_video(input_video, input_audio, output_path, audio_start=0.0):
    """
    Merge audio with video, starting audio at specified time.

    Args:
        input_video: Input video file path
        input_audio: Input audio file path
        output_path: Output video file path
        audio_start: Seconds to delay audio start (default: 0.0)
    """
    print(f"Input video: {input_video}")
    print(f"Input audio: {input_audio}")
    print(f"Audio start delay: {audio_start:.2f} seconds")
    print(f"Output video: {output_path}")
    print("-" * 50)

    # Check if input files exist
    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        return False

    if not os.path.exists(input_audio):
        print(f"Error: Input audio not found: {input_audio}")
        return False

    # Use ffmpeg to merge audio with video
    # -af adelay=delay1|delay2: delay audio in milliseconds
    # We need to convert seconds to milliseconds
    delay_ms = int(audio_start * 1000)
    
    cmd = [
        'ffmpeg', '-i', input_video, '-i', input_audio,
        '-c:v', 'copy',  # Copy video stream without re-encoding
        '-c:a', 'aac',  # Encode audio to AAC
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        '-shortest',  # End when the shortest input ends
        '-y',  # Overwrite output file
        output_path
    ]

    # If audio_start > 0, add audio delay filter
    if audio_start > 0:
        cmd = [
            'ffmpeg', '-i', input_video, '-i', input_audio,
            '-filter_complex',
            f'[1:a]adelay={delay_ms}|{delay_ms}[aout]',
            '-map', '0:v:0',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            output_path
        ]

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Video successfully merged and saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Merge audio with video, supporting custom audio start time.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python audio_video_merge.py video.mp4 audio.mp3
  python audio_video_merge.py video.mp4 audio.mp3 --start 3.5
  python audio_video_merge.py video.mp4 audio.mp3 -s 2.0 -o output.mp4
        '''
    )

    parser.add_argument('input_video',
                        help='Input video file path')
    parser.add_argument('input_audio',
                        help='Input audio file path')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='Audio start delay in seconds (default: 0.0)')
    parser.add_argument('-o', '--output',
                        help='Output video file path (default: overwrites input video)')

    args = parser.parse_args()

    input_video = args.input_video
    input_audio = args.input_audio
    audio_start_seconds = args.start

    # Generate output path if not specified
    if args.output:
        output_video = args.output
    else:
        # Default to overwriting the input video
        output_video = input_video

    # Merge audio and video
    success = merge_audio_video(input_video, input_audio, output_video,
                                audio_start_seconds)

    if success:
        print("-" * 50)
        print("Audio-video merge completed successfully!")
    else:
        print("Audio-video merge failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
