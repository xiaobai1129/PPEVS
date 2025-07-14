#!/usr/bin/env python3
"""
combine.py - 简单直接的视频合并工具
"""

import os
import subprocess
import glob

def combine_segments(segments_dir, output_video):
    """
    简单直接地合并视频片段
    """
    
    print(f"片段目录: {segments_dir}")
    print(f"输出视频: {output_video}")
    
    # 获取所有片段文件，按文件名排序
    pattern = os.path.join(segments_dir, "*.mp4")
    segment_files = sorted(glob.glob(pattern))
    
    if not segment_files:
        print("❌ 未找到任何视频片段")
        return False
    
    print(f"找到 {len(segment_files)} 个片段:")
    for i, file in enumerate(segment_files):
        print(f"  {i:03d}: {os.path.basename(file)}")
    
    # 创建文件列表
    list_file = "filelist.txt"
    with open(list_file, "w") as f:
        for video_file in segment_files:
            # 使用绝对路径避免路径问题
            abs_path = os.path.abspath(video_file)
            f.write(f"file '{abs_path}'\n")
    
    print(f"\n创建文件列表: {list_file}")
    
    # 显示文件列表内容
    with open(list_file, "r") as f:
        content = f.read()
        print("文件列表内容:")
        print(content)
    
    try:
        # 使用最简单的ffmpeg concat方法
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", 
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_video
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 执行合并
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 合并成功: {output_video}")
            
            # 检查输出文件
            if os.path.exists(output_video):
                file_size = os.path.getsize(output_video) / (1024*1024)  # MB
                print(f"输出文件大小: {file_size:.1f} MB")
            
            return True
        else:
            print(f"❌ ffmpeg执行失败:")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False
    
    finally:
        # 清理临时文件
        if os.path.exists(list_file):
            os.remove(list_file)

if __name__ == "__main__":
    print("=== 简单视频合并工具 ===\n")
    
    # 配置
    segments_dir = "/home/jay/workspace/EraseVideoSubtitles/PPEVS/video_segments"
    output_video = "output_combined.mp4"
    
    # 执行合并
    success = combine_segments(segments_dir, output_video)
    
    if success:
        print(f"\n🎉 完成! 输出文件: {output_video}")
    else:
        print(f"\n❌ 合并失败!")