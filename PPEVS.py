import os
import cv2
import numpy as np
import subprocess
from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm
import shutil
import glob
import math

# 权重目录
os.environ['TORCH_HOME'] = os.getcwd()

# 强制使用编号为0的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 导入torch后设置设备
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"使用GPU: {torch.cuda.get_device_name(0)} (物理GPU 0)")

# 初始化 OCR
ocr = PaddleOCR(use_textline_orientation=True, lang='ch', use_gpu=True)

def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def extract_frames(video_path, out_dir, start_frame=0, end_frame=None):
    """
    提取视频帧到 out_dir，支持指定帧范围
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame is None:
        end_frame = total_frames
    
    count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 只保存指定范围内的帧
        if start_frame <= count < end_frame:
            cv2.imwrite(f"{out_dir}/{extracted_count:05d}.png", frame)
            extracted_count += 1
            
        count += 1
        
        # 超过结束帧就停止
        if count >= end_frame:
            break
    
    cap.release()
    return fps, extracted_count

def extract_and_trim_audio(video_path, audio_out, start_time, duration):
    """
    提取指定时间段的音频
    """
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),  # 开始时间
        "-t", str(duration),     # 持续时间
        "-vn", "-acodec", "copy",
        audio_out
    ], check=True)

def generate_mask(frame, target_y_start=1200, target_y_end=1400):
    """
    只在指定的纵坐标范围内生成掩膜
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 创建指定范围的ROI
    roi_frame = frame[target_y_start:target_y_end, :]
    
    # 对ROI进行OCR
    results = ocr.ocr(roi_frame)
    if not results or not results[0]:
        return mask
    
    for item in results[0]:
        try:
            box = item[0]
            text_info = item[1]
            text = text_info[0] if isinstance(text_info, tuple) else str(text_info)
            confidence = text_info[1] if isinstance(text_info, tuple) else 1.0
            
            # 过滤低置信度文本
            if confidence < 0.5:
                continue
                
            # 将ROI坐标转换回原图坐标
            pts = np.array(box, dtype=np.int32).reshape(-1, 2)
            pts[:, 1] += target_y_start  # Y坐标偏移
            
            # 检查是否在目标范围内
            if np.all(pts[:, 1] >= target_y_start) and np.all(pts[:, 1] <= target_y_end):
                cv2.fillPoly(mask, [pts], 255)
                
        except Exception as e:
            print(f"处理文本框时出错: {e}")
            continue
    
    # 如果有检测到的文本区域，进行形态学操作
    if np.any(mask > 0):
        # 水平膨胀
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        mask = cv2.dilate(mask, kernel_h, iterations=1)
        
        # 垂直膨胀
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
        mask = cv2.dilate(mask, kernel_v, iterations=1)
        
        # 闭运算连接临近区域
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    return mask

def run_propainter_inpainting(frames_dir, masks_dir, output_dir):
    """
    使用ProPainter进行时空修复
    """
    # 检查ProPainter是否在hub目录下
    current_dir = os.getcwd()
    propainter_script = os.path.join(current_dir, "hub", "ProPainter", "inference_propainter.py")
    
    print(f"当前工作目录: {current_dir}")
    print(f"查找ProPainter脚本: {propainter_script}")
    
    if not os.path.exists(propainter_script):
        raise FileNotFoundError(f"ProPainter script not found at {propainter_script}")
    
    # 转换为绝对路径
    frames_dir = os.path.abspath(frames_dir)
    masks_dir = os.path.abspath(masks_dir)
    output_dir = os.path.abspath(output_dir)
    
    # 构建ProPainter命令 - 优化版本
    cmd = [
        "python", "inference_propainter.py",
        "-i", frames_dir,
        "-m", masks_dir,
        "-o", output_dir,
        "--save_fps", "30",
        "--save_frames",  # 保存帧
        "--fp16",         # 半精度加速
        "--height", "960",
        "--width", "540",
        "--subvideo_length", "40",
        "--neighbor_length", "10",
        "--ref_stride", "3",
        "--raft_iter", "10"
    ]
    
    # 运行ProPainter
    print("运行ProPainter进行时空修复...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: ./hub/ProPainter")
    
    # 设置工作目录为ProPainter目录
    work_dir = "./hub/ProPainter"
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"使用GPU设备: {env.get('CUDA_VISIBLE_DEVICES', 'default')}")
    
    try:
        # 运行ProPainter
        result = subprocess.run(cmd, cwd=work_dir, check=True, env=env)
        print("ProPainter执行成功!")
        
        # 查找生成的视频文件
        propainter_video_path = os.path.join(output_dir, "frames", "inpaint_out.mp4")
        
        print(f"查找ProPainter输出视频: {propainter_video_path}")
        
        if os.path.exists(propainter_video_path):
            propainter_video = os.path.abspath(propainter_video_path)
            print(f"✓ 找到ProPainter输出视频: {propainter_video}")
            return propainter_video
        else:
            print(f"✗ 未找到视频文件: {propainter_video_path}")
            # 列出输出目录内容进行调试
            print("输出目录结构:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"ProPainter执行失败: {e}")
        raise e  # 直接抛出异常而不是返回None

def debug_mask_visualization(frames_dir, masks_dir, total_frames, debug_output="debug_masks"):
    """
    生成掩膜可视化图像用于调试
    """
    os.makedirs(debug_output, exist_ok=True)
    
    for i in range(min(5, total_frames)):  # 只可视化前5帧
        frame = cv2.imread(f"{frames_dir}/{i:05d}.png")
        mask = cv2.imread(f"{masks_dir}/{i:05d}.png", 0)
        
        if frame is not None and mask is not None:
            # 创建可视化
            overlay = frame.copy()
            overlay[mask > 0] = [0, 0, 255]  # 红色显示掩膜区域
            result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            cv2.imwrite(f"{debug_output}/mask_debug_{i:05d}.png", result)
    
    print(f"掩膜调试图像保存到: {debug_output}")

def process_video_segment(in_vid, out_vid, start_frame, end_frame, segment_index):
    """
    处理视频的单个片段
    参数:
    - start_frame, end_frame: 实际需要的帧范围（100帧）
    - 会自动添加前后15帧作为参考帧进行处理
    - 最终只保留中间的100帧用于合成
    """
    # 创建二级目录结构
    processing_root = "processing_segments"
    os.makedirs(processing_root, exist_ok=True)
    
    base = os.path.join(processing_root, f"segment_{segment_index:03d}_{start_frame}_{end_frame}")
    frames_dir = os.path.join(base, "frames")
    masks_dir = os.path.join(base, "masks") 
    propainter_output = os.path.join(base, "propainter_output")
    trimmed_output = os.path.join(base, "trimmed_output")
    audio_tmp = os.path.join(base, "audio.aac")
    
    # 创建目录（不删除已存在的）
    for d in [frames_dir, masks_dir, propainter_output, trimmed_output]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(base, exist_ok=True)
    
    try:
        print(f"=== 处理视频片段: {in_vid} 实际帧范围: {start_frame}-{end_frame} ===")
        
        # 计算包含参考帧的范围
        ref_start = max(0, start_frame - 15)
        ref_end = min(get_video_info(in_vid)[1], end_frame + 15)
        
        print(f"包含参考帧的范围: {ref_start}-{ref_end} (共{ref_end - ref_start}帧)")
        
        # 步骤1: 提取包含参考帧的帧
        print("步骤1: 提取视频帧（包含参考帧）...")
        fps, total_extracted = extract_frames(in_vid, frames_dir, ref_start, ref_end)
        
        # 计算实际需要的帧数和音频参数
        actual_frames = end_frame - start_frame
        duration = actual_frames / fps
        start_time = start_frame / fps
        
        print(f"提取了 {total_extracted} 帧用于处理，最终需要 {actual_frames} 帧")
        print(f"最终视频帧率 {fps} FPS，时长 ~{duration:.2f}s")
        
        # 步骤2: 提取对应时间段的音频（只提取实际需要的部分）
        print("步骤2: 提取音频...")
        extract_and_trim_audio(in_vid, audio_tmp, start_time, duration)
        
        # 步骤3: 生成掩膜 - 检测字幕区域
        print("步骤3: 生成字幕掩膜...")
        has_subtitles = False
        
        for i in tqdm(range(total_extracted), desc="Generating masks"):
            frame = cv2.imread(f"{frames_dir}/{i:05d}.png")
            mask = generate_mask(frame, target_y_start=1200, target_y_end=1400)
            cv2.imwrite(f"{masks_dir}/{i:05d}.png", mask)
            
            if np.any(mask > 0):
                has_subtitles = True
        
        if has_subtitles:
            print("✓ 检测到字幕，生成调试可视化...")
            debug_output = os.path.join(base, "debug_masks")
            debug_mask_visualization(frames_dir, masks_dir, total_extracted, debug_output=debug_output)
        
        if not has_subtitles:
            print("⚠ 未检测到字幕，直接提取目标帧...")
            
            # 计算需要提取的帧范围（在已提取的帧中）
            skip_start = start_frame - ref_start  # 需要跳过的开始帧数
            skip_end = skip_start + actual_frames  # 需要提取到的结束帧数
            
            # 提取目标帧到trimmed_output
            for i in range(actual_frames):
                src_frame = f"{frames_dir}/{skip_start + i:05d}.png"
                dst_frame = f"{trimmed_output}/{i:05d}.png"
                shutil.copy2(src_frame, dst_frame)
            
            # 从trimmed_output生成视频
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", f"{trimmed_output}/%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_vid
            ], check=True)
            
            # 合并音频
            temp_video = out_vid.replace('.mp4', '_temp.mp4')
            shutil.move(out_vid, temp_video)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_tmp,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                out_vid
            ], check=True)
            os.remove(temp_video)
            
        else:
            print("步骤4: 使用ProPainter进行时空修复...")
            
            # 使用ProPainter进行修复
            propainter_video = run_propainter_inpainting(frames_dir, masks_dir, propainter_output)
            
            if propainter_video and os.path.exists(propainter_video):
                print("步骤5: 从ProPainter结果中提取目标帧...")
                
                # 首先检查ProPainter的输出结构
                print(f"调试: 检查ProPainter输出结构...")
                print(f"ProPainter视频文件: {propainter_video}")
                
                propainter_frames_dir = os.path.join(propainter_output, "frames", "frames")
                print(f"查找帧目录: {propainter_frames_dir}")
                
                # 列出propainter_output目录结构
                if os.path.exists(propainter_output):
                    print("ProPainter输出目录结构:")
                    for root, dirs, files in os.walk(propainter_output):
                        level = root.replace(propainter_output, '').count(os.sep)
                        indent = ' ' * 2 * level
                        print(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files[:10]:  # 只显示前10个文件
                            print(f"{subindent}{file}")
                        if len(files) > 10:
                            print(f"{subindent}... 还有 {len(files) - 10} 个文件")
                
                # 方法1：尝试从帧文件夹提取
                if os.path.exists(propainter_frames_dir) and len(os.listdir(propainter_frames_dir)) > 0:
                    print("使用方法1: 从帧文件夹提取目标帧")
                    
                    frame_files = sorted([f for f in os.listdir(propainter_frames_dir) if f.endswith('.png')])
                    print(f"找到 {len(frame_files)} 个帧文件")
                    
                    if len(frame_files) >= actual_frames:
                        # 计算需要提取的帧范围
                        skip_start = start_frame - ref_start
                        skip_end = skip_start + actual_frames
                        
                        print(f"从第 {skip_start} 帧开始提取 {actual_frames} 帧")
                        
                        # 提取目标帧到trimmed_output
                        extracted_count = 0
                        for i in range(actual_frames):
                            frame_index = skip_start + i
                            if frame_index < len(frame_files):
                                src_frame = os.path.join(propainter_frames_dir, frame_files[frame_index])
                                dst_frame = f"{trimmed_output}/{i:05d}.png"
                                if os.path.exists(src_frame):
                                    shutil.copy2(src_frame, dst_frame)
                                    extracted_count += 1
                                else:
                                    print(f"警告: 缺少帧 {src_frame}")
                        
                        print(f"成功提取 {extracted_count}/{actual_frames} 帧")
                        
                        if extracted_count >= actual_frames * 0.9:  # 至少90%的帧存在
                            # 从trimmed_output生成最终视频
                            subprocess.run([
                                "ffmpeg", "-y", "-framerate", str(fps),
                                "-i", f"{trimmed_output}/%05d.png",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                out_vid
                            ], check=True)
                        else:
                            print("帧文件不足，使用方法2")
                            raise Exception("帧文件不足")
                    else:
                        print(f"帧文件数量不足 ({len(frame_files)} < {actual_frames})，使用方法2")
                        raise Exception("帧文件数量不足")
                        
                else:
                    print("使用方法2: 直接从ProPainter视频文件中提取")
                    
                    # 方法2：直接从ProPainter的视频文件中提取目标时间段
                    skip_start_time = (start_frame - ref_start) / fps
                    
                    print(f"从ProPainter视频的第 {skip_start_time:.2f} 秒开始提取 {duration:.2f} 秒")
                    
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-i", propainter_video,
                        "-ss", str(skip_start_time),
                        "-t", str(duration),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        out_vid
                    ], check=True)
                
                print("步骤6: 合并音频到修复后的视频...")
                
                # 合并音频
                temp_video = out_vid.replace('.mp4', '_temp.mp4')
                shutil.move(out_vid, temp_video)
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", temp_video,
                    "-i", audio_tmp,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    out_vid
                ], check=True)
                os.remove(temp_video)
                
                print(f"✓ 成功！修复后的视频片段已保存为: {out_vid}")
                
            else:
                print("✗ ProPainter执行失败，抛出异常...")
                raise Exception("ProPainter处理失败")
        
        print(f"=== 处理完成: {out_vid} ===")
        print(f"处理文件保存在: {base}")
        
    except Exception as e:
        print(f"处理片段失败: {e}")
        print(f"错误片段的文件保存在: {base}")
        raise e

def merge_video_segments(segment_videos, output_video):
    """
    合并多个视频片段
    此函数已移至 combine.py 文件
    """
    print("合并功能已移至 combine.py 文件，请使用该文件进行视频合成")
    pass

def process_video_with_segments(in_vid, segment_length=100, test_segments=None):
    """
    分段处理视频，只生成segments，不进行合成
    注意：每段处理130帧（包含参考帧），但最终只保留中间100帧
    
    参数:
    - test_segments: 如果指定，只处理前N个片段用于测试
    """
    # 获取视频信息
    fps, total_frames = get_video_info(in_vid)
    print(f"视频总帧数: {total_frames}, 帧率: {fps}")
    
    # 计算需要处理的片段数
    num_segments = math.ceil(total_frames / segment_length)
    
    # 测试模式：只处理前几个片段
    if test_segments is not None:
        num_segments = min(num_segments, test_segments)
        print(f"🧪 测试模式：只处理前 {num_segments} 个片段")
    
    print(f"将处理 {num_segments} 个片段，每个片段 {segment_length} 帧")
    print(f"每个片段实际处理约 {segment_length + 30} 帧（包含前后参考帧），最终保留 {segment_length} 帧")
    
    # 创建输出目录
    segments_dir = "/home/jay/workspace/EraseVideoSubtitles/PPEVS/video_segments"
    os.makedirs(segments_dir, exist_ok=True)
    
    try:
        # 分段处理
        for i in range(num_segments):
            start_frame = i * segment_length
            end_frame = min((i + 1) * segment_length, total_frames)
            
            segment_output = os.path.join(segments_dir, f"{i:03d}.mp4")
            
            print(f"\n处理片段 {i+1}/{num_segments}: 目标帧 {start_frame}-{end_frame} (共{end_frame - start_frame}帧)")
            
            try:
                process_video_segment(in_vid, segment_output, start_frame, end_frame, i)
                print(f"✓ 片段 {i+1} 处理完成，保存为: {segment_output}")
            except Exception as e:
                print(f"✗ 片段 {i+1} 处理失败: {e}")
                raise e
        
        print(f"\n🎉 所有片段处理完成！")
        if test_segments is not None:
            print(f"🧪 测试完成：处理了前 {test_segments} 个片段")
            print(f"📊 测试结果可用于评估整体处理效果")
        print(f"片段文件保存在: {segments_dir}")
        print(f"处理文件保存在: ./processing_segments/")
        print("注意: 处理文件已保留，如需清理请手动删除 ./processing_segments/ 目录")
        
    except Exception as e:
        print(f"视频处理失败: {e}")
        print(f"已处理的片段文件保存在: {segments_dir}")
        print(f"调试文件保存在: ./processing_segments/")
        raise e

if __name__ == "__main__":
    print("""
    ==============================================
    EVS (Erase Video Subtitles) - 分段处理版本
    ==============================================
    
    功能: 分段处理视频去除字幕，避免内存溢出
    
    工作流程:
    1. 将视频分段处理（每段100帧）
    2. 对每段进行字幕检测和去除
    3. 合并所有处理后的片段
    
    改进特性:
    - 强制使用GPU0
    - 显示详细进度信息
    - 失败时直接报错而非使用原始帧
    - 保留所有处理文件在二级目录结构中
    - 便于调试和检查处理结果
    - 支持测试模式（只处理前N个片段）
    
    """)
    
    input_video = "input_1.mp4"
    
    # ===========================================
    # 🧪 测试模式：只处理前10个片段
    # 注释掉下面这行可以处理完整视频
    # ===========================================
    # print("🧪 开始测试模式：处理前10个片段...")
    # process_video_with_segments(input_video, segment_length=100, test_segments=10)
    
    # ===========================================
    # 🚀 完整处理模式：处理整个视频
    # 取消注释下面这行可以处理完整视频
    # ===========================================
    print("🚀 开始完整处理模式：处理整个视频...")
    process_video_with_segments(input_video, segment_length=100)
    
    print("\n🎉 片段生成完成！")
    print("使用 combine.py 文件进行视频合成")