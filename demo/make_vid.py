import yaml
import subprocess
import glob

with open("configs/demo_config.yaml", 'r') as f:
    demo_config = yaml.safe_load(f)

YOLO_MODEL_PATH = demo_config['model']['yolo_model_path']
FRAMEDIFF_MODEL_PATH = demo_config['model']['framediff_model_path']
DEMO_VIDEO_NAME = demo_config['data']['demo_video_name']
FRAMES_PATH = demo_config['data']['frames_folder']
SRC_PATH = demo_config['modules']['src_path']
SKIP_K_FRAMES = demo_config['output']['skip_k_frames']
LEARNED_SKIP_FRAMES = demo_config['output']['learned_skip_frames']
FULL_COMPUTE_FRAMES = demo_config['output']['full_compute_frames']
VIDEO_OUTPUT = demo_config['output']['video_output']

for folder in [SKIP_K_FRAMES, LEARNED_SKIP_FRAMES, FULL_COMPUTE_FRAMES]:
    frames = sorted(glob.glob(f"{folder}/*.jpg"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(frames) == 0:
        print(f"No frames found in {folder}, skipping video compilation.")
        continue

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-pattern_type", "glob",
        "-i", f"{folder}/*.jpg",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",  # needed for browser/windows compatibility
        f"{VIDEO_OUTPUT}/{folder.split('/')[-1]}_output.mp4"
    ])
    print(f"Saved {len(frames)} frames to {VIDEO_OUTPUT}/{folder.split('/')[-1]}_output.mp4")
