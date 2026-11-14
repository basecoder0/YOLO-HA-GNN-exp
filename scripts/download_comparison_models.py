# scripts/download_comparison_models.py
import torch
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights", "comparison")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"✅ 已存在: {save_path}")
        return

    print(f"⬇️ 正在下载: {os.path.basename(save_path)} ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def main():
    print(f"📂 权重保存目录: {WEIGHTS_DIR}\n")

    # 1. Faster R-CNN (ResNet50-FPN)
    # PyTorch 官方权重通常自动下载到缓存，这里我们手动下载一个常用的 .pth 用于加载
    # 这里我们使用 torchvision 的加载方式，稍后在评估脚本里会自动处理，
    # 但为了符合你的"下载到文件夹"的要求，我们下载一个 COCO 预训练权重
    print("1️⃣ 准备 Faster R-CNN (ResNet50-FPN)...")
    faster_rcnn_url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    download_file(faster_rcnn_url, os.path.join(WEIGHTS_DIR, "fasterrcnn_resnet50_fpn.pth"))

    # 2. RT-DETR (作为 DETR 的代表，性能更好且兼容性好)
    # Ultralytics 官方权重
    print("\n2️⃣ 准备 RT-DETR (Transformer 检测)...")
    rtdetr_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt"
    download_file(rtdetr_url, os.path.join(WEIGHTS_DIR, "rtdetr-l.pt"))

    # 3. YOLOv8-Nano (作为 RTMDet 的轻量化替代/对比)
    # 如果你不想装 mmdetection，YOLOv8n 是最好的轻量化对比
    print("\n3️⃣ 准备 YOLOv8-Nano (轻量化检测)...")
    yolov8n_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    download_file(yolov8n_url, os.path.join(WEIGHTS_DIR, "yolov8n.pt"))

    # 4. RetinaNet (可选，作为单阶段非 YOLO 的对比，如果需要的话)
    print("\n4️⃣ 准备 RetinaNet (ResNet50-FPN)...")
    retinanet_url = "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth"
    download_file(retinanet_url, os.path.join(WEIGHTS_DIR, "retinanet_resnet50_fpn.pth"))

    print("\n✅ 所有权重下载完成！")
    print(f"   请检查: {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()