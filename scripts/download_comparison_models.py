# scripts/download_comparison_models.py
import torch
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# addprojectrootdirectory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights", "comparison")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"✅ alreadyexist: {save_path}")
        return

    print(f"⬇️ positiveat/inbelowload: {os.path.basename(save_path)} ...")
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
    print(f"📂 weightsavedirectory: {WEIGHTS_DIR}\n")

    # 1. Faster R-CNN (ResNet50-FPN)
    # PyTorch officialweightusuallyautobelowloadtocache，hereourmanualbelowloadoneaoftenuseof .pth forload
    # hereourusing torchvision ofloadmethodstyle，稍afterat/inevaluationscriptinwillautoprocessing，
    # but to match your"belowloadtofolder"ofneed求，ourbelowloadonea COCO predict/pretrainingweight
    print("1️⃣ prepare Faster R-CNN (ResNet50-FPN)...")
    faster_rcnn_url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    download_file(faster_rcnn_url, os.path.join(WEIGHTS_DIR, "fasterrcnn_resnet50_fpn.pth"))

    # 2. RT-DETR (as DETR ofrepresent，propertyability/canmoregoodandcompatiblepropertygood)
    # Ultralytics officialweight
    print("\n2️⃣ prepare RT-DETR (Transformer detection)...")
    rtdetr_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt"
    download_file(rtdetr_url, os.path.join(WEIGHTS_DIR, "rtdetr-l.pt"))

    # 3. YOLOv8-Nano (as RTMDet of轻amounttransformreplacegeneration/comparison)
    # ifyounot想install mmdetection，YOLOv8n ismostgoodof轻amounttransformcomparison
    print("\n3️⃣ prepare YOLOv8-Nano (轻amounttransformdetection)...")
    yolov8n_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    download_file(yolov8n_url, os.path.join(WEIGHTS_DIR, "yolov8n.pt"))

    # 4. RetinaNet (optional，assinglestagenon YOLO ofcomparison，ifneedof话)
    print("\n4️⃣ prepare RetinaNet (ResNet50-FPN)...")
    retinanet_url = "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth"
    download_file(retinanet_url, os.path.join(WEIGHTS_DIR, "retinanet_resnet50_fpn.pth"))

    print("\n✅ allweightbelowloadcompleted！")
    print(f"   pleasecheck: {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()