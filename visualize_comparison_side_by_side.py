import os
import sys
import torch
import cv2
import json
import numpy as np
import torchvision
from tqdm import tqdm
from ultralytics import YOLO

# auto locate project rootdirectory
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from models.yolo.yolo_with_fusion import YOLOWithFusion
from utils.device_utils import setup_device
from ultralytics.utils.ops import xywh2xyxy

# ================= 🔧 configuration area =================

# 🔥🔥🔥 threshold adjustment area (modify here manually) 🔥🔥🔥
CONF_THRES_BASE = 0.40  # Baseline (YOLOv8) threshold
CONF_THRES_OURS = 0.40  # Ours (HA-GNN) threshold

DATASET_DIR = "stage2_gnn_dataset_GAP_1280"
# Ours weight
CHECKPOINT_OURS = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_GAP_1280\epoch_50.pth"
# Baseline weight (pure YOLOv8)
CHECKPOINT_BASE = r"D:\YOLO-GNN-Fusion\weights\yolov8s.pt"

# outputdirectory
OUTPUT_DIR = os.path.join(project_root, "comparison_vis", "side_by_side")

# category mapping (0-based)
CLASS_MAP = {
    0: 'Car', 1: 'Person', 2: 'Bicycle', 3: 'Motorcycle',
    4: 'Bus', 5: 'Truck', 6: 'Tramway', 7: 'Escooter'
}

# color table (BGRformat) - Baseline and Ours countoneusingthistable
COLORS_OURS = {
    0: (0, 255, 0),  # Car: Green
    1: (0, 255, 255),  # Person: Yellow
    2: (0, 165, 255),  # Bicycle: Orange
    3: (255, 144, 30),  # Motorcycle: Blueish
    4: (0, 0, 255),  # Bus: Red
    5: (255, 0, 255),  # Truck: Magenta
    6: (128, 0, 128),  # Tramway
    7: (255, 255, 0),  # Escooter
    'default': (200, 200, 200)
}


# ================= 🛠️ utility functions =================
def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


def apply_nms(boxes, scores, labels, iou_thres=0.3):
    if boxes.numel() == 0: return [], [], []
    max_wh = 4096
    c = labels.float() * max_wh
    boxes_offset = boxes + c.unsqueeze(1)
    keep_indices = torchvision.ops.nms(boxes_offset, scores, iou_thres)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def draw_box(img, box, cls_id, score, is_ours=True):
    x1, y1, x2, y2 = map(int, box)

    # countoneusing COLORS_OURS getcolor
    color = COLORS_OURS.get(cls_id, COLORS_OURS['default'])
    label = f"{CLASS_MAP.get(cls_id, 'Unk')} {score:.2f}"

    # toareascore/divide Baseline and Ours，Baseline usefinea bitofline
    thick = 2 if is_ours else 1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

    # labelbackground
    font_scale = 0.6
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    t_y1 = y1
    t_y2 = y1 + th + baseline + 5
    if t_y1 - th < 0:
        t_y1 = y1
        t_y2 = y1 + th + 10
    else:
        t_y2 = y1
        t_y1 = y1 - th - 10

    cv2.rectangle(img, (x1, t_y1), (x1 + tw + 5, t_y2), color, -1)

    # textcolor
    text_color = (0, 0, 0)
    cv2.putText(img, label, (x1 + 2, t_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)


def main():
    device = setup_device()
    print(f"🚀 startgeneratecomparisonimage (Basethreshold={CONF_THRES_BASE}, Oursthreshold={CONF_THRES_OURS})...")

    # 1. load Baseline (YOLOv8)
    model_base = YOLO(CHECKPOINT_BASE)

    # 2. load Ours (Stage 2)
    list_path = os.path.join(project_root, DATASET_DIR, "stage2_test_list.json")
    with open(list_path, 'r') as f:
        samples = json.load(f)

    p0 = samples[0]['path']
    if not os.path.isabs(p0): p0 = os.path.join(project_root, p0)
    feat_dim = torch.load(p0, map_location='cpu')['features'].shape[1]

    model_ours = HierarchicalGNN(node_feature_dim=feat_dim, hidden_dim=256).to(device)
    head_ex = torch.nn.Linear(128, 1).to(device)
    head_cl = torch.nn.Linear(128, 8).to(device)

    ckpt = torch.load(CHECKPOINT_OURS, map_location=device)
    if 'model' in ckpt:
        model_ours.load_state_dict(ckpt['model'])
        head_ex.load_state_dict(ckpt['head_exist'])
        head_cl.load_state_dict(ckpt['head_cls'])
    else:
        model_ours.load_state_dict(ckpt, strict=False)
    model_ours.eval();
    head_ex.eval();
    head_cl.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. traverse
    save_count = 0
    for info in tqdm(samples, desc="Comparing"):
        try:
            pt_path = info['path']
            if not os.path.isabs(pt_path): pt_path = os.path.join(project_root, pt_path)

            parts = info['id'].split('_')
            img_path = f"D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T/rgb/{parts[0]}/{parts[1]}.png"
            if not os.path.exists(img_path): continue

            img0 = cv2.imread(img_path)
            h0, w0 = img0.shape[:2]

            # === Ours Inference ===
            data = torch.load(pt_path, map_location=device)
            feat = torch.nn.functional.normalize(torch.nan_to_num(data['features'].float(), 0.0).clamp(-100, 100), p=2,
                                                 dim=1)
            norm_xywh = data['boxes'].float()
            pixel_xywh = norm_xywh * 1280
            pixel_xyxy = torch.zeros_like(pixel_xywh)
            pixel_xyxy[:, 0] = pixel_xywh[:, 0] - pixel_xywh[:, 2] / 2
            pixel_xyxy[:, 1] = pixel_xywh[:, 1] - pixel_xywh[:, 3] / 2
            pixel_xyxy[:, 2] = pixel_xywh[:, 0] + pixel_xywh[:, 2] / 2
            pixel_xyxy[:, 3] = pixel_xywh[:, 1] + pixel_xywh[:, 3] / 2

            scores = data['scores'][:, 0]
            with torch.no_grad():
                out, keep, _ = model_ours([feat], [norm_xywh], [scores])
                p_ex = torch.sigmoid(head_ex(out))
                p_cl = torch.argmax(head_cl(out), dim=1)

            orig_boxes_ours = scale_coords((1280, 1280), pixel_xyxy.clone(), (h0, w0))

            ours_boxes, ours_scores, ours_cls = [], [], []
            if keep and len(keep) > 0:
                for i, idx in enumerate(keep[0]):
                    conf = p_ex[i].item()
                    # 🔥 use threshold from configuration
                    if conf > CONF_THRES_OURS:
                        cls = p_cl[i].item()
                        if cls not in CLASS_MAP: continue
                        ours_boxes.append(orig_boxes_ours[int(idx)])
                        ours_scores.append(conf)
                        ours_cls.append(cls)

            if ours_boxes:
                ob, os_t, oc = torch.stack(ours_boxes), torch.tensor(ours_scores).to(device), torch.tensor(ours_cls).to(
                    device)
                ob, os_t, oc = apply_nms(ob, os_t, oc)
                ours_boxes = ob
                ours_scores = os_t
                ours_cls = oc

            # === Baseline Inference ===
            res_base = model_base(img0, verbose=False, imgsz=1280)[0]
            base_boxes = res_base.boxes.xyxy.cpu()
            base_cls = res_base.boxes.cls.cpu()
            base_conf = res_base.boxes.conf.cpu()

            canvas_base = img0.copy()
            canvas_ours = img0.copy()

            # COCOmappingtable
            coco_map = {0: 1, 1: 2, 2: 0, 3: 3, 5: 4, 7: 5}

            for b, c, s in zip(base_boxes, base_cls, base_conf):
                # 🔥 use threshold from configuration
                if s < CONF_THRES_BASE: continue

                coco_id = int(c)
                if coco_id not in coco_map: continue
                mapped_id = coco_map[coco_id]
                draw_box(canvas_base, b, mapped_id, s.item(), is_ours=False)

            # Draw Ours
            if len(ours_boxes) > 0:
                for b, s, c in zip(ours_boxes, ours_scores, ours_cls):
                    draw_box(canvas_ours, b, c.item(), s.item(), is_ours=True)

            # Add Titles
            cv2.putText(canvas_base, "Baseline (YOLOv8)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 255), 3)
            cv2.putText(canvas_ours, "Ours (HA-GNN)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 255, 50), 3)

            # Stitch
            combined = np.hstack([canvas_base, canvas_ours])

            save_name = f"{parts[0]}_{parts[1]}_compare.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), combined)
            save_count += 1

        except Exception as e:
            continue

    print(f"\n✅ comparisonimagegeneratecompleted！total {save_count} tensor，please check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()