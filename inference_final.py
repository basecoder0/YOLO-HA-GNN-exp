import os
import sys
import torch
import cv2
import json
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm

# Auto-locate project root directory
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from utils.device_utils import setup_device

# 🔥 Import LiDAR tools
try:
    from utils.lidar_utils import LidarProcessor
except ImportError:
    print("❌ Error: Could not find utils.lidar_utils, please ensure this file exists!")
    sys.exit(1)

# ================= 🔧 Core threshold configuration area (modify manually here) =================

# 1. Detection confidence threshold (GNN Confidence Threshold)
# Function: Only objects with confidence greater than this value will be drawn and risk calculated.
# Lower (e.g., 0.2): Detect more objects, prevent missed detections, but may increase false detections.
# Higher (e.g., 0.6): Keep only very certain objects, cleaner image, but may miss detections.
CONF_THRES = 0.40

# 2. Risk level thresholds (Risk Level Thresholds)
# Function: Determine the color of the box (red/orange/green).
# Logic: Risk value > HIGH -> Red (dangerous)
#       Risk value > MID  -> Orange (warning)
#       Others          -> Green (safe)
RISK_THRES_HIGH = 0.70  # High risk threshold
RISK_THRES_MID = 0.40  # Mid risk threshold

# 3. NMS deduplication threshold (IoU Threshold)
# Function: If two boxes overlap (IoU) exceeds this value, keep only the one with higher score.
# Usually keep 0.3 or 0.45, no need to change frequently.
NMS_IOU_THRES = 0.30

# ================= 📂 Path configuration area =================
DATASET_DIR = "stage2_gnn_dataset_GAP_1280"
CHECKPOINT = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_GAP_1280\epoch_50.pth"
OUTPUT_DIR = os.path.join(project_root, "comparison_vis", "risk_assessment_final")

ROOT_DATA_DIR = r"D:\BaiduNetdiskDownload\R-LiViT\R-LiViT"
MAPPING_FILE = os.path.join(ROOT_DATA_DIR, "rgb_to_lidar_sequenceuenceuence_mapping.csv")
CALIB_FILE = os.path.join(ROOT_DATA_DIR, "calibration_params.json")

# Class mapping
CLASS_MAP = {
    0: 'Car', 1: 'Person', 2: 'Bicycle', 3: 'Motorcycle',
    4: 'Bus', 5: 'Truck', 6: 'Tramway', 7: 'Escooter'
}


# ================= 🧠 Risk calculator (TrafficRiskCalculator) =================
class TrafficRiskCalculator:
    def __init__(self):
        self.idx2name = {
            0: 'car', 1: 'person', 2: 'bicycle', 3: 'motorcycle',
            4: 'bus', 5: 'truck', 6: 'tramway', 7: 'escooter'
        }
        self.params = {
            'person': {'w': 1.00, 'd_safe': 10.0, 'alpha': 0.1, 'beta': 2.5},
            'bicycle': {'w': 0.90, 'd_safe': 12.0, 'alpha': 0.2, 'beta': 2.0},
            'escooter': {'w': 0.95, 'd_safe': 12.0, 'alpha': 0.2, 'beta': 2.0},
            'motorcycle': {'w': 0.85, 'd_safe': 15.0, 'alpha': 0.3, 'beta': 1.8},
            'car': {'w': 0.60, 'd_safe': 15.0, 'alpha': 0.3, 'beta': 1.5},
            'bus': {'w': 0.80, 'd_safe': 20.0, 'alpha': 0.2, 'beta': 1.2},
            'truck': {'w': 0.80, 'd_safe': 20.0, 'alpha': 0.2, 'beta': 1.2},
            'tramway': {'w': 0.70, 'd_safe': 25.0, 'alpha': 0.1, 'beta': 1.5},
        }
        self.default_param = {'w': 0.5, 'd_safe': 15.0, 'alpha': 0.1, 'beta': 1.5}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate(self, cls_id, conf, depth, speed=0.0):
        cls_name = self.idx2name.get(int(cls_id), 'car').lower()
        p = self.params.get(cls_name, self.default_param)

        # 1. Base risk
        r_base = p['w'] * conf
        # 2. Motion risk
        r_motion = 1.0 + p['alpha'] * abs(speed)
        # 3. Distance risk
        if depth is None or np.isnan(depth) or depth <= 0: depth = 999.0
        dist_diff = p['d_safe'] - depth
        r_prox = self.sigmoid(p['beta'] * dist_diff)

        r_final = r_base * r_motion * r_prox
        return min(r_final, 1.0)


# ================= 🛠️ Utility functions =================
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


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


def draw_risk_box(img, box, cls_id, score, risk):
    x1, y1, x2, y2 = map(int, box)

    # 🔥 Use configured risk threshold
    if risk > RISK_THRES_HIGH:
        color = (0, 0, 255)  # Red (High Risk)
        text_color = (255, 255, 255)
    elif risk > RISK_THRES_MID:
        color = (0, 165, 255)  # Orange (Medium Risk)
        text_color = (0, 0, 0)
    else:
        color = (0, 255, 0)  # Green (Safe)
        text_color = (0, 0, 0)

    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Label content
    cls_name = CLASS_MAP.get(cls_id, 'Unk')
    label = f"{cls_name}: {score:.2f} R: {risk:.2f}"

    # Label background
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    t_y2 = y1
    t_y1 = y1 - th - 6
    if t_y1 < 0:
        t_y1 = y1
        t_y2 = y1 + th + 6

    cv2.rectangle(img, (x1, t_y1), (x1 + tw + 10, t_y2), color, -1)
    cv2.putText(img, label, (x1 + 5, t_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


# ================= 🚀 Main program =================
def main():
    device = setup_device()
    print(f"🚀 Starting risk assessment visualization...")
    print(f"   - Detection threshold (Conf): {CONF_THRES}")
    print(f"   - High risk threshold (Red): {RISK_THRES_HIGH}")
    print(f"   - Mid risk threshold (Orange): {RISK_THRES_MID}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Mapping file
    if not os.path.exists(MAPPING_FILE):
        print(f"❌ Not found: {MAPPING_FILE}")
        return
    df_map = pd.read_csv(MAPPING_FILE, dtype=str)
    rgb2lidar = {row['rgbt_sequenceuence_id'].strip().zfill(3): row['lidar_sequenceuence_id'].strip().zfill(4) for _, row in
                 df_map.iterrows()}

    # 2. Initialize
    risk_calc = TrafficRiskCalculator()
    try:
        lidar_proc = LidarProcessor(CALIB_FILE)
        print("✅ LiDAR module ready")
    except Exception as e:
        print(f"❌ LiDAR initialization failed: {e}")
        return

    # 3. Model loading
    list_path = os.path.join(project_root, DATASET_DIR, "stage2_test_list.json")
    with open(list_path, 'r') as f:
        samples = json.load(f)
    p0 = samples[0]['path']
    if not os.path.isabs(p0): p0 = os.path.join(project_root, p0)
    feat_dim = torch.load(p0, map_location='cpu')['features'].shape[1]

    model = HierarchicalGNN(node_feature_dim=feat_dim, hidden_dim=256).to(device)
    head_ex = torch.nn.Linear(128, 1).to(device)
    head_cl = torch.nn.Linear(128, 8).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    if 'head_exist' in ckpt:
        head_ex.load_state_dict(ckpt['head_exist'])
        head_cl.load_state_dict(ckpt['head_cls'])
    model.eval();
    head_ex.eval();
    head_cl.eval()

    print(f"🚀 Starting to process {len(samples)} images...")

    for info in tqdm(samples, desc="Processing"):
        try:
            pt_path = info['path']
            if not os.path.isabs(pt_path): pt_path = os.path.join(project_root, pt_path)

            parts = info['id'].split('_')
            rgb_sequenceuence = parts[0]
            frame_id = parts[1]

            if rgb_sequenceuence not in rgb2lidar: continue
            lidar_sequenceuence = rgb2lidar[rgb_sequenceuence]

            img_path = os.path.join(ROOT_DATA_DIR, 'R-LiViT_RGB-T', 'rgb', rgb_sequenceuence, f"{frame_id}.png")
            try:
                l_idx = int(frame_id)
                lidar_path = os.path.join(ROOT_DATA_DIR, 'R-LiViT_LiDAR', 'point_clouds', lidar_sequenceuence, f"{l_idx:06d}.bin")
            except:
                continue

            if not os.path.exists(img_path) or not os.path.exists(lidar_path): continue

            img0 = cv2.imread(img_path)
            h0, w0 = img0.shape[:2]
            points_3d = lidar_proc.load_bin_file(lidar_path)

            # GNN inference
            data = torch.load(pt_path, map_location=device)
            feat = torch.nn.functional.normalize(torch.nan_to_num(data['features'].float(), 0.0).clamp(-100, 100), p=2,
                                                 dim=1)
            norm_xywh = data['boxes'].float()
            scores = data['scores'][:, 0]

            pixel_xywh = norm_xywh * 1280
            pixel_xyxy = xywh2xyxy(pixel_xywh)
            orig_boxes_all = scale_coords((1280, 1280), pixel_xyxy.clone(), (h0, w0))

            with torch.no_grad():
                out, keep, _ = model([feat], [norm_xywh], [scores])
                p_ex = torch.sigmoid(head_ex(out)).view(-1)
                p_cl = torch.argmax(head_cl(out), dim=1)

            if keep and len(keep) > 0 and len(keep[0]) > 0:
                keep_idxs = keep[0]
                boxes_t, scores_t, cls_t = [], [], []

                for k_i, idx in enumerate(keep_idxs):
                    # 🔥 Use configured detection threshold
                    if p_ex[k_i] > CONF_THRES:
                        raw_idx = int(idx)
                        boxes_t.append(orig_boxes_all[raw_idx])
                        scores_t.append(p_ex[k_i])
                        cls_t.append(p_cl[k_i])

                if boxes_t:
                    bt = torch.stack(boxes_t)
                    st = torch.stack(scores_t)
                    ct = torch.stack(cls_t)

                    # 🔥 Use configured NMS threshold
                    c_offset = ct.float() * 4096
                    keep_nms = torchvision.ops.nms(bt + c_offset.unsqueeze(1), st, NMS_IOU_THRES)

                    for ki in keep_nms:
                        final_box = bt[ki]
                        final_score = st[ki].item()
                        final_cls = ct[ki].item()

                        if final_cls not in CLASS_MAP: continue

                        # 1. Physical distance measurement
                        bbox_list = final_box.cpu().numpy().tolist()
                        depth, speed, _ = lidar_proc.get_object_depth_and_speed(points_3d, bbox_list)

                        # 2. Risk calculation
                        risk = risk_calc.calculate(final_cls, final_score, depth, speed)

                        # 3. Drawing
                        draw_risk_box(img0, final_box, final_cls, final_score, risk)

            save_name = f"{rgb_sequenceuence}_{frame_id}_risk.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), img0)

        except Exception as e:
            continue

    print(f"\n✅ Risk visualization complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()