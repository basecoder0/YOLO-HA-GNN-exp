import os
import sys
import torch
import cv2
import json
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm

# auto locate project rootdirectory
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from utils.device_utils import setup_device
# assumeyoualreadyhave utils.lidar_utils
from utils.lidar_utils import LidarProcessor

# ================= 🔧 configuration area =================
# datasetdirectory (GNNfeature)
DATASET_DIR = "stage2_gnn_dataset_GAP_1280"
# GNN weight
CHECKPOINT = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_GAP_1280\epoch_50.pth"
# outputdirectory
OUTPUT_DIR = os.path.join(project_root, "comparison_vis", "risk_assessment")

# dataset root directory (forfindimagepieceandradar)
ROOT_DATA_DIR = r"D:\BaiduNetdiskDownload\R-LiViT\R-LiViT"
MAPPING_FILE = os.path.join(ROOT_DATA_DIR, "rgb_to_lidar_sequenceuenceuence_mapping.csv")
CALIB_FILE = os.path.join(ROOT_DATA_DIR, "calibration_params.json")

# category mapping
CLASS_MAP = {
    0: 'Car', 1: 'Person', 2: 'Bicycle', 3: 'Motorcycle',
    4: 'Bus', 5: 'Truck', 6: 'Tramway', 7: 'Escooter'
}


# ================= 🧠 riskcalculationdevice =================
class TrafficRiskCalculator:
    def __init__(self):
        # riskparameter (weight, safety/fulldistance, velocitycoefficient, distancecoefficient)
        self.params = {
            'Person': {'w': 1.00, 'd_safe': 10.0, 'alpha': 0.1, 'beta': 2.5},
            'Bicycle': {'w': 0.90, 'd_safe': 12.0, 'alpha': 0.2, 'beta': 2.0},
            'Escooter': {'w': 0.95, 'd_safe': 12.0, 'alpha': 0.2, 'beta': 2.0},
            'Motorcycle': {'w': 0.85, 'd_safe': 15.0, 'alpha': 0.3, 'beta': 1.8},
            'Car': {'w': 0.60, 'd_safe': 15.0, 'alpha': 0.3, 'beta': 1.5},
            'Bus': {'w': 0.80, 'd_safe': 20.0, 'alpha': 0.2, 'beta': 1.2},
            'Truck': {'w': 0.80, 'd_safe': 20.0, 'alpha': 0.2, 'beta': 1.2},
            'Tramway': {'w': 0.70, 'd_safe': 25.0, 'alpha': 0.1, 'beta': 1.5},
        }
        self.default_param = {'w': 0.5, 'd_safe': 15.0, 'alpha': 0.1, 'beta': 1.5}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate(self, cls_name, conf, depth, speed=0.0):
        p = self.params.get(cls_name, self.default_param)

        # 1. basicrisk (categorysensitivity * confidence)
        r_base = p['w'] * conf

        # 2. motionrisk (velocityfasterriskmore/greaterlarge)
        r_motion = 1.0 + p['alpha'] * abs(speed)

        # 3. distancerisk (distancemore/greatercloseriskmore/greaterlarge)
        if depth <= 0: depth = 999.0  # avoidinvaliddepth
        dist_diff = p['d_safe'] - depth
        r_prox = self.sigmoid(p['beta'] * dist_diff)

        # comprehensiverisk
        r_final = r_base * r_motion * r_prox
        return min(r_final, 1.0)  # limitat/in 0-1


# ================= 🛠️ auxiliaryfunction =================
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


def draw_risk_box(img, box, cls_name, risk, depth, speed):
    x1, y1, x2, y2 = map(int, box)

    # colorencoding：green -> orange -> red
    if risk > 0.7:
        color = (0, 0, 255)  # High Risk
        status = "DANGER"
    elif risk > 0.4:
        color = (0, 165, 255)  # Medium Risk
        status = "WARNING"
    else:
        color = (0, 255, 0)  # Low Risk
        status = "SAFE"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # labelinsidecontainer
    dist_str = f"{depth:.1f}m" if depth < 200 else "N/A"
    speed_str = f"{abs(speed):.1f}m/s" if abs(speed) > 0.1 else "Static"

    label_top = f"{cls_name} [{status}]"
    label_bot = f"R:{risk:.2f} D:{dist_str}"

    # drawlabelbackground
    font_scale = 0.5
    (tw, th), base = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    # abovelabel (category+status)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + 120, y1), color, -1)
    cv2.putText(img, label_top, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    # belowlabel (detaileddata) - draw inside boxbottom
    cv2.rectangle(img, (x1, y2 - th - 4), (x1 + 120, y2), color, -1)
    cv2.putText(img, label_bot, (x1, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)


# ================= 🚀 majorprogram =================
def main():
    device = setup_device()
    print("🚀 startriskevaluationvisualization (GNN + LiDAR)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. loadmappingfile (RGB Seq -> LiDAR Seq)
    if not os.path.exists(MAPPING_FILE):
        print(f"❌ missingmappingfile: {MAPPING_FILE}")
        return
    df_map = pd.read_csv(MAPPING_FILE, dtype=str)
    # establishmappingdictionary: rgbt_sequenceuence_id -> lidar_sequenceuence_id
    rgb2lidar = {row['rgbt_sequenceuence_id'].strip().zfill(3): row['lidar_sequenceuence_id'].strip().zfill(4) for _, row in
                 df_map.iterrows()}

    # 2. initializationmodule
    risk_calc = TrafficRiskCalculator()
    try:
        lidar_proc = LidarProcessor(CALIB_FILE)
        print("✅ LiDAR processingdeviceloadingsuccess")
    except Exception as e:
        print(f"❌ LiDAR initialization failed: {e}")
        return

    # 3. load GNN model
    list_path = os.path.join(project_root, DATASET_DIR, "stage2_test_list.json")
    with open(list_path, 'r') as f:
        samples = json.load(f)

    # autodetectdimension
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

    print(f"🚀 startprocessing {len(samples)} asample...")

    for info in tqdm(samples, desc="Risk Assessing"):
        try:
            # A. parsepathandID
            pt_path = info['path']
            if not os.path.isabs(pt_path): pt_path = os.path.join(project_root, pt_path)

            parts = info['id'].split('_')  # [sequenceuence_id, frame_id]
            rgb_sequenceuence = parts[0]
            frame_id = parts[1]

            # checkfindcorrespondingof LiDAR sequenceuenceuential
            if rgb_sequenceuence not in rgb2lidar:
                # print(f"⚠️ noLiDARmapping: {rgb_sequenceuence}")
                continue
            lidar_sequenceuence = rgb2lidar[rgb_sequenceuence]

            # buildfilepath
            img_path = os.path.join(ROOT_DATA_DIR, 'R-LiViT_RGB-T', 'rgb', rgb_sequenceuence, f"{frame_id}.png")
            # LiDAR filenameusuallyis 6digitcountcharacter (000015.bin)
            try:
                lidar_idx = int(frame_id)
                lidar_path = os.path.join(ROOT_DATA_DIR, 'R-LiViT_LiDAR', 'point_clouds', lidar_sequenceuence,
                                          f"{lidar_idx:06d}.bin")
            except:
                continue

            if not os.path.exists(img_path) or not os.path.exists(lidar_path):
                continue

            # B. loaddata
            img0 = cv2.imread(img_path)
            h0, w0 = img0.shape[:2]
            points_3d = lidar_proc.load_bin_file(lidar_path)  # loadpoint cloud

            # GNN inference
            data = torch.load(pt_path, map_location=device)
            feat = torch.nn.functional.normalize(torch.nan_to_num(data['features'].float(), 0.0).clamp(-100, 100), p=2,
                                                 dim=1)
            norm_xywh = data['boxes'].float()
            scores = data['scores'][:, 0]

            # C. coordinaterestore (Norm XYWH -> Pixel XYXY Original)
            pixel_xywh = norm_xywh * 1280
            pixel_xyxy = xywh2xyxy(pixel_xywh)
            orig_boxes_all = scale_coords((1280, 1280), pixel_xyxy.clone(), (h0, w0))

            # D. executeinference
            with torch.no_grad():
                out, keep, _ = model([feat], [norm_xywh], [scores])
                p_ex = torch.sigmoid(head_ex(out)).view(-1)
                p_cl = torch.argmax(head_cl(out), dim=1)

            # E. riskcalculationloop
            valid_dets = []
            if keep and len(keep) > 0 and len(keep[0]) > 0:
                keep_idxs = keep[0]

                # collectallboxperform NMS
                boxes_t, scores_t, cls_t = [], [], []
                raw_indices = []  # recordoriginalindexby/withconvenientfindfallback orig_boxes

                for k_i, idx in enumerate(keep_idxs):
                    if p_ex[k_i] > 0.4:  # GNN threshold
                        raw_idx = int(idx)
                        boxes_t.append(orig_boxes_all[raw_idx])
                        scores_t.append(p_ex[k_i])
                        cls_t.append(p_cl[k_i])
                        raw_indices.append(raw_idx)

                if boxes_t:
                    bt = torch.stack(boxes_t)
                    st = torch.stack(scores_t)
                    ct = torch.stack(cls_t)

                    # NMS deduplication
                    c_offset = ct.float() * 4096
                    keep_nms = torchvision.ops.nms(bt + c_offset.unsqueeze(1), st, 0.3)

                    # traversefinalresult
                    for ki in keep_nms:
                        final_box = bt[ki]
                        final_score = st[ki].item()
                        final_cls = ct[ki].item()

                        if final_cls not in CLASS_MAP: continue
                        cls_name = CLASS_MAP[final_cls]

                        # 🔥 core：LiDAR distance measurementandspeed measurement
                        # box format [x1, y1, x2, y2]
                        bbox_list = final_box.cpu().numpy().tolist()
                        depth, speed, _ = lidar_proc.get_object_depth_and_speed(points_3d, bbox_list)

                        # calculationrisk
                        risk = risk_calc.calculate(cls_name, final_score, depth, speed)

                        draw_risk_box(img0, final_box, cls_name, risk, depth, speed)

            # save
            save_name = f"{rgb_sequenceuence}_{frame_id}_risk.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), img0)

        except Exception as e:
            # print(f"Error {info['id']}: {e}")
            continue

    print(f"\n✅ riskevaluationvisualizationcompleted！resultat/in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()