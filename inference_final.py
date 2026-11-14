import os
import sys
import torch
import cv2
import json
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm

# 自动定位项目根目录
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from utils.device_utils import setup_device

# 🔥 导入 LiDAR 工具
try:
    from utils.lidar_utils import LidarProcessor
except ImportError:
    print("❌ 错误: 找不到 utils.lidar_utils，请确保该文件存在！")
    sys.exit(1)

# ================= 🔧 核心阈值配置区 (在这里手动修改) =================

# 1. 检测置信度阈值 (GNN Confidence Threshold)
# 作用: 只有置信度大于此值的物体才会被画出来并计算风险。
# 调低 (如 0.2): 检出更多物体，防止漏检，但可能增加误检。
# 调高 (如 0.6): 只保留非常确定的物体，画面更干净，但可能漏检。
CONF_THRES = 0.40

# 2. 风险等级阈值 (Risk Level Thresholds)
# 作用: 决定框的颜色 (红/橙/绿)。
# 逻辑: 风险值 > HIGH -> 红色 (危险)
#       风险值 > MID  -> 橙色 (警告)
#       其他          -> 绿色 (安全)
RISK_THRES_HIGH = 0.70  # 高风险阈值
RISK_THRES_MID = 0.40  # 中风险阈值

# 3. NMS 去重阈值 (IoU Threshold)
# 作用: 如果两个框重叠度(IoU)超过此值，只保留分高的那个。
# 通常保持 0.3 或 0.45 即可，不需要经常改。
NMS_IOU_THRES = 0.30

# ================= 📂 路径配置区 =================
DATASET_DIR = "stage2_gnn_dataset_GAP_1280"
CHECKPOINT = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_GAP_1280\epoch_50.pth"
OUTPUT_DIR = os.path.join(project_root, "comparison_vis", "risk_assessment_final")

ROOT_DATA_DIR = r"D:\BaiduNetdiskDownload\R-LiViT\R-LiViT"
MAPPING_FILE = os.path.join(ROOT_DATA_DIR, "rgb_to_lidar_sequence_mapping.csv")
CALIB_FILE = os.path.join(ROOT_DATA_DIR, "calibration_params.json")

# 类别映射
CLASS_MAP = {
    0: 'Car', 1: 'Person', 2: 'Bicycle', 3: 'Motorcycle',
    4: 'Bus', 5: 'Truck', 6: 'Tramway', 7: 'Escooter'
}


# ================= 🧠 风险计算器 (TrafficRiskCalculator) =================
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

        # 1. 基础风险
        r_base = p['w'] * conf
        # 2. 运动风险
        r_motion = 1.0 + p['alpha'] * abs(speed)
        # 3. 距离风险
        if depth is None or np.isnan(depth) or depth <= 0: depth = 999.0
        dist_diff = p['d_safe'] - depth
        r_prox = self.sigmoid(p['beta'] * dist_diff)

        r_final = r_base * r_motion * r_prox
        return min(r_final, 1.0)


# ================= 🛠️ 工具函数 =================
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

    # 🔥 使用配置的风险阈值
    if risk > RISK_THRES_HIGH:
        color = (0, 0, 255)  # Red (High Risk)
        text_color = (255, 255, 255)
    elif risk > RISK_THRES_MID:
        color = (0, 165, 255)  # Orange (Medium Risk)
        text_color = (0, 0, 0)
    else:
        color = (0, 255, 0)  # Green (Safe)
        text_color = (0, 0, 0)

    # 画框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 标签内容
    cls_name = CLASS_MAP.get(cls_id, 'Unk')
    label = f"{cls_name}: {score:.2f} R: {risk:.2f}"

    # 标签背景
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


# ================= 🚀 主程序 =================
def main():
    device = setup_device()
    print(f"🚀 启动风险评估可视化...")
    print(f"   - 检测阈值 (Conf): {CONF_THRES}")
    print(f"   - 高风险阈值 (Red): {RISK_THRES_HIGH}")
    print(f"   - 中风险阈值 (Orange): {RISK_THRES_MID}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 映射文件
    if not os.path.exists(MAPPING_FILE):
        print(f"❌ 找不到: {MAPPING_FILE}")
        return
    df_map = pd.read_csv(MAPPING_FILE, dtype=str)
    rgb2lidar = {row['rgbt_seq_id'].strip().zfill(3): row['lidar_seq_id'].strip().zfill(4) for _, row in
                 df_map.iterrows()}

    # 2. 初始化
    risk_calc = TrafficRiskCalculator()
    try:
        lidar_proc = LidarProcessor(CALIB_FILE)
        print("✅ LiDAR 模块就绪")
    except Exception as e:
        print(f"❌ LiDAR 初始化失败: {e}")
        return

    # 3. 模型加载
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

    print(f"🚀 开始处理 {len(samples)} 张图片...")

    for info in tqdm(samples, desc="Processing"):
        try:
            pt_path = info['path']
            if not os.path.isabs(pt_path): pt_path = os.path.join(project_root, pt_path)

            parts = info['id'].split('_')
            rgb_seq = parts[0]
            frame_id = parts[1]

            if rgb_seq not in rgb2lidar: continue
            lidar_seq = rgb2lidar[rgb_seq]

            img_path = os.path.join(ROOT_DATA_DIR, 'R-LiViT_RGB-T', 'rgb', rgb_seq, f"{frame_id}.png")
            try:
                l_idx = int(frame_id)
                lidar_path = os.path.join(ROOT_DATA_DIR, 'R-LiViT_LiDAR', 'point_clouds', lidar_seq, f"{l_idx:06d}.bin")
            except:
                continue

            if not os.path.exists(img_path) or not os.path.exists(lidar_path): continue

            img0 = cv2.imread(img_path)
            h0, w0 = img0.shape[:2]
            points_3d = lidar_proc.load_bin_file(lidar_path)

            # GNN 推理
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
                    # 🔥 使用配置的检测阈值
                    if p_ex[k_i] > CONF_THRES:
                        raw_idx = int(idx)
                        boxes_t.append(orig_boxes_all[raw_idx])
                        scores_t.append(p_ex[k_i])
                        cls_t.append(p_cl[k_i])

                if boxes_t:
                    bt = torch.stack(boxes_t)
                    st = torch.stack(scores_t)
                    ct = torch.stack(cls_t)

                    # 🔥 使用配置的 NMS 阈值
                    c_offset = ct.float() * 4096
                    keep_nms = torchvision.ops.nms(bt + c_offset.unsqueeze(1), st, NMS_IOU_THRES)

                    for ki in keep_nms:
                        final_box = bt[ki]
                        final_score = st[ki].item()
                        final_cls = ct[ki].item()

                        if final_cls not in CLASS_MAP: continue

                        # 1. 物理测距
                        bbox_list = final_box.cpu().numpy().tolist()
                        depth, speed, _ = lidar_proc.get_object_depth_and_speed(points_3d, bbox_list)

                        # 2. 风险计算
                        risk = risk_calc.calculate(final_cls, final_score, depth, speed)

                        # 3. 绘图
                        draw_risk_box(img0, final_box, final_cls, final_score, risk)

            save_name = f"{rgb_seq}_{frame_id}_risk.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), img0)

        except Exception as e:
            continue

    print(f"\n✅ 风险图生成完毕: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()