# testing/inference_final.py
import torch
import cv2
import numpy as np
import os
import sys
import yaml
import pandas as pd
import warnings
import torchvision
from tqdm import tqdm  # 进度条库

# === 1. 环境设置 ===
sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

from models.risk_assessment.risk_calculator import RiskCalculator
from utils.lidar_utils import LidarProcessor
from models.gnn.hierarchical_gnn import HierarchicalGNN
from ultralytics.utils.ops import xywh2xyxy

# 🔥 导入 YOLO 类
try:
    from models.yolo.yolo_with_fusion import YOLOWithFusion
except ImportError:
    print("❌ 错误: 无法导入 YOLOWithFusion。请确认 models/yolo/yolo_with_fusion.py 文件存在。")
    sys.exit(1)


# ==============================================================================
# 🧠 风险计算器 (内置交通场景参数)
# ==============================================================================
class TrafficRiskCalculator:
    def __init__(self):
        # 类别映射
        self.idx2name = {
            0: 'car', 1: 'person', 2: 'bicycle', 3: 'motorcycle',
            4: 'bus', 5: 'truck', 6: 'tramway', 7: 'escooter'
        }

        # 🚦 核心参数设计
        # w: 类别基础权重, d_safe: 安全距离阈值, alpha: 速度敏感度, beta: 距离敏感度
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
        cls_name = self.idx2name.get(int(cls_id), 'car')
        p = self.params.get(cls_name, self.default_param)

        # 1. 基础风险
        r_base = p['w'] * conf

        # 2. 运动风险
        r_motion = 1.0 + p['alpha'] * abs(speed)

        # 3. 距离风险
        # 如果 depth = 999 (未检测到), 风险项会趋近于 0
        dist_diff = p['d_safe'] - depth
        r_prox = self.sigmoid(p['beta'] * dist_diff)

        r_final = r_base * r_motion * r_prox

        return r_final, cls_name


# ==============================================================================
# 🛠️ 辅助函数
# ==============================================================================
def simple_nms_final(prediction, conf_thres=0.1, iou_thres=0.5):
    """NMS 后处理 (适配 1280 分辨率)"""
    if prediction.shape[1] == 4 + 8:
        prediction = prediction.permute(0, 2, 1)

    bs = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        box = x[:, :4]
        cls_score = x[:, 4:]
        conf, j = cls_score.max(1, keepdim=True)
        mask = conf.view(-1) > conf_thres
        x_filtered = torch.cat((box, conf, j.float()), 1)[mask]

        if x_filtered.shape[0] == 0: continue

        box_xywh = x_filtered[:, :4]
        box_xyxy = xywh2xyxy(box_xywh)

        c = x_filtered[:, 5:6] * 7680
        boxes, scores = box_xyxy + c, x_filtered[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = torch.cat((box_xyxy[i], x_filtered[i, 4:]), 1)

    return output


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto: dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


# ==============================================================================
# 🚀 全系统推理类
# ==============================================================================
class FullSystemInference:
    def __init__(self, stage1_weights, stage2_weights, base_model_path, calib_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 初始化系统 (Device: {self.device})")

        # 1. 加载 Stage 1
        print(f"📦 加载 Stage 1: {stage1_weights}")
        self.model_s1 = YOLOWithFusion(model_path=base_model_path, num_classes=8)
        state_dict = torch.load(stage1_weights, map_location=self.device)
        self.model_s1.load_state_dict(state_dict)
        self.model_s1.to(self.device).eval().float()

        # 2. 加载 Stage 2
        print(f"📦 加载 Stage 2 权重...")
        try:
            self.gnn_model = HierarchicalGNN(node_feature_dim=256, hidden_dim=256).to(self.device)
            # 暂时不启用 GNN 推理，仅确保环境和权重加载无误
            ckpt = torch.load(stage2_weights, map_location=self.device)
            self.gnn_model.eval()
        except Exception as e:
            print(f"⚠️ Stage 2 加载警告: {e}")

        # 3. 工具
        self.risk_calc = TrafficRiskCalculator()
        self.lidar_proc = LidarProcessor(calib_file)

    def run_inference(self, rgb_path, thermal_path, lidar_path):
        rgb0 = cv2.imread(rgb_path)
        thm0 = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        points_3d = self.lidar_proc.load_bin_file(lidar_path)

        if rgb0 is None: return None

        # 1. 预处理
        img_size = 1280
        img_rgb, ratio, (dw, dh) = letterbox(rgb0, img_size, auto=False)
        t_rgb = torch.from_numpy(img_rgb.transpose((2, 0, 1))[::-1].copy()).to(self.device).float() / 255.0

        img_thm, _, _ = letterbox(thm0, img_size, auto=False)
        if len(img_thm.shape) == 2: img_thm = np.stack([img_thm] * 3, axis=-1)
        t_thm = torch.from_numpy(img_thm.transpose((2, 0, 1)).copy()).to(self.device).float() / 255.0

        if t_rgb.ndimension() == 3: t_rgb = t_rgb.unsqueeze(0)
        if t_thm.ndimension() == 3: t_thm = t_thm.unsqueeze(0)

        # 2. Stage 1 推理
        detections = []
        with torch.no_grad():
            preds = self.model_s1({'rgb': t_rgb, 'thermal': t_thm})
            if isinstance(preds, tuple): preds = preds[0]

            final_preds = simple_nms_final(preds, conf_thres=0.25, iou_thres=0.45)

            det = final_preds[0]
            if len(det) > 0:
                det[:, [0, 2]] -= dw
                det[:, [1, 3]] -= dh
                det[:, :4] /= ratio[0]

                for *xyxy, conf, cls_id in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([int(cls_id), float(conf), x1, y1, x2, y2])

        # 3. 风险计算与绘图
        results_img = rgb0.copy()

        for det in detections:
            cls_id, conf, x1, y1, x2, y2 = det
            bbox = [x1, y1, x2, y2]

            # 物理测距
            depth, speed, _ = self.lidar_proc.get_object_depth_and_speed(points_3d, bbox)

            # 计算分数
            risk_score, cls_name = self.risk_calc.calculate(cls_id, conf, depth, speed)

            # 绘图逻辑
            color = (0, 255, 0)  # Green
            if risk_score > 0.3: color = (0, 165, 255)  # Orange
            if risk_score > 0.7: color = (0, 0, 255)  # Red

            h, w = results_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cv2.rectangle(results_img, (x1, y1), (x2, y2), color, 2)

            dist_str = f"{depth:.1f}m" if depth < 900 else "NaN"
            label = f"{cls_name} {dist_str} R:{risk_score:.2f}"

            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(results_img, label, (x1, text_y), 0, 0.6, color, 2)

        return results_img


# ==============================================================================
# 🚀 主程序
# ==============================================================================
if __name__ == "__main__":
    # 1. 路径配置
    ROOT = r"D:\BaiduNetdiskDownload\R-LiViT\R-LiViT"

    BASE_MODEL_PT = r"D:\YOLO-GNN-Fusion\weights\yolov8s.pt"
    S1_WEIGHTS = r"D:\YOLO-GNN-Fusion\runs\stage1_rlivit_highres_8cls\best.pt"
    S2_WEIGHTS = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_1280\epoch_100.pth"

    MAPPING_FILE = os.path.join(ROOT, "rgb_to_lidar_sequence_mapping.csv")
    CALIB_FILE = os.path.join(ROOT, "calibration_params.json")

    OUTPUT_DIR = r"D:\YOLO-GNN-Fusion\inference_stage3_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 初始化
    system = FullSystemInference(S1_WEIGHTS, S2_WEIGHTS, BASE_MODEL_PT, CALIB_FILE)

    if os.path.exists(MAPPING_FILE):
        df = pd.read_csv(MAPPING_FILE, dtype=str)
        total_seqs = len(df)
        print(f"📄 共发现 {total_seqs} 个序列，准备开始全量推理...")
        print(f"💾 结果将保存至: {OUTPUT_DIR}")

        # 3. 进度条遍历
        # 移除了 .head(5)，使用 tqdm 显示进度
        for index, row in tqdm(df.iterrows(), total=total_seqs, desc="Processing"):
            try:
                rgb_seq_id = str(row['rgbt_seq_id']).strip().zfill(3)
                lidar_seq_id = str(row['lidar_seq_id']).strip().zfill(4)

                # 创建序列对应的输出子文件夹
                seq_output_dir = os.path.join(OUTPUT_DIR, rgb_seq_id)
                os.makedirs(seq_output_dir, exist_ok=True)

                raw_imgs = str(row['rgb_imgs'])
                img_list_str = raw_imgs.replace('[', '').replace(']', '').replace("'", "").split(',')
                img_ids = [x.strip() for x in img_list_str if x.strip()]

                for img_id in img_ids:
                    frame_name = img_id.zfill(2)
                    rgb_path = os.path.join(ROOT, 'R-LiViT_RGB-T', 'rgb', rgb_seq_id, f"{frame_name}.png")
                    thermal_path = os.path.join(ROOT, 'R-LiViT_RGB-T', 'thermal', rgb_seq_id, f"{frame_name}.png")

                    try:
                        l_idx = int(frame_name)
                    except:
                        l_idx = 0
                    lidar_path = os.path.join(ROOT, 'R-LiViT_LiDAR', 'point_clouds', lidar_seq_id, f"{l_idx:06d}.bin")

                    # 容错：如果找不到png找jpg
                    if not os.path.exists(rgb_path): rgb_path = rgb_path.replace('.png', '.jpg')

                    # 检查文件完整性
                    if not (os.path.exists(rgb_path) and os.path.exists(thermal_path) and os.path.exists(lidar_path)):
                        continue

                    # 执行推理
                    res_img = system.run_inference(rgb_path, thermal_path, lidar_path)

                    if res_img is not None:
                        save_p = os.path.join(seq_output_dir, f"{frame_name}.jpg")
                        cv2.imwrite(save_p, res_img)

            except Exception as e:
                # 打印错误但不中断循环
                # print(f"❌ Seq {rgb_seq_id} Error: {e}")
                continue

    print(f"\n🎉 全部完成！请检查文件夹: {OUTPUT_DIR}")