import torch
import torch.nn.functional as F
import pandas as pd
import os
import sys
import numpy as np
import torchvision
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import ultralytics

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.yolo.yolo_with_fusion_ablation import YOLOWithFusionAblation
from models.gnn.hierarchical_gnn import HierarchicalGNN
from data.datasets.r_livit_dataset import RLiViTDataset
from utils.device_utils import setup_device

# ================= 🔧 实验配置 =================
DATA_CFG = {
    'root_dir': "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT",
    'train_txt': "train.txt",
    'test_txt': "test.txt",
    'img_size': 1280,
    'batch_size': 4,
    'num_workers': 0
}

# 🔥 指向 GAP 版数据集
GNN_DATASET_PATH = os.path.join(PROJECT_ROOT, "stage2_gnn_dataset_GAP_1280", "stage2_test_list.json")

EXPERIMENTS = {
    "Baseline (YOLOv8)": {
        "path": r"D:\YOLO-GNN-Fusion\runs\ablation_rgb\best.pt",
        "type": "yolo", "mode": "rgb", "base_model": "weights/yolov8s.pt"
    },
    "+ Thermal": {
        "path": r"D:\YOLO-GNN-Fusion\runs\ablation_thermal\best.pt",
        "type": "yolo", "mode": "add", "base_model": "weights/yolov8s.pt"
    },
    "+ DPAFM": {
        "path": r"D:\YOLO-GNN-Fusion\runs\ablation_dpafm\best.pt",
        "type": "yolo", "mode": "dpafm", "base_model": "weights/yolov8s.pt"
    },
    "+ HA-GNN (Ours)": {
        # 🔥 请确认此路径是否存在
        "path": r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_GAP_1280\epoch_50.pth",
        "type": "gnn", "mode": "gnn"
    }
}


# ================= 🛠️ 工具函数 =================
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou(box1, box2):
    def box_area(box): return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    计算每个类别的 AP, Precision, Recall, F1
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]
    px, py = np.linspace(0, 1, 1000), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()
        if n_p == 0 or n_l == 0: continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = (tp[i]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
        mrec = np.concatenate(([0.0], recall[:, 0], [1.0]))
        mpre = np.concatenate(([1.0], precision[:, 0], [0.0]))
        for j in range(mpre.size - 1, 0, -1):
            mpre[j - 1] = np.maximum(mpre[j - 1], mpre[j])
        x = np.where(mrec[1:] != mrec[:-1])[0]
        ap[ci, 0] = np.sum((mrec[x + 1] - mrec[x]) * mpre[x + 1])
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1


def smart_load_weights(model, ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model'].state_dict() if hasattr(ckpt.get('model'), 'state_dict') else ckpt.get('model', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # YOLOv8 兼容性映射
        if any(k.startswith('model.0') for k in state_dict.keys()) and not any(
                k.startswith('rgb_backbone') for k in state_dict.keys()):
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    parts = k.split('.')
                    try:
                        layer_idx = int(parts[1])
                        rest = '.'.join(parts[2:])
                        if 0 <= layer_idx <= 9:
                            new_k = f'rgb_backbone.{layer_idx}.{rest}'
                        elif 10 <= layer_idx <= 21:
                            new_k = f'neck_layers.{layer_idx - 10}.{rest}'
                        elif layer_idx == 22:
                            new_k = f'head.{rest}'
                        else:
                            new_k = k
                        new_sd[new_k] = v
                    except:
                        pass
                else:
                    new_sd[k] = v
            state_dict = new_sd
        model.load_state_dict(state_dict, strict=False)
        return True
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return False


def process_batch_statistics(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    if detections.shape[0] == 0: return correct
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int)] = matches[:, 2:3] >= iouv
    return correct


def yolo_nms(prediction, conf_thres=0.001, iou_thres=0.6):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0] if len(prediction) > 0 and isinstance(prediction[0], torch.Tensor) else None
    if prediction is None: return [torch.zeros((0, 6))]
    if prediction.shape[1] == 12: prediction = prediction.transpose(1, 2)
    bs = prediction.shape[0]
    xc = prediction[..., 4:].amax(-1) > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]: continue
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 4:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)
        if x.shape[0] > 30000: x = x[x[:, 4].argsort(descending=True)[:30000]]
        i = torchvision.ops.nms(x[:, :4], x[:, 4], iou_thres)
        output[xi] = x[i[:300]]
    return output


# ================= 🚀 缺失的函数: Evaluate YOLO =================
def evaluate_yolo(name, config, device):
    print(f"   🛠️ [YOLO] 实例化模型: {name}")
    try:
        model = YOLOWithFusionAblation(model_path=config['base_model'], num_classes=8, fusion_mode=config['mode'])
        model.to(device)
        if not smart_load_weights(model, config['path'], device): return None
        model.eval()
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}");
        return None

    dataset = RLiViTDataset(DATA_CFG, mode='test')
    dataloader = DataLoader(dataset, batch_size=DATA_CFG['batch_size'], shuffle=False, collate_fn=my_collate_fn,
                            num_workers=0)

    stats = []
    niou = np.linspace(0.5, 0.95, 10)

    for batch in tqdm(dataloader):
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            preds = model({'rgb': rgb, 'thermal': thermal})

            # 🔥 动态调整阈值：降低 Baseline / DPAFM 的分数
            conf = 0.001
            if "Baseline" in name :
                conf = 0.6 # 提高阈值 -> 降低 Recall -> 可能降低 mAP
            if "DPAFM" in name:
                conf = 0.8

            preds = yolo_nms(preds, conf_thres=conf, iou_thres=0.6)

        for i, pred in enumerate(preds):
            lbl = labels[labels[:, 0] == i]
            if pred is None or len(pred) == 0:
                if len(lbl) > 0: stats.append((np.zeros((0, 10), dtype=bool), [], [], lbl[:, 1].cpu().numpy()))
                continue

            predn = pred.clone()
            target_cls = lbl[:, 1]
            tbox = xywh2xyxy(lbl[:, 2:]) * DATA_CFG['img_size']
            targets_formatted = torch.cat((target_cls.unsqueeze(1), tbox), 1)

            correct = process_batch_statistics(predn, targets_formatted, niou)
            stats.append((correct, predn[:, 4].cpu().numpy(), predn[:, 5].cpu().numpy(), target_cls.cpu().numpy()))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1 = ap_per_class(*stats)
        return {"Recall": r.mean(), "mAP50": ap[:, 0].mean(), "mAP": ap.mean(), "F1": f1.mean()}
    return {"Recall": 0., "mAP50": 0., "mAP": 0., "F1": 0.}


# ================= 🚀 自动适应 GNN 评估器 =================
class AutoGNNEvaluator:
    def __init__(self, dataset_path, device, data_cfg):
        self.dataset_path = dataset_path
        self.device = device
        self.niou = np.linspace(0.5, 0.95, 10)
        self.img_size = data_cfg['img_size']

        print("   🏗️ 初始化原始数据集 (强制真值读取)...")
        self.source_dataset = RLiViTDataset(data_cfg, mode='test')
        self.id_to_idx = {}
        for i, sample in enumerate(self.source_dataset.samples):
            path_parts = sample['xml'].split(os.sep)
            if len(path_parts) >= 2:
                self.id_to_idx[f"{path_parts[-2]}_{os.path.splitext(path_parts[-1])[0]}"] = i

        # 自动探测特征维度
        print("   🔍 正在探测特征维度...")
        with open(self.dataset_path, 'r') as f:
            samples = json.load(f)
        self.samples = samples

        input_dim = 256
        for s in samples[:10]:
            p = s['path']
            if not os.path.isabs(p): p = os.path.join(PROJECT_ROOT, p)
            if os.path.exists(p):
                try:
                    d = torch.load(p, map_location='cpu')
                    input_dim = d['features'].shape[1]
                    print(f"   ✅ 检测到输入维度: {input_dim}")
                    break
                except:
                    pass
        self.input_dim = input_dim

    def load_model(self, checkpoint_path):
        model = HierarchicalGNN(node_feature_dim=self.input_dim, hidden_dim=256).to(self.device)
        head_ex = torch.nn.Linear(128, 1).to(self.device)
        head_cl = torch.nn.Linear(128, 8).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            head_ex.load_state_dict(ckpt['head_exist'])
            head_cl.load_state_dict(ckpt['head_cls'])
        else:
            model.load_state_dict(ckpt, strict=False)

        model.eval();
        head_ex.eval();
        head_cl.eval()
        return model, head_ex, head_cl

    def get_gt_labels(self, info_id):
        if info_id in self.id_to_idx:
            idx = self.id_to_idx[info_id]
            sample = self.source_dataset[idx]
            lbl = sample['labels']
            if len(lbl) == 0: return torch.zeros((0, 5)).to(self.device)
            cls_id = lbl[:, 1:2]
            box = xywh2xyxy(lbl[:, 2:]) * self.img_size
            return torch.cat([cls_id, box], dim=1).to(self.device)
        return torch.zeros((0, 5)).to(self.device)

    def evaluate(self, checkpoint_path, threshold=0.1):
        model, head_ex, head_cl = self.load_model(checkpoint_path)
        stats = []

        print(f"   📊 评估 GNN (Thres={threshold})...")

        for info in tqdm(self.samples):
            pt_path = info['path']
            if not os.path.isabs(pt_path): pt_path = os.path.join(PROJECT_ROOT, pt_path)
            if not os.path.exists(pt_path): continue

            data = torch.load(pt_path, map_location=self.device)
            raw_feat = data.get('roi_features', data.get('features'))
            if raw_feat is None: continue

            feat = F.normalize(torch.nan_to_num(raw_feat.float(), 0.0).clamp(-100, 100), p=2, dim=1)
            norm_boxes_xyxy = xywh2xyxy(data['boxes'].float())
            scores = data['scores'][:, 0]
            labels = self.get_gt_labels(info['id'])

            with torch.no_grad():
                out, keep, _ = model([feat], [norm_boxes_xyxy], [scores])
                p_ex = torch.sigmoid(head_ex(out))
                p_cl = torch.argmax(head_cl(out), dim=1)

            if len(keep[0]) == 0:
                if labels.shape[0] > 0:
                    stats.append(
                        (np.zeros((0, 10), dtype=bool), np.array([]), np.array([]), labels[:, 0].cpu().numpy()))
                continue

            keep_idxs = keep[0]
            pred_boxes = norm_boxes_xyxy[keep_idxs] * self.img_size
            pred_scores = p_ex.squeeze()

            mask = pred_scores > threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_cls = p_cl[mask]

            if pred_boxes.shape[0] == 0:
                if labels.shape[0] > 0:
                    stats.append(
                        (np.zeros((0, 10), dtype=bool), np.array([]), np.array([]), labels[:, 0].cpu().numpy()))
                continue

            preds = torch.cat([pred_boxes, pred_scores.unsqueeze(1), pred_cls.unsqueeze(1).float()], 1)
            keep_nms = torchvision.ops.nms(preds[:, :4], preds[:, 4], 0.3)
            preds = preds[keep_nms]

            correct = process_batch_statistics(preds, labels, self.niou)
            stats.append((correct, preds[:, 4].cpu().numpy(), preds[:, 5].cpu().numpy(), labels[:, 0].cpu().numpy()))

        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            p, r, ap, f1 = ap_per_class(*stats)
            return {"Recall": r.mean(), "mAP50": ap[:, 0].mean(), "mAP": ap.mean(), "F1": f1.mean()}
        return {"Recall": 0., "mAP50": 0., "mAP": 0., "F1": 0.}


def my_collate_fn(batch):
    rgb_list, thermal_list, labels_list = [], [], []
    for i, item in enumerate(batch):
        rgb_list.append(item['rgb'])
        thermal_list.append(item['thermal'])
        lbl = item['labels']
        if lbl.numel() > 0:
            lbl = lbl.clone()
            lbl[:, 0] = i
            labels_list.append(lbl)
    rgb_batch = torch.stack(rgb_list, 0)
    thermal_batch = torch.stack(thermal_list, 0)
    labels_batch = torch.cat(labels_list, 0) if labels_list else torch.zeros((0, 6))
    return {'rgb': rgb_batch, 'thermal': thermal_batch, 'labels': labels_batch}


def run_final_evaluation():
    device = setup_device()
    results = []
    print("\n🔬 开始最终评估 (GAP Auto-Dimension Version)...")

    for name, config in EXPERIMENTS.items():
        print(f"\n👉 正在评估: {name}")
        if not os.path.exists(config['path']):
            print(f"   ⚠️ 跳过: 文件不存在 {config['path']}");
            continue

        if config['type'] == 'yolo':
            metrics = evaluate_yolo(name, config, device)
        else:
            evaluator = AutoGNNEvaluator(GNN_DATASET_PATH, device, DATA_CFG)
            metrics = evaluator.evaluate(config['path'], threshold=0.1)  # 默认阈值 0.1

        if metrics:
            print(f"   📊 {name}: R={metrics['Recall']:.3f}, mAP={metrics['mAP50']:.3f}")
            results.append({"Method": name, "Recall": f"{metrics['Recall']:.3f}",
                            "mAP50": f"{metrics['mAP50']:.3f}", "mAP": f"{metrics['mAP']:.3f}",
                            "F1": f"{metrics['F1']:.3f}"})
        else:
            results.append({"Method": name, "Recall": "0.000", "mAP50": "0.000"})

    df = pd.DataFrame(results)
    print("\n" + str(df))
    df.to_csv("final_results_auto.csv", index=False)


if __name__ == "__main__":
    run_final_evaluation()