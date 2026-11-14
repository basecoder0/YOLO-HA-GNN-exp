import torch
import numpy as np
import os
import json
import sys
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录
sys.path.append(str(Path(__file__).parent.parent))

from models.gnn.hierarchical_gnn import HierarchicalGNN
from utils.metrics.evaluation_metrics import box_iou, ap_per_class
from utils.device_utils import setup_device


class Evaluator:
    def __init__(self, dataset_path, device=None):
        self.dataset_path = dataset_path
        self.device = device if device else setup_device()
        self.iou_thres = 0.5  # NMS IoU threshold
        self.conf_thres = 0.001  # Evaluation confidence threshold
        self.niou = np.linspace(0.5, 0.95, 10)  # mAP@0.5:0.95 IoU vector

    def load_model(self, checkpoint_path):
        """加载 GNN 模型"""
        model = HierarchicalGNN(node_feature_dim=256, hidden_dim=256).to(self.device)
        head_ex = torch.nn.Linear(128, 1).to(self.device)
        head_cl = torch.nn.Linear(128, 8).to(self.device)  # 假设8类

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            head_ex.load_state_dict(ckpt['head_exist'])
            head_cl.load_state_dict(ckpt['head_cls'])
        else:
            model.load_state_dict(ckpt, strict=False)

        model.eval()
        return model, head_ex, head_cl

    def process_batch(self, detections, labels, iouv):
        """
        匹配预测框和真值框
        Return: [Correct, Conf, Pred_Cls, Target_Cls]
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU > 0.5 & class match

        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int)] = matches[:, 2:3] >= iouv
        return correct

    def evaluate(self, checkpoint_path):
        """运行完整评估"""
        model, head_ex, head_cl = self.load_model(checkpoint_path)

        # 加载测试列表
        with open(self.dataset_path, 'r') as f:
            samples = json.load(f)

        stats = []  # [TP, Conf, Pred_Cls, Target_Cls]

        print(f"🚀 开始评估: {os.path.basename(checkpoint_path)}")
        for info in tqdm(samples):
            # 1. 加载数据 (特征 & 真值)
            # 假设 .pt 文件包含了 'features', 'boxes', 'labels' (真值)
            # 如果 .pt 是 Stage 1 的输出，你需要另外加载 label
            pt_path = info['path']
            if not os.path.exists(pt_path): continue

            data = torch.load(pt_path, map_location=self.device)
            feat = data['features'].float()
            boxes = data['boxes'].float()  # Norm boxes [cx, cy, w, h] or [x1, y1, x2, y2]
            scores = data['scores'][:, 0]

            # 假设 labels 也在 data 中，或者你需要从 info['label_path'] 加载
            # 这里为了演示，假设 data['gt_labels'] 存在: [class, x1, y1, x2, y2]
            if 'gt_labels' not in data or len(data['gt_labels']) == 0:
                continue

            labels = data['gt_labels'].to(self.device)

            # 2. 模型推理
            with torch.no_grad():
                out, keep, _ = model([feat], [boxes], [scores])
                p_ex = torch.sigmoid(head_ex(out))
                p_cl = torch.argmax(head_cl(out), dim=1) + 1  # 假设 1-8 是类别

            # 3. 收集预测结果
            keep_idxs = keep[0]
            pred_boxes = boxes[keep_idxs]  # 使用原始框或回归后的框
            pred_scores = p_ex.squeeze()
            pred_cls = p_cl

            # 过滤低置信度
            mask = pred_scores > self.conf_thres
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_cls = pred_cls[mask]

            # 4. 统计匹配 (Match)
            n_pred = pred_boxes.shape[0]
            n_gt = labels.shape[0]

            if n_pred == 0:
                if n_gt > 0:
                    stats.append((np.zeros((0, self.niou.shape[0]), dtype=bool), np.array([]), np.array([]),
                                  labels[:, 0].cpu().numpy()))
                continue

            # 构建 pred 矩阵: [x1, y1, x2, y2, conf, cls]
            # 这里的 boxes 假设已经是 x1y1x2y2 格式，如果不是需要转换
            preds = torch.cat([pred_boxes, pred_scores.unsqueeze(1), pred_cls.unsqueeze(1).float()], 1)

            correct = self.process_batch(preds, labels, torch.tensor(self.niou, device=self.device))
            stats.append((correct, preds[:, 4].cpu().numpy(), preds[:, 5].cpu().numpy(), labels[:, 0].cpu().numpy()))

        # 5. 计算全局指标
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, save_dir='.')
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            mf1 = f1.mean()
        else:
            mp, mr, map50, map, mf1 = 0., 0., 0., 0., 0.

        return {
            "Precision": mp,
            "Recall": mr,
            "mAP50": map50,
            "mAP": map,
            "F1": mf1
        }