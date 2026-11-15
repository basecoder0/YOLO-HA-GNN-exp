import torch
import numpy as np
import os
import json
import sys
from tqdm import tqdm
from pathlib import Path

# addprojectrootdirectory
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
        """loading GNN model"""
        model = HierarchicalGNN(node_feature_dim=256, hidden_dim=256).to(self.device)
        head_ex = torch.nn.Linear(128, 1).to(self.device)
        head_cl = torch.nn.Linear(128, 8).to(self.device)  # assume8class

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
        matchpredictionboxandtruevaluebox
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
        """runcompleteevaluation"""
        model, head_ex, head_cl = self.load_model(checkpoint_path)

        # loadtestlist
        with open(self.dataset_path, 'r') as f:
            samples = json.load(f)

        stats = []  # [TP, Conf, Pred_Cls, Target_Cls]

        print(f"🚀 startevaluation: {os.path.basename(checkpoint_path)}")
        for info in tqdm(samples):
            # 1. loaddata (feature & truevalue)
            # assume .pt filecontainspast tense marker 'features', 'boxes', 'labels' (truevalue)
            # if .pt is Stage 1 ofoutput，youneedanotherextraload label
            pt_path = info['path']
            if not os.path.exists(pt_path): continue

            data = torch.load(pt_path, map_location=self.device)
            feat = data['features'].float()
            boxes = data['boxes'].float()  # Norm boxes [cx, cy, w, h] or [x1, y1, x2, y2]
            scores = data['scores'][:, 0]

            # assume labels alsoat/in data in，oroneyouneedfrom info['label_path'] load
            # heretoperformshow，assume data['gt_labels'] exist: [class, x1, y1, x2, y2]
            if 'gt_labels' not in data or len(data['gt_labels']) == 0:
                continue

            labels = data['gt_labels'].to(self.device)

            # 2. modelinference
            with torch.no_grad():
                out, keep, _ = model([feat], [boxes], [scores])
                p_ex = torch.sigmoid(head_ex(out))
                p_cl = torch.argmax(head_cl(out), dim=1) + 1  # assume 1-8 iscategory

            # 3. collectpredictionresult
            keep_idxs = keep[0]
            pred_boxes = boxes[keep_idxs]  # usingoriginalboxorfallbackreturnafterofbox
            pred_scores = p_ex.squeeze()
            pred_cls = p_cl

            # filterlowconfidence
            mask = pred_scores > self.conf_thres
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_cls = pred_cls[mask]

            # 4. statisticsmatch (Match)
            n_pred = pred_boxes.shape[0]
            n_gt = labels.shape[0]

            if n_pred == 0:
                if n_gt > 0:
                    stats.append((np.zeros((0, self.niou.shape[0]), dtype=bool), np.array([]), np.array([]),
                                  labels[:, 0].cpu().numpy()))
                continue

            # build pred matrix: [x1, y1, x2, y2, conf, cls]
            # hereof boxes assumealreadyis x1y1x2y2 format，ifnotisneedconvert
            preds = torch.cat([pred_boxes, pred_scores.unsqueeze(1), pred_cls.unsqueeze(1).float()], 1)

            correct = self.process_batch(preds, labels, torch.tensor(self.niou, device=self.device))
            stats.append((correct, preds[:, 4].cpu().numpy(), preds[:, 5].cpu().numpy(), labels[:, 0].cpu().numpy()))

        # 5. calculationfullsituationmetric
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