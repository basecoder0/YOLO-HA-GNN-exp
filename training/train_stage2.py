import os
import sys
import json
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from data.datasets.r_livit_dataset import RLiViTDataset
from utils.device_utils import setup_device


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    return inter_area / (area1[:, None] + area2 - inter_area + 1e-6)


class MatchingDataset(Dataset):
    def __init__(self, list_path, data_cfg, mode='train'):
        with open(list_path, 'r') as f:
            self.pt_samples = json.load(f)
        self.gt_dataset = RLiViTDataset(data_cfg, mode=mode)
        self.id_to_idx = {}
        for i, s in enumerate(self.gt_dataset.samples):
            p = s['xml'].split(os.sep)
            if len(p) >= 2: self.id_to_idx[f"{p[-2]}_{os.path.splitext(p[-1])[0]}"] = i

    def __len__(self):
        return len(self.pt_samples)

    def __getitem__(self, idx):
        info = self.pt_samples[idx]
        pt_path = info['path']
        if not os.path.isabs(pt_path): pt_path = os.path.join(project_root, pt_path)
        try:
            data = torch.load(pt_path, map_location='cpu')
            feat = data.get('roi_features', data.get('features')).float()
            # 🔥 归一化
            feat = F.normalize(torch.nan_to_num(feat, 0.0).clamp(-100, 100), p=2, dim=1)
            boxes = xywh2xyxy(data['boxes'].float())

            gt_labels = torch.zeros((0, 5))
            if info['id'] in self.id_to_idx:
                s = self.gt_dataset[self.id_to_idx[info['id']]]
                if len(s['labels']) > 0: gt_labels = torch.cat([s['labels'][:, 1:2], xywh2xyxy(s['labels'][:, 2:])], 1)

            target_exist = torch.zeros(len(boxes))
            target_cls = torch.full((len(boxes),), -1, dtype=torch.long)
            if len(gt_labels) > 0:
                ious = box_iou(boxes, gt_labels[:, 1:])
                max_iou, max_idx = ious.max(1)
                pos = max_iou > 0.5
                target_exist[pos] = 1.0
                target_cls[pos] = gt_labels[max_idx[pos], 0].long()
                target_exist[(max_iou > 0.3) & (max_iou <= 0.5)] = -1

            return {'node_features': feat, 'boxes': boxes, 'scores': data['scores'],
                    'target_exist': target_exist, 'target_cls': target_cls, 'valid': True}
        except:
            return {'valid': False}


def collate_fn(batch): return [b for b in batch if b['valid']]


class GNNTrainerAuto:
    def __init__(self):
        self.device = setup_device()

        # 🔥🔥🔥 修改这里：强制保存到 D:\YOLO-GNN-Fusion\runs 🔥🔥🔥
        # 也可以写成 self.save_dir = r"D:\YOLO-GNN-Fusion\runs"
        self.save_dir = os.path.join(project_root, "runs")

        # 1. 自动定位 GAP 数据集
        data_dir = os.path.join(project_root, "stage2_gnn_dataset_GAP_1280")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ 找不到GAP特征目录: {data_dir}\n请先运行 extract_features_GAP.py")

        print(f"📂 使用特征目录: {data_dir}")
        train_list = os.path.join(data_dir, "stage2_train_list.json")

        # 2. 检查特征维度 (Auto Detect)
        print("🔍 自动检测特征维度...")
        with open(train_list) as f:
            sample0 = json.load(f)[0]
        p0 = sample0['path']
        if not os.path.isabs(p0): p0 = os.path.join(project_root, p0)
        d0 = torch.load(p0, map_location='cpu')
        input_dim = d0['features'].shape[1]
        print(f"✅ 检测到输入维度: {input_dim}")

        data_cfg = {'root_dir': "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT", 'train_txt': "train.txt",
                    'test_txt': "test.txt", 'img_size': 1280}
        self.loader = DataLoader(MatchingDataset(train_list, data_cfg), batch_size=4, shuffle=True,
                                 collate_fn=collate_fn)

        # 3. 初始化模型 (Input Dim = Detected Dim)
        self.model = HierarchicalGNN(node_feature_dim=input_dim, hidden_dim=256, num_layers=3).to(self.device)
        self.head_exist = nn.Linear(128, 1).to(self.device)
        self.head_cls = nn.Linear(128, 8).to(self.device)

        self.opt = optim.AdamW(
            list(self.model.parameters()) + list(self.head_exist.parameters()) + list(self.head_cls.parameters()),
            lr=1e-4, weight_decay=1e-4)
        self.crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(self.device))
        self.crit_ce = nn.CrossEntropyLoss()

    def train(self):
        print(f"🚀 开始训练 (Auto Mode), 模型将保存至: {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.train()

        for epoch in range(50):  # 50 Epochs enough
            total_loss = 0
            pbar = tqdm(self.loader, desc=f"Ep {epoch + 1}")
            for batch in pbar:
                if not batch: continue
                feats = [b['node_features'].to(self.device) for b in batch]
                boxes = [b['boxes'].to(self.device) for b in batch]
                scores = [b['scores'][:, 0].to(self.device) for b in batch]
                tgt_ex = [b['target_exist'].to(self.device) for b in batch]
                tgt_cl = [b['target_cls'].to(self.device) for b in batch]

                self.opt.zero_grad()
                out, keep, _ = self.model(feats, boxes, scores)
                if out.size(0) == 0: continue

                p_ex, p_cl = self.head_exist(out), self.head_cls(out)

                # Match Targets
                flat_tgt_ex, flat_tgt_cl = [], []
                for i, idx in enumerate(keep):
                    if len(idx) > 0:
                        flat_tgt_ex.append(tgt_ex[i][idx])
                        flat_tgt_cl.append(tgt_cl[i][idx])

                if not flat_tgt_ex: continue
                flat_tgt_ex = torch.cat(flat_tgt_ex).view(-1, 1)
                flat_tgt_cl = torch.cat(flat_tgt_cl)

                # Loss
                mask = flat_tgt_ex != -1
                if mask.sum() == 0: continue
                l_ex = self.crit_bce(p_ex[mask], flat_tgt_ex[mask])

                pos = (flat_tgt_ex == 1).squeeze()
                l_cl = self.crit_ce(p_cl[pos], flat_tgt_cl[pos]) if pos.sum() > 0 else 0.0

                loss = l_ex + l_cl
                loss.backward()
                self.opt.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            if (epoch + 1) % 10 == 0:
                torch.save({'model': self.model.state_dict(), 'head_exist': self.head_exist.state_dict(),
                            'head_cls': self.head_cls.state_dict()}, f"{self.save_dir}/epoch_{epoch + 1}.pth")
                print(f"💾 Checkpoint saved: {self.save_dir}/epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    GNNTrainerAuto().train()