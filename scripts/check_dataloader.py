import torch
import yaml
import cv2
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn


def check_data():
    # loadconfiguration
    with open('configs/train/stage1.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # initializationdataset
    dataset = RLiViTDataset(cfg['data'], mode='train')
    loader = DataLoader(dataset, batch_size=4, collate_fn=fusion_collate_fn, shuffle=True)

    print(f"🔍 check DataLoader output...")

    # getonea Batch
    batch = next(iter(loader))

    rgbs = batch['rgb']  # [B, 3, H, W]
    thermals = batch['thermal']
    bboxes = batch['bboxes']  # [N, 4] (cx, cy, w, h) normalized
    batch_idx = batch['batch_idx']
    cls = batch['cls']

    save_dir = 'runs/dataloader_check'
    os.makedirs(save_dir, exist_ok=True)

    # visualizationfirst 4 tensorimage
    for i in range(min(4, rgbs.shape[0])):
        # 1. restoreimage (Tensor -> Numpy -> BGR)
        img = rgbs[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # restorefor/as BGR forshow

        h, w = img.shape[:2]

        # 2. findtoattributeforthistensorimageof label
        mask = (batch_idx == i)
        curr_bboxes = bboxes[mask]
        curr_cls = cls[mask]

        # 3. draw bounding box
        for box, c in zip(curr_bboxes, curr_cls):
            cx, cy, bw, bh = box.tolist()
            # 反normalize
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, str(int(c.item())), (x1, y1 - 5), 0, 0.6, (0, 0, 255), 2)

        cv2.imwrite(f"{save_dir}/sample_{i}.jpg", img)
        print(f"   alreadysavesample: {save_dir}/sample_{i}.jpg")

    print("✅ check完毕。pleaseopenimagepiece，confirmboxwhether准sure套at/inobjectbodyabove？colorwhetherpositiveoften？")


if __name__ == '__main__':
    check_data()