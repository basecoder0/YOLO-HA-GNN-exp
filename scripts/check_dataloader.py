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
    # 加载配置
    with open('configs/train/stage1.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 初始化数据集
    dataset = RLiViTDataset(cfg['data'], mode='train')
    loader = DataLoader(dataset, batch_size=4, collate_fn=fusion_collate_fn, shuffle=True)

    print(f"🔍 检查 DataLoader 输出...")

    # 取一个 Batch
    batch = next(iter(loader))

    rgbs = batch['rgb']  # [B, 3, H, W]
    thermals = batch['thermal']
    bboxes = batch['bboxes']  # [N, 4] (cx, cy, w, h) normalized
    batch_idx = batch['batch_idx']
    cls = batch['cls']

    save_dir = 'runs/dataloader_check'
    os.makedirs(save_dir, exist_ok=True)

    # 可视化前 4 张图
    for i in range(min(4, rgbs.shape[0])):
        # 1. 还原图像 (Tensor -> Numpy -> BGR)
        img = rgbs[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 还原为 BGR 用于显示

        h, w = img.shape[:2]

        # 2. 找到属于这张图的 label
        mask = (batch_idx == i)
        curr_bboxes = bboxes[mask]
        curr_cls = cls[mask]

        # 3. 画框
        for box, c in zip(curr_bboxes, curr_cls):
            cx, cy, bw, bh = box.tolist()
            # 反归一化
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, str(int(c.item())), (x1, y1 - 5), 0, 0.6, (0, 0, 255), 2)

        cv2.imwrite(f"{save_dir}/sample_{i}.jpg", img)
        print(f"   已保存样本: {save_dir}/sample_{i}.jpg")

    print("✅ 检查完毕。请打开图片，确认框是否准确套在物体上？颜色是否正常？")


if __name__ == '__main__':
    check_data()