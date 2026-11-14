import torch
import torch.optim as optim
import torch.nn as nn
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# === 导入官方 Loss 和工具类 ===
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace

# 尝试导入 NMS (用于验证阶段，虽然本脚本主要跑训练，但保留接口)
try:
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    from ultralytics.utils.nms import non_max_suppression

# 添加项目路径
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn


def train():
    # 1. 加载配置
    config_path = 'configs/train/stage1.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Official Stage 1 Training on {device}...")

    # 2. 数据加载 (全量模式)
    print("📚 Loading Full Dataset...")
    train_dataset = RLiViTDataset(cfg['data'], mode='train')

    # === 🚨 正式训练：移除切片限制 ===
    # train_dataset.samples = train_dataset.samples[:16] <--- 已移除

    print(f"✅ Loaded {len(train_dataset)} training samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,  # 正式训练必须打乱！
        collate_fn=fusion_collate_fn,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    # 3. 模型初始化
    model = YOLOWithFusion(model_path='weights/yolov8s.pt', num_classes=cfg['model']['num_classes'])
    model.to(device)

    print("🔥 Model initialized. Backbone & Neck are unlocked.")

    # 4. 优化器 (使用 AdamW，因为之前的测试证明它收敛效果好)
    hyp = cfg['hyperparameters']
    # 稍微调低一点学习率，全量数据下 0.002 可能太激进，0.001 比较稳
    base_lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=hyp['weight_decay'])

    # 5. 学习率调度策略
    # 线性衰减：从 lr 到 lr * 0.01
    lf = lambda x: ((1 - x / hyp['epochs']) * (1.0 - hyp['lrf']) + hyp['lrf'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 6. Loss 配置
    # 之前的测试证明 box=10.0 有效，我们保留这个强化设置
    if 'box' not in hyp: hyp['box'] = 7.5
    if 'cls' not in hyp: hyp['cls'] = 0.5
    if 'dfl' not in hyp: hyp['dfl'] = 1.5

    model.args = IterableSimpleNamespace(**hyp)
    model.model = [model.head]

    # 再次确保 Head 参数正确
    if not hasattr(model.head, 'reg_max'):
        model.head.reg_max = 16

    criterion = v8DetectionLoss(model)

    # 7. 训练循环
    save_dir = f"runs/{cfg['experiment_name']}"
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'results.csv')

    # Warmup 参数
    for x in optimizer.param_groups:
        x['initial_lr'] = x['lr']

    # 3个 Epoch 的 Warmup
    nw = max(round(3 * len(train_loader)), 100)

    # 写入日志表头
    with open(log_file, 'w') as f:
        f.write('epoch,train/box_loss,train/cls_loss,train/dfl_loss,total_loss,lr\n')

    best_loss = float('inf')

    for epoch in range(hyp['epochs']):
        model.train()
        epoch_loss = 0
        loss_items_sum = torch.zeros(3, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hyp['epochs']}")
        for i, batch in enumerate(pbar):
            # Warmup logic
            ni = i + len(train_loader) * epoch
            if ni <= nw:
                xi = [0, nw]
                for j, x in enumerate(optimizer.param_groups):
                    # x['initial_lr'] * lf(epoch)
                    x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.8, hyp['momentum']])

            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)

            # 归一化急救 (保留以防万一)
            if rgb.max() > 1.0: rgb = rgb.float() / 255.0
            if thermal.max() > 1.0: thermal = thermal.float() / 255.0

            inputs = {'rgb': rgb, 'thermal': thermal}
            loss_batch = {
                'batch_idx': batch['batch_idx'].to(device),
                'cls': batch['cls'].to(device),
                'bboxes': batch['bboxes'].to(device)
            }

            # Forward
            preds = model(inputs)
            loss, loss_items = criterion(preds, loss_batch)
            loss = loss.sum()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            loss_items_sum += loss_items

            pbar.set_postfix({
                'box': f"{loss_items[0].item():.2f}",
                'cls': f"{loss_items[1].item():.2f}",
                'loss': f"{loss.item():.4f}"
            })

        # Epoch end
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_items = loss_items_sum / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # Logging
        with open(log_file, 'a') as f:
            f.write(
                f"{epoch + 1},{avg_items[0].item():.4f},{avg_items[1].item():.4f},{avg_items[2].item():.4f},{avg_loss:.4f},{current_lr:.6f}\n")

        # Saving
        torch.save(model.state_dict(), os.path.join(save_dir, 'last.pt'))

        # 简单保存策略：loss 创新低就保存 best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            print(f"💾 New best model saved (Loss: {best_loss:.4f})")


if __name__ == '__main__':
    train()