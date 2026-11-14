# scripts/train_ablation.py
import argparse
import yaml
import torch
import torch.optim as optim
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolo.yolo_with_fusion_ablation import YOLOWithFusionAblation
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn


def train_ablation(task_name, fusion_mode):
    # 1. 基础配置
    config_path = 'configs/train/stage1.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 自动设置实验名称
    exp_name = f"ablation_{task_name}"
    cfg['experiment_name'] = exp_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n" + "=" * 50)
    print(f"🚀 启动消融实验: {task_name.upper()}")
    print(f"🔩 融合模式 (Mode): {fusion_mode}")
    print(f"📂 结果保存路径: runs/{exp_name}")
    print(f"=" * 50 + "\n")

    # 2. 数据加载
    print("📚 正在加载数据集...")
    train_dataset = RLiViTDataset(cfg['data'], mode='train')

    # ⚡ 调试开关：如果想快速测试代码，取消下面注释 (跑前100张图)
    # train_dataset.samples = train_dataset.samples[:100]

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        collate_fn=fusion_collate_fn,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    # 3. 初始化模型
    print(f"🏗️ 初始化模型 (Fusion Mode: {fusion_mode})...")
    model = YOLOWithFusionAblation(
        model_path='weights/yolov8s.pt',
        num_classes=cfg['model']['num_classes'],
        fusion_mode=fusion_mode
    )
    model.to(device)

    # 4. 优化器配置
    hyp = cfg['hyperparameters']
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=hyp['weight_decay'])
    lf = lambda x: ((1 - x / hyp['epochs']) * (1.0 - hyp['lrf']) + hyp['lrf'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 5. Loss 配置 (包含核心修复)
    model.args = IterableSimpleNamespace(**hyp)

    # === 🔥 [核心修复] 手动指定 model.model 以修复 v8DetectionLoss 报错 ===
    model.model = [model.head]
    # ===================================================================

    if not hasattr(model.head, 'reg_max'): model.head.reg_max = 16
    criterion = v8DetectionLoss(model)

    # 6. 训练循环
    save_dir = f"runs/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    # 消融实验建议跑 30-50 Epoch 即可看清趋势
    epochs = 30

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # 进度条
        pbar = tqdm(train_loader, desc=f"[{task_name}] Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)

            inputs = {'rgb': rgb, 'thermal': thermal}
            loss_batch = {
                'batch_idx': batch['batch_idx'].to(device),
                'cls': batch['cls'].to(device),
                'bboxes': batch['bboxes'].to(device)
            }

            # 前向传播
            preds = model(inputs)

            # 计算 Loss
            loss, loss_items = criterion(preds, loss_batch)
            loss = loss.sum()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 学习率调整
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))

    print(f"\n✅ 实验 {task_name} 完成！")
    print(f"💾 最佳权重已保存至: {save_dir}/best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Ablation Studies for YOLO-GNN-Fusion')
    parser.add_argument('--task', type=str, required=True,
                        choices=['rgb', 'thermal', 'dpafm'],
                        help='选择实验任务: rgb (Baseline), thermal (+Thermal), dpafm (+DPAFM)')

    args = parser.parse_args()

    # 映射任务名到模型模式
    mode_map = {
        'rgb': 'rgb',  # Baseline: 仅 RGB
        'thermal': 'add',  # + Thermal: 简单相加
        'dpafm': 'dpafm'  # + DPAFM: 注意力融合
    }

    if args.task in mode_map:
        train_ablation(task_name=args.task, fusion_mode=mode_map[args.task])