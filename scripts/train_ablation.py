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

# addprojectpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolo.yolo_with_fusion_ablation import YOLOWithFusionAblation
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn


def train_ablation(task_name, fusion_mode):
    # 1. basicconfiguration
    config_path = 'configs/train/stage1.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # autosetexperimentname称
    exp_name = f"ablation_{task_name}"
    cfg['experiment_name'] = exp_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n" + "=" * 50)
    print(f"🚀 starteliminate融experiment: {task_name.upper()}")
    print(f"🔩 fusionmode (Mode): {fusion_mode}")
    print(f"📂 resultsavepath: runs/{exp_name}")
    print(f"=" * 50 + "\n")

    # 2. dataload
    print("📚 positiveat/inloadingdataset...")
    train_dataset = RLiViTDataset(cfg['data'], mode='train')

    # ⚡ debugopen关：if想fasttestcode，geteliminatebelowfacenoteexplain (runfirst100tensorimage)
    # train_dataset.samples = train_dataset.samples[:100]

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        collate_fn=fusion_collate_fn,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    # 3. initializationmodel
    print(f"🏗️ initializationmodel (Fusion Mode: {fusion_mode})...")
    model = YOLOWithFusionAblation(
        model_path='weights/yolov8s.pt',
        num_classes=cfg['model']['num_classes'],
        fusion_mode=fusion_mode
    )
    model.to(device)

    # 4. optimizerconfiguration
    hyp = cfg['hyperparameters']
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=hyp['weight_decay'])
    lf = lambda x: ((1 - x / hyp['epochs']) * (1.0 - hyp['lrf']) + hyp['lrf'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 5. Loss configuration (containscore fix)
    model.args = IterableSimpleNamespace(**hyp)

    # === 🔥 [core fix] manualpointdefine model.model by/withfix v8DetectionLoss error ===
    model.model = [model.head]
    # ===================================================================

    if not hasattr(model.head, 'reg_max'): model.head.reg_max = 16
    criterion = v8DetectionLoss(model)

    # 6. trainingloop
    save_dir = f"runs/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    # eliminate融experimentrecommendrun 30-50 Epoch 即optionalseecleartrendmomentum
    epochs = 30

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # progress bar
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

            # forward propagation
            preds = model(inputs)

            # calculation Loss
            loss, loss_items = criterion(preds, loss_batch)
            loss = loss.sum()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # learning rateadjustment
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # savebestmodel
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))

    print(f"\n✅ experiment {task_name} completed！")
    print(f"💾 bestweightalreadysaveto: {save_dir}/best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Ablation Studies for YOLO-GNN-Fusion')
    parser.add_argument('--task', type=str, required=True,
                        choices=['rgb', 'thermal', 'dpafm'],
                        help='chooseexperimenttask: rgb (Baseline), thermal (+Thermal), dpafm (+DPAFM)')

    args = parser.parse_args()

    # mappingtasknametomodelmode
    mode_map = {
        'rgb': 'rgb',  # Baseline: only RGB
        'thermal': 'add',  # + Thermal: simplemutualadd
        'dpafm': 'dpafm'  # + DPAFM: noteabilityfusion
    }

    if args.task in mode_map:
        train_ablation(task_name=args.task, fusion_mode=mode_map[args.task])