import torch
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms

# 添加项目路径
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn
from ultralytics.utils.ops import xywh2xyxy


def evaluate():
    # 1. 配置
    config_path = 'configs/train/stage1.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Evaluation on {device}...")

    # 2. 加载验证集 (注意：这里用 test_txt 或 train_txt 做验证均可，根据你的划分)
    # 假设我们先验证 train set 看看是不是拟合了
    eval_dataset = RLiViTDataset(cfg['data'], mode='train')

    # 为了节省时间，可以先只跑 200 张看看指标
    # eval_dataset.samples = eval_dataset.samples[:200]

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,  # 验证可以用大 Batch
        shuffle=False,
        collate_fn=fusion_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 3. 加载模型
    model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])

    # 优先加载 best.pt
    ckpt_path = f"runs/{cfg['experiment_name']}/best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"runs/{cfg['experiment_name']}/last.pt"

    print(f"📂 Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 指标统计
    stats = []  # [tp, conf, pred_cls, target_cls]
    iou_thres = 0.5
    conf_thres = 0.001  # 评估时通常用极低阈值以计算 mAP

    print("running evaluation...")
    for batch in tqdm(eval_loader):
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)

        # 归一化
        if rgb.max() > 1.0: rgb = rgb.float() / 255.0
        if thermal.max() > 1.0: thermal = thermal.float() / 255.0

        inputs = {'rgb': rgb, 'thermal': thermal}

        with torch.no_grad():
            # 推理
            preds = model(inputs)
            if isinstance(preds, tuple): preds = preds[0]

            # [bs, 67, 8400] -> [bs, 8400, 67]
            preds = preds.permute(0, 2, 1)

            # 遍历 Batch 中的每一张图
            for i in range(len(preds)):
                # --- 解析预测 ---
                pred = preds[i]  # [8400, 4+nc]

                # 1. 解析 Box (XYWH) 和 Score
                box_xywh = pred[:, :4]
                cls_scores = pred[:, 4:]
                conf, cls_id = cls_scores.max(1)

                # 2. 阈值筛选
                mask = conf > conf_thres
                box_xywh = box_xywh[mask]
                conf = conf[mask]
                cls_id = cls_id[mask]

                if len(box_xywh) == 0:
                    continue

                # 3. 🔥🔥 关键修正：XYWH -> XYXY 🔥🔥
                box_xyxy = xywh2xyxy(box_xywh)

                # 4. NMS
                # 简单的 Class-agnostic NMS
                keep = nms(box_xyxy, conf, 0.7)
                box_xyxy = box_xyxy[keep]
                conf = conf[keep]
                cls_id = cls_id[keep]

                # --- 解析真值 ---
                b_idx = batch['batch_idx']
                target_mask = (b_idx == i)
                target_boxes = batch['bboxes'][target_mask].to(device)  # [N, 4] (cx, cy, w, h)
                target_cls = batch['cls'][target_mask].to(device)

                if len(target_boxes) == 0:
                    continue

                # 🔥 真值也要转 XYXY (如果 Dataset 给的是 xywh)
                # 根据 r_livit_dataset.py，它输出的是 normalized cx, cy, w, h
                # 需要先反归一化 (x640)，再转 XYXY
                img_size = 640
                target_boxes = target_boxes * img_size
                target_boxes_xyxy = xywh2xyxy(target_boxes)

                # --- 计算 Matches (TP/FP) ---
                # 计算预测框和真值框的 IoU
                ious = box_iou(box_xyxy, target_boxes_xyxy)  # [M, N]

                # 简单的贪心匹配
                matches = []  # 记录匹配上的预测框索引
                detected_targets = set()

                if ious.numel() > 0:
                    # 获取每个预测框最大的 IoU
                    max_iou, max_idx = ious.max(1)

                    for p_idx, (iou, t_idx) in enumerate(zip(max_iou, max_idx)):
                        if iou > iou_thres and t_idx.item() not in detected_targets:
                            # 还要检查类别是否一致
                            if cls_id[p_idx] == target_cls[t_idx]:
                                matches.append(p_idx)
                                detected_targets.add(t_idx.item())

                # 记录结果 [tp, conf]
                # 这里做简化计算：只算 Recall
                tp_count = len(matches)
                total_targets = len(target_boxes)
                stats.append((tp_count, total_targets))

    # 汇总计算 Recall
    total_tp = sum(x[0] for x in stats)
    total_objects = sum(x[1] for x in stats)

    recall = total_tp / total_objects if total_objects > 0 else 0
    print(f"\n✅ Evaluation Finished!")
    print(f"🎯 Total Objects: {total_objects}")
    print(f"🎯 Correctly Detected (TP): {total_tp}")
    print(f"🚀 Estimated Recall @ IoU 0.5: {recall:.4f}")

    if recall > 0.3:
        print("\n🎉 PASS! 模型已准备好进行 Stage 2 (GNN) 训练。")
    else:
        print("\n⚠️ FAIL. Recall 依然过低，请检查 NMS 参数或训练时长。")


if __name__ == '__main__':
    evaluate()