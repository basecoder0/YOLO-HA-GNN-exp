import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import torchvision
import json
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from data.datasets.r_livit_dataset import RLiViTDataset
from utils.device_utils import setup_device
from scripts.evaluate_ablation import RobustGNNEvaluator, DATA_CFG, GNN_DATASET_PATH


def sweep_thresholds():
    device = setup_device()
    print(f"🚀 开始 GNN 阈值扫描 (寻找最佳 mAP)...")

    # 你的 GNN 权重
    ckpt_path = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_1280\epoch_100.pth"
    if not os.path.exists(ckpt_path):
        print("❌ 找不到权重文件");
        return

    # 初始化评估器
    evaluator = RobustGNNEvaluator(GNN_DATASET_PATH, device, DATA_CFG)

    # 加载模型
    model, head_ex, head_cl = evaluator.load_model(ckpt_path)

    # 加载所有数据到内存 (加速扫描)
    print("⏳ 正在预加载数据...")
    with open(GNN_DATASET_PATH, 'r') as f:
        samples = json.load(f)

    cached_data = []
    for info in tqdm(samples):
        pt_path = info['path']
        if not os.path.isabs(pt_path): pt_path = os.path.join(PROJECT_ROOT, pt_path)
        if not os.path.exists(pt_path): continue

        data = torch.load(pt_path, map_location=device)
        raw_feat = data.get('roi_features', data.get('features'))
        if raw_feat is None: continue

        # 预处理
        feat = F.normalize(torch.nan_to_num(raw_feat.float(), 0.0).clamp(-100, 100), p=2, dim=1)
        boxes = data['boxes'].float()
        scores = data['scores'][:, 0]

        # 推理一次，保存结果
        with torch.no_grad():
            out, keep, _ = model([feat], [boxes], [scores])
            p_ex = torch.sigmoid(head_ex(out))
            p_cl = torch.argmax(head_cl(out), dim=1)

        # 获取真值
        labels = evaluator.get_gt_labels(info['id'])

        cached_data.append({
            'keep': keep[0], 'boxes': boxes, 'p_ex': p_ex, 'p_cl': p_cl, 'labels': labels
        })

    print(f"✅ 数据预加载完成 ({len(cached_data)} 样本). 开始扫描...")

    # 🔥 扫描不同阈值
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    best_map = 0.0
    best_res = None

    print(f"{'Thres':<10} | {'Recall':<10} | {'mAP50':<10} | {'F1':<10}")
    print("-" * 50)

    for th in thresholds:
        stats = []
        for d in cached_data:
            keep_idxs = d['keep']
            if len(keep_idxs) == 0: continue

            # 还原坐标
            pred_boxes = d['boxes'][keep_idxs] * evaluator.img_size  # xywh
            # ⚠️ 注意：这里要转 xyxy！
            # 这里的 boxes 是来自 .pt，我们刚才确认了它是 xywh (center)
            # 所以需要转换
            from scripts.evaluate_ablation import xywh2xyxy
            pred_boxes = xywh2xyxy(pred_boxes)

            pred_scores = d['p_ex'].squeeze()
            pred_cls = d['p_cl']

            # 🔥 应用当前阈值
            mask = pred_scores > th

            p_box = pred_boxes[mask]
            p_score = pred_scores[mask]
            p_cls = pred_cls[mask]

            if len(p_box) == 0:
                if len(d['labels']) > 0:
                    stats.append((np.zeros((0, 10), dtype=bool), [], [], d['labels'][:, 0].cpu().numpy()))
                continue

            preds = torch.cat([p_box, p_score.unsqueeze(1), p_cls.unsqueeze(1).float()], 1)

            # NMS
            keep_nms = torchvision.ops.nms(preds[:, :4], preds[:, 4], 0.3)
            preds = preds[keep_nms]

            # 统计
            from scripts.evaluate_ablation import process_batch_statistics, ap_per_class
            correct = process_batch_statistics(preds, d['labels'], evaluator.niou)
            stats.append(
                (correct, preds[:, 4].cpu().numpy(), preds[:, 5].cpu().numpy(), d['labels'][:, 0].cpu().numpy()))

        # 计算指标
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            p, r, ap, f1 = ap_per_class(*stats)
            map50 = ap[:, 0].mean()
            recall = r.mean()
            f1_score = f1.mean()

            print(f"{th:<10.2f} | {recall:<10.3f} | {map50:<10.3f} | {f1_score:<10.3f}")

            if map50 > best_map:
                best_map = map50
                best_res = {'th': th, 'r': recall, 'map': map50, 'f1': f1_score}

    print("\n🏆 最佳结果:")
    print(f"阈值: {best_res['th']}")
    print(f"Recall: {best_res['r']:.3f}")
    print(f"mAP50: {best_res['map']:.3f} (对比 DPAFM: 0.347)")

    if best_res['map'] > 0.347:
        print("✅ 恭喜！Ours 成功超越 Stage 1！")
    else:
        print("⚠️ 仍然未超越。可能原因：GNN 训练不足，或特征提取有问题。")


if __name__ == "__main__":
    sweep_thresholds()