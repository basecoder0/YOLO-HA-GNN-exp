import torch
import torch.nn.functional as F
import os
import sys
import json
import numpy as np

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from utils.device_utils import setup_device

# ================= 配置 =================
# 指向你的 Stage 2 数据集和模型
DATASET_JSON = os.path.join(PROJECT_ROOT, "stage2_gnn_dataset_1280", "stage2_test_list.json")
# 你的 GNN 权重路径
CKPT_PATH = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_1280\epoch_100.pth"


# =======================================

def diagnose():
    device = setup_device()
    print(f"🔬 启动 GNN 深度诊断 (Device: {device})")
    print(f"   📂 项目根目录: {PROJECT_ROOT}")

    # 1. 检查数据集索引文件
    if not os.path.exists(DATASET_JSON):
        print(f"❌ 找不到数据集索引: {DATASET_JSON}")
        return

    with open(DATASET_JSON, 'r') as f:
        samples = json.load(f)
    print(f"✅ 数据集索引加载成功，共 {len(samples)} 个样本")

    # 2. 寻找样本 (包含路径修复逻辑)
    target_sample = None
    first_path_raw = samples[0]['path'] if len(samples) > 0 else "None"
    print(f"   ℹ️ JSON 中记录的第一条路径 (原始): {first_path_raw}")

    for s in samples:
        pt_path = s['path']

        # 🔥🔥🔥 核心修复: 自动拼接绝对路径 🔥🔥🔥
        if not os.path.isabs(pt_path):
            pt_path = os.path.join(PROJECT_ROOT, pt_path)

        if os.path.exists(pt_path):
            target_sample = s
            target_sample['path'] = pt_path  # 更新为正确的绝对路径
            break

    if target_sample is None:
        print("❌ 严重错误：无法找到任何存在的 .pt 文件！")
        print(f"   尝试寻找的路径示例: {os.path.join(PROJECT_ROOT, samples[0]['path'])}")
        print("   请检查: stage2_gnn_dataset_1280 文件夹是否在项目根目录下？")
        return

    print(f"\n🧐 正在解剖样本: {target_sample['id']}")
    print(f"   📂 实际加载路径: {target_sample['path']}")

    # 3. 加载 .pt 数据
    data = torch.load(target_sample['path'], map_location=device, weights_only=False)
    print("\n📦 [.pt 数据内容检查]")
    print(f"   Keys: {list(data.keys())}")

    # 检查特征
    raw_feat = data.get('roi_features', data.get('features'))
    if raw_feat is None:
        print("   ❌ 特征缺失！找不到 'roi_features' 或 'features'")
        return
    print(f"   Feature Shape: {raw_feat.shape}")
    print(f"   Feature Stats: Min={raw_feat.min():.4f}, Max={raw_feat.max():.4f}, Mean={raw_feat.mean():.4f}")
    if torch.isnan(raw_feat).any():
        print("   ❌ 警告：特征中包含 NaN！")

    # 检查盒子
    boxes = data['boxes']
    print(f"   Boxes Shape: {boxes.shape} (预期是归一化坐标)")

    # 检查分数
    scores = data['scores']
    print(f"   Scores Shape: {scores.shape}")

    # 检查标签 (这是关键！)
    labels = data.get('labels', data.get('gt_labels'))
    if labels is None:
        print("   ⚠️ 警告：.pt 文件中没有 'labels' 或 'gt_labels'！")
    else:
        print(f"   Labels Found! Shape: {labels.shape}")
        if len(labels) > 0:
            print(f"   Labels Sample (Raw): {labels[0].tolist()}")
            # 检查是否是 1-based
            unique_cls = labels[:, 0].unique() if labels.dim() > 1 else labels.unique()
            print(f"   Unique Classes: {unique_cls.tolist()}")
            if unique_cls.min() >= 1:
                print("   ℹ️ 发现标签是 1-based (1-8)，评估脚本已包含自动减 1 的修复。")
            else:
                print("   ℹ️ 标签看起来是 0-based (0-7)。")

    # 4. 加载模型
    print("\n🤖 [模型加载与推理]")
    model = HierarchicalGNN(node_feature_dim=256, hidden_dim=256).to(device)
    head_ex = torch.nn.Linear(128, 1).to(device)
    head_cl = torch.nn.Linear(128, 8).to(device)

    if not os.path.exists(CKPT_PATH):
        print(f"❌ 找不到权重文件: {CKPT_PATH}")
        return

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # 兼容性加载
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
        head_ex.load_state_dict(ckpt['head_exist'])
        head_cl.load_state_dict(ckpt['head_cls'])
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval();
    head_ex.eval();
    head_cl.eval()
    print("   ✅ 模型权重加载成功")

    # 检查权重统计
    w_ex = head_ex.weight.data
    print(f"   Head_Exist Weight Stats: Mean={w_ex.mean():.6f}, Std={w_ex.std():.6f}")

    # 5. 前向传播
    # 模拟评估时的预处理
    feat = F.normalize(torch.nan_to_num(raw_feat.float(), 0.0).clamp(-100, 100), p=2, dim=1)

    with torch.no_grad():
        out, keep, _ = model([feat], [boxes.float()], [scores[:, 0]])

        # 打印中间特征
        print(f"   GNN Output Shape: {out.shape}")

        # 预测 Exist
        logits_ex = head_ex(out)
        prob_ex = torch.sigmoid(logits_ex)

        # 预测 Class
        logits_cl = head_cl(out)
        prob_cl = torch.softmax(logits_cl, dim=1)
        pred_cls = torch.argmax(prob_cl, dim=1)

    print("\n🔍 [预测结果分析]")
    print(f"   Exist Logits (Raw): Min={logits_ex.min():.4f}, Max={logits_ex.max():.4f}, Mean={logits_ex.mean():.4f}")
    print(f"   Exist Prob (Sigmoid): Min={prob_ex.min():.4f}, Max={prob_ex.max():.4f}, Mean={prob_ex.mean():.4f}")

    # 统计有多少个样本超过了 0.4 的阈值
    pass_count = (prob_ex > 0.4).sum().item()
    total_count = prob_ex.numel()
    print(f"   🔥 阈值测试 (Threshold=0.4): {pass_count}/{total_count} 个节点通过")

    if pass_count == 0:
        print("   ❌ 潜在问题：所有节点的存在性分数都低于 0.4。")
        print("      建议：在 evaluate_final.py 中将 GNN 阈值从 0.4 降低到 0.1 或 0.01 试试。")
    else:
        print("   ✅ 存在性分数分布正常。")

    print(f"   Class Predictions (Top 10): {pred_cls[:10].tolist()}")


if __name__ == "__main__":
    diagnose()