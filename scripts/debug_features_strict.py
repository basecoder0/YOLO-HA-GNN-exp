import torch
import torch.nn.functional as F
import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.datasets.r_livit_dataset import RLiViTDataset


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou(box1, box2):
    def box_area(box): return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def run_debug_probe():
    # 配置
    data_cfg = {
        'root_dir': "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT",
        'train_txt': "train.txt",
        'test_txt': "test.txt",
        'img_size': 1280
    }

    test_list_path = os.path.join(PROJECT_ROOT, "stage2_gnn_dataset_1280/stage2_test_list.json")
    if not os.path.exists(test_list_path):
        print(f"❌ 找不到列表文件: {test_list_path}")
        return

    print("📚 初始化 GT 数据集...")
    gt_dataset = RLiViTDataset(data_cfg, mode='test')

    # 建立 ID 索引
    id_to_idx = {}
    print("🗺️ 建立 ID 映射...")
    for i, s in enumerate(gt_dataset.samples):
        path_parts = s['xml'].replace('\\', '/').split('/')
        if len(path_parts) >= 2:
            key = f"{path_parts[-2]}_{os.path.splitext(path_parts[-1])[0]}"
            id_to_idx[key] = i

    print(f"✅ ID 映射建立完成，共 {len(id_to_idx)} 个样本")

    with open(test_list_path, 'r') as f:
        samples = json.load(f)

    X, y_exist = [], []
    limit = 200  # 稍微增加一点样本量以确保统计显著
    print(f"🚀 开始加载前 {limit} 个样本 (修复版 - 已添加 detach)...")

    for i, info in enumerate(samples):
        if i >= limit: break

        pt_path = info['path']
        if not os.path.isabs(pt_path):
            pt_path = os.path.join(PROJECT_ROOT, pt_path)

        print(f"📄 [{i}] 处理: {info['id']}")

        if not os.path.exists(pt_path):
            print(f"    ❌ 文件不存在!")
            continue

        data = torch.load(pt_path, map_location='cpu')

        # 检查特征
        raw_feat = data.get('roi_features', data.get('features'))
        if raw_feat is None:
            print(f"    ❌ 特征缺失! Keys: {data.keys()}")
            continue

        # 🔥🔥🔥 核心修复：加上 .detach() 🔥🔥🔥
        # 即使 tensor 有梯度，detach 后也能安全转 numpy
        feat = F.normalize(torch.nan_to_num(raw_feat.float(), 0.0).clamp(-100, 100), p=2, dim=1).detach().numpy()

        # 检查 Box (确保转换格式)
        boxes_xywh = data['boxes'].float()
        boxes = xywh2xyxy(boxes_xywh)  # 确保使用 xyxy 计算 IoU

        # 匹配 GT
        labels_exist = np.zeros(len(feat))

        if info['id'] in id_to_idx:
            idx = id_to_idx[info['id']]
            sample = gt_dataset[idx]

            if len(sample['labels']) > 0:
                # 再次确认 Dataset 返回的是 Normalized XYWH 还是 Pixel XYWH?
                # RLiViTDataset 返回 Normalized XYWH
                # pt 里的 boxes 也是 Normalized XYWH
                # 所以我们都转成 Normalized XYXY 来算 IoU 是没问题的

                gt_xywh = sample['labels'][:, 2:]
                gt_box = xywh2xyxy(gt_xywh)

                ious = box_iou(boxes, gt_box)  # [N, M]
                if ious.numel() > 0:
                    max_iou, _ = ious.max(dim=1)
                    # 严格定义正样本
                    labels_exist[max_iou > 0.5] = 1
        else:
            print(f"    ⚠️ ID 未找到: {info['id']}")

        X.append(feat)
        y_exist.append(labels_exist)

    if len(X) == 0:
        print("❌ 最终 X 为空，未成功加载任何数据")
        return

    X = np.concatenate(X, axis=0)
    y_exist = np.concatenate(y_exist, axis=0)

    print(f"\n📊 加载成功! 总RoI数: {len(y_exist)}")
    print(f"   正样本 (IoU>0.5): {sum(y_exist == 1)}")
    print(f"   负样本 (IoU<=0.5): {sum(y_exist == 0)}")

    if sum(y_exist == 1) < 10:
        print("⚠️ 正样本太少，无法运行分类器。说明 Stage 1 召回率极低，或者 Label Assignment 还有问题。")
        return

    # 简单的 Linear Probe
    print("\n🧪 正在运行线性探测 (Linear Probe)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_exist, test_size=0.3, random_state=42, stratify=y_exist)

    # 增加 max_iter 防止不收敛
    clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    f1_pos = report['1.0']['f1-score']

    print("-" * 30)
    print(f"🎯 正样本 F1-Score: {f1_pos:.4f}")
    print("-" * 30)

    if f1_pos < 0.3:
        print("❌❌❌ 结论：特征无效！")
        print("   简单的线性分类器完全分不开正负样本。")
        print("   可能原因：")
        print("   1. 特征提取层级错位 (Shape Trap)")
        print("   2. RoI Align 坐标错位 (Format Trap)")
    elif f1_pos < 0.6:
        print("⚠️ 结论：特征质量一般")
        print("   包含一定信息，但噪声很大。")
    else:
        print("✅✅✅ 结论：特征质量优秀")
        print("   特征本身是有区分度的，GNN 应该能训练好。")


if __name__ == "__main__":
    run_debug_probe()