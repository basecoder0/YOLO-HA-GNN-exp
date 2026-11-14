import torch
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.gnn.hierarchical_gnn import HierarchicalGNN
from torch_geometric.nn import GATConv


# ==========================================
# 1. 定义一个带 Hook 的 GNN 包装器
# ==========================================
class VisualizableGNN(HierarchicalGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = {}  # 存储每一层的注意力

    def forward_with_attention(self, node_features, bboxes, scores):
        # 这是一个简化版的 forward，专门用于提取 attention
        # 我们只处理 batch size = 1 的情况进行可视化

        # 1. 预处理 (Filter & Build Graph)
        feat, box, keep_idx = self._filter_nodes(node_features[0], bboxes[0], scores[0])
        if feat.size(0) < 2: return None, None, None  # 节点太少，没法画图

        edge_index = self._build_graph_for_sample(feat, box)

        # 2. 编码
        x = self.node_encoder(feat)

        # 3. 提取 Appearance Attention
        # 临时替换 GATConv 的 forward 行为
        x_app, (edge_index_app, alpha_app) = self.appearance_attention.attention(x, edge_index,
                                                                                 return_attention_weights=True)
        self.attn_weights['appearance'] = (edge_index_app, alpha_app)

        # 4. 提取 Spatial Attention
        sp_weight = self.compute_spatial_weights(box, device=x.device)
        # Spatial Layer 有点特殊，它在 forward 里做了乘法，我们这里手动模拟一下
        x_spa_raw, (edge_index_spa, alpha_spa) = self.spatial_attention.attention(x, edge_index,
                                                                                  return_attention_weights=True)
        self.attn_weights['spatial'] = (edge_index_spa, alpha_spa)

        return box, self.attn_weights, keep_idx


# ==========================================
# 2. 可视化绘图函数
# ==========================================
def draw_attention_map(img, boxes, edge_index, attn_scores, title, save_path):
    """
    在图片上画出节点之间的连线，线的粗细/颜色深浅代表注意力权重
    """
    vis_img = img.copy()
    h, w = vis_img.shape[:2]

    # 归一化注意力分数以便绘图
    attn_scores = attn_scores.mean(dim=1).detach().cpu().numpy()  # Average over heads
    attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-6)

    edge_index = edge_index.cpu().numpy()

    # 画框
    centers = []
    for box in boxes:
        # box is [x, y, w, h] (normalized) or [x1, y1, x2, y2]
        # 假设是归一化 [cx, cy, w, h] -> 还原到像素
        cx, cy, bw, bh = box
        px, py = int(cx * 1280), int(cy * 1280)  # 假设 1280 尺寸
        centers.append((px, py))

        x1, y1 = int((cx - bw / 2) * 1280), int((cy - bh / 2) * 1280)
        x2, y2 = int((cx + bw / 2) * 1280), int((cy + bh / 2) * 1280)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 画线
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        score = attn_scores[i]

        if score < 0.2: continue  # 忽略弱连接

        pt1 = centers[src]
        pt2 = centers[dst]

        # 颜色：红色代表强关联，蓝色代表弱关联
        color_val = int(255 * score)
        color = (255 - color_val, 0, color_val)  # BGR
        thickness = max(1, int(score * 3))

        cv2.line(vis_img, pt1, pt2, color, thickness)

    cv2.putText(vis_img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(save_path, vis_img)
    print(f"🎨 Saved attention map: {save_path}")


# ==========================================
# 3. 主流程
# ==========================================
def run_visualization(ckpt_path, sample_pt_path, img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型 (使用我们的 Wrapper)
    model = VisualizableGNN(node_feature_dim=256, hidden_dim=256).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 加载数据
    data = torch.load(sample_pt_path, map_location=device)
    feat = [data['features'].float()]
    boxes = [data['boxes'].float()]
    scores = [data['scores'][:, 0]]

    # 提取注意力
    boxes_filtered, attn_weights, _ = model.forward_with_attention(feat, boxes, scores)

    if boxes_filtered is None:
        print("❌ 节点数量不足，无法可视化")
        return

    # 加载原图
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1280, 1280))  # 确保尺寸匹配

    # 绘制 Appearance Attention
    edge_idx, alpha = attn_weights['appearance']
    draw_attention_map(img, boxes_filtered, edge_idx, alpha, "Appearance Attention", "vis_attn_app.jpg")

    # 绘制 Spatial Attention
    edge_idx, alpha = attn_weights['spatial']
    draw_attention_map(img, boxes_filtered, edge_idx, alpha, "Spatial Attention", "vis_attn_spa.jpg")


if __name__ == "__main__":
    # 填入实际路径
    ckpt = "runs/stage2_gnn_1280/epoch_100.pth"
    pt_file = "stage2_gnn_dataset_1280/processed/sample_0.pt"  # 需要找一个具体的样本
    img_file = "raw_data/images/sample_0.png"

    if os.path.exists(ckpt) and os.path.exists(pt_file):
        run_visualization(ckpt, pt_file, img_file)
    else:
        print("请检查文件路径配置")