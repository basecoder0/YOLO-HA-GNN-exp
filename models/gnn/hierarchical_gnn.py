# models/gnn/hierarchical_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from typing import List
import math

class AppearanceAttentionLayer(nn.Module):
    def __init__(self, node_feature_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = GATConv(
            node_feature_dim, node_feature_dim // heads,
            heads=heads, dropout=dropout, concat=True
        )
        self.norm = nn.LayerNorm(node_feature_dim)

    def forward(self, x, edge_index):
        if x.size(0) == 0: return x
        out = self.attention(x, edge_index)
        return self.norm(out + x)

class SpatialAttentionLayer(nn.Module):
    def __init__(self, node_feature_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = GATConv(
            node_feature_dim, node_feature_dim // heads,
            heads=heads, dropout=dropout, concat=True
        )
        self.norm = nn.LayerNorm(node_feature_dim)

    def forward(self, x, edge_index, spatial_weights=None):
        if x.size(0) == 0: return x
        out = self.attention(x, edge_index)
        if spatial_weights is not None:
            # 维度对齐保护
            out = out * spatial_weights.unsqueeze(1)
        return self.norm(out + x)

class MotionAttentionLayer(nn.Module):
    def __init__(self, node_feature_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = GATConv(
            node_feature_dim, node_feature_dim // heads,
            heads=heads, dropout=dropout, concat=True
        )
        self.norm = nn.LayerNorm(node_feature_dim)

    def forward(self, x, edge_index, motion_weights=None):
        if x.size(0) == 0: return x
        out = self.attention(x, edge_index)
        if motion_weights is not None:
            out = out * motion_weights.unsqueeze(1)
        return self.norm(out + x)

class HierarchicalGNN(nn.Module):
    """
    分层注意力 GNN (最终稳健版：LayerNorm + Tanh + Epsilon + Clamp)
    """
    def __init__(self,
                 node_feature_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.1,
                 occlusion_aware: bool = True,
                 max_nodes: int = 100,
                 k_neighbors: int = 10,
                 distance_threshold: float = 0.2):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.occlusion_aware = occlusion_aware
        self.max_nodes = max_nodes
        self.k_neighbors = k_neighbors

        # 1. 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 输入稳健化
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 2. 分层注意力
        self.appearance_attention = AppearanceAttentionLayer(hidden_dim, heads, dropout)
        self.spatial_attention = SpatialAttentionLayer(hidden_dim, heads, dropout)
        self.motion_attention = MotionAttentionLayer(hidden_dim, heads, dropout)

        if occlusion_aware:
            self.occlusion_weights = nn.Parameter(torch.ones(3))

        # 3. GNN 层
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 4. 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(), # 🔥 限制输出范围 [-1, 1]，防止梯度爆炸
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

    def _filter_nodes(self, features, bboxes, scores):
        num_nodes = features.size(0)
        # 空图保护
        if num_nodes == 0: return features, bboxes, []

        if num_nodes <= self.max_nodes:
            return features, bboxes, list(range(num_nodes))

        if scores is not None:
            s = scores if isinstance(scores, torch.Tensor) else torch.tensor(scores)
            if s.dim() > 1: s = s[:, 0]

            # 长度不匹配保护
            if s.size(0) != num_nodes:
                keep_indices = torch.randperm(num_nodes)[:self.max_nodes].tolist()
            else:
                _, indices = torch.topk(s, self.max_nodes)
                keep_indices = indices.tolist()
        else:
            keep_indices = torch.randperm(num_nodes)[:self.max_nodes].tolist()

        filtered_feat = features[keep_indices]
        filtered_box = [bboxes[i] for i in keep_indices]
        return filtered_feat, filtered_box, keep_indices

    def _build_graph_for_sample(self, x, bboxes):
        device = x.device
        num_nodes = x.size(0)
        if num_nodes < 2:
            return torch.zeros((2, 0), dtype=torch.long, device=device)

        centers = torch.tensor([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in bboxes], device=device)
        dist = torch.cdist(centers, centers)
        dist.fill_diagonal_(float('inf'))

        k = min(self.k_neighbors, num_nodes - 1)
        _, indices = dist.topk(k, largest=False)

        source = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, k).view(-1)
        target = indices.view(-1)
        return torch.stack([source, target], dim=0)

    def compute_spatial_weights(self, bboxes, image_size=(640,640), device=None):
        num_nodes = len(bboxes)
        if num_nodes == 0: return torch.zeros(0, device=device)

        weights = torch.zeros(num_nodes, device=device)

        # 🔥 防除零保护 (Epsilon)
        diag = math.sqrt(image_size[0]**2 + image_size[1]**2) + 1e-6
        cx, cy = image_size[0]/2, image_size[1]/2

        for i, b in enumerate(bboxes):
            bcx, bcy = (b[0]+b[2])/2, (b[1]+b[3])/2
            d = math.sqrt((bcx-cx)**2 + (bcy-cy)**2)

            # 🔥 数值计算保护
            weights[i] = 1.0 - (d / diag)

        # 🔥 范围强制截断 [0, 1]
        return torch.clamp(weights, min=0.0, max=1.0)

    def forward(self, node_features: List[torch.Tensor], bboxes: List[List[List[float]]], confidence_scores: List[torch.Tensor] = None):
        batch_data_list = []
        keep_indices_list = []

        # 空批次保护
        if not node_features:
            return torch.zeros(0, self.hidden_dim // 2), [], None

        device = node_features[0].device

        for i, (feat, box) in enumerate(zip(node_features, bboxes)):
            score = confidence_scores[i] if confidence_scores else None
            f_feat, f_box, keep_idx = self._filter_nodes(feat, box, score)
            keep_indices_list.append(keep_idx)

            if f_feat.size(0) > 0:
                edge_index = self._build_graph_for_sample(f_feat, f_box)
                sp_weight = self.compute_spatial_weights(f_box, device=device)
                data = Data(x=f_feat, edge_index=edge_index, sp_weight=sp_weight)
                batch_data_list.append(data)
            else:
                data = Data(x=torch.zeros(0, self.node_feature_dim, device=device), 
                           edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
                           sp_weight=torch.zeros(0, device=device))
                batch_data_list.append(data)

        batch = Batch.from_data_list(batch_data_list)
        x, edge_index = batch.x, batch.edge_index

        if x.size(0) == 0:
            return torch.zeros(0, self.hidden_dim // 2, device=device), keep_indices_list, batch.batch

        # GNN 前向传播
        x = self.node_encoder(x)

        x_app = self.appearance_attention(x, edge_index)
        x_spa = self.spatial_attention(x, edge_index, batch.sp_weight)
        x_mot = x 

        if self.occlusion_aware:
            w = torch.softmax(self.occlusion_weights, dim=0)
            x_enhanced = w[0]*x_app + w[1]*x_spa + w[2]*x_mot
        else:
            x_enhanced = (x_app + x_spa + x_mot) / 3.0

        for layer, norm in zip(self.gnn_layers, self.norms):
            res = x_enhanced
            x_enhanced = layer(x_enhanced, edge_index)
            x_enhanced = norm(F.relu(x_enhanced) + res)

        out = self.output_proj(x_enhanced)
        return out, keep_indices_list, batch.batch
