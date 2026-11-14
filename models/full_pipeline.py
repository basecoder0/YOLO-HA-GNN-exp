# models/full_pipeline.py - 最终去依赖修复版

import torch
import torch.nn as nn
import torchvision  # 使用 torchvision 替代 ultralytics 的工具
import time
from typing import Dict, List, Optional

from .yolo.yolo_with_fusion import YOLOWithFusion
from .gnn.hierarchical_gnn import HierarchicalGNN
from .risk_assessment.risk_calculator import RiskCalculator


class YOLOGNNFusion(nn.Module):
    """
    基于多模态融合的完整流水线 (去 ultralytics 依赖版)
    """

    def __init__(self,
                 num_classes: int = 3,
                 confidence_threshold: float = 0.01,  # 默认高召回
                 nms_threshold: float = 0.6,  # 默认宽松 NMS
                 use_risk_assessment: bool = False,
                 gnn_max_nodes: int = 100,
                 use_gnn: bool = False):
        super().__init__()

        self.num_classes = num_classes

        # 将阈值作为类属性保存
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.use_risk_assessment = use_risk_assessment
        self.use_gnn = use_gnn
        self.gnn_max_nodes = gnn_max_nodes

        # --- 1. 初始化 YOLO 融合模块 ---
        # ⚠️ 修正：不传入 confidence_threshold，因为 YOLOWithFusion 不接受
        self.yolo_fusion = YOLOWithFusion(num_classes=num_classes)

        # --- 2. GNN 模块 (可选) ---
        self.gnn_enhancement = None
        if use_gnn:
            self.gnn_enhancement = HierarchicalGNN(
                node_feature_dim=512,
                hidden_dim=128,
                num_layers=3,
                dropout=0.1,
                max_nodes=gnn_max_nodes
            )

        # --- 3. 风险评估 (可选) ---
        self.risk_calculator = None
        if use_risk_assessment:
            self.risk_calculator = RiskCalculator()

    def _xywh2xyxy(self, x):
        """将中心坐标 (cx, cy, w, h) 转换为左上右下 (x1, y1, x2, y2)"""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def custom_nms(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """
        手动实现 NMS，不依赖 ultralytics
        Input: prediction [Batch, 4+Classes, Anchors]
        Output: List[Tensor] per image, Tensor shape [N, 6] (x1,y1,x2,y2,conf,cls)
        """
        # 1. 检查输入格式并转置
        # YOLOv8 输出通常是 [Batch, 4+NC, Anchors]，我们需要 [Batch, Anchors, 4+NC]
        if prediction.shape[-1] > prediction.shape[-2]:
            # 假设 Anchors 是最后一维 (例如 8400)，且比 4+NC (例如 7) 大
            prediction = prediction.transpose(-1, -2)

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

        for xi, x in enumerate(prediction):
            # x: [Anchors, 4+NC]
            # 拆分 box 和 cls
            box = x[:, :4]  # cx, cy, w, h
            cls = x[:, 4:]  # class scores

            # 找到每个 Anchor 最大的类别分数
            conf, j = cls.max(1, keepdim=True)

            # 阈值过滤
            mask = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float()), 1)[mask]

            if not x.shape[0]: continue

            # 坐标转换 xywh -> xyxy
            x[:, :4] = self._xywh2xyxy(x[:, :4])

            # 限制检测数量
            if x.shape[0] > 3000:  # 预过滤，防止 NMS 太慢
                x = x[x[:, 4].argsort(descending=True)[:3000]]

            # 提取需要的量进行 NMS
            boxes, scores, idxs = x[:, :4], x[:, 4], x[:, 5]

            # Batched NMS (处理不同类别)
            # 技巧：给不同类别的框加上偏移量，使得不同类别的框 IoU 为 0，从而独立 NMS
            max_wh = 7680  # (pixels) maximum box width and height
            c = x[:, 5:6] * max_wh  # classes
            boxes_offset = boxes + c

            keep = torchvision.ops.nms(boxes_offset, scores, iou_thres)

            if keep.shape[0] > max_det:
                keep = keep[:max_det]

            output[xi] = x[keep]

        return output

    def forward(self, rgb, thermal):
        """
        前向传播
        """
        # 1. 获取 YOLO 原始输出 [B, 4+C, N]
        raw_preds = self.yolo_fusion({'rgb': rgb, 'thermal': thermal})

        # 如果 YOLOWithFusion 返回的是列表 (多尺度)，通常在 inference 模式下 head 会处理
        # 如果 raw_preds 是 tuple/list，取第一个元素 (假设是合并后的)
        preds_for_nms = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds

        # 2. 执行自定义 NMS
        detections = self.custom_nms(
            preds_for_nms,
            conf_thres=self.confidence_threshold,
            iou_thres=self.nms_threshold,
            max_det=300
        )

        # detections 是一个 list，长度为 batch_size
        # 每个元素是 [N, 6] (x1, y1, x2, y2, conf, cls)

        # 3. 封装输出
        output = {
            'detections': detections[0] if len(detections) > 0 else torch.empty(0, 6).to(rgb.device),
            'raw_preds': raw_preds
        }

        return output

    # 兼容性接口
    def train_stage1(self):
        self.yolo_fusion.train()
        if self.gnn_enhancement: self.gnn_enhancement.eval()

    def train_stage2(self):
        self.yolo_fusion.eval()
        if self.gnn_enhancement: self.gnn_enhancement.train()