import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    用于GNN增强阶段的三个任务：
    1. 置信度重评分
    2. 边界框微调
    3. 风险评分
    """

    def __init__(self,
                 bbox_weight: float = 1.0,
                 confidence_weight: float = 1.0,
                 risk_weight: float = 2.0,
                 smooth_l1_beta: float = 1.0):
        super().__init__()

        self.bbox_weight = bbox_weight
        self.confidence_weight = confidence_weight
        self.risk_weight = risk_weight
        self.smooth_l1_beta = smooth_l1_beta

        self.bce_loss = nn.BCELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=smooth_l1_beta)

    def forward(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        计算多任务损失

        Args:
            predictions: 模型预测输出
            targets: 真实标签

        Returns:
            total_loss: 总损失
        """
        total_loss = 0.0
        loss_components = {}

        # 置信度重评分损失
        if 'confidence_scores' in predictions and 'confidence_targets' in targets:
            conf_pred = predictions['confidence_scores']
            conf_target = targets['confidence_targets']

            if conf_pred.numel() > 0 and conf_target.numel() > 0:
                conf_loss = self.bce_loss(conf_pred, conf_target)
                total_loss += self.confidence_weight * conf_loss
                loss_components['confidence_loss'] = conf_loss.item()

        # 边界框微调损失
        if 'bbox_deltas' in predictions and 'bbox_targets' in targets:
            bbox_pred = predictions['bbox_deltas']
            bbox_target = targets['bbox_targets']

            if bbox_pred.numel() > 0 and bbox_target.numel() > 0:
                bbox_loss = self.smooth_l1_loss(bbox_pred, bbox_target)
                total_loss += self.bbox_weight * bbox_loss
                loss_components['bbox_loss'] = bbox_loss.item()

        # 风险评分损失
        if 'risk_scores' in predictions and 'risk_targets' in targets:
            risk_pred = predictions['risk_scores']
            risk_target = targets['risk_targets']

            if risk_pred.numel() > 0 and risk_target.numel() > 0:
                risk_loss = self.bce_loss(risk_pred, risk_target)
                total_loss += self.risk_weight * risk_loss
                loss_components['risk_loss'] = risk_loss.item()

        # 如果没有有效的预测，返回零损失
        if total_loss == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        return total_loss

    def compute_detection_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """
        计算检测相关指标
        """
        metrics = {}

        # 置信度准确率
        if 'confidence_scores' in predictions and 'confidence_targets' in targets:
            conf_pred = predictions['confidence_scores']
            conf_target = targets['confidence_targets']

            if conf_pred.numel() > 0:
                conf_pred_binary = (conf_pred > 0.5).float()
                conf_accuracy = (conf_pred_binary == conf_target).float().mean()
                metrics['confidence_accuracy'] = conf_accuracy.item()

        # 边界框IoU改进
        if 'bbox_deltas' in predictions:
            bbox_deltas = predictions['bbox_deltas']
            if bbox_deltas.numel() > 0:
                metrics['bbox_delta_norm'] = bbox_deltas.norm(dim=1).mean().item()

        return metrics