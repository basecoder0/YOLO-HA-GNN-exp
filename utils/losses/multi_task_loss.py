import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    multipletaskloss function
    forGNNthree tasks in the enhancement phase：
    1. confidenceweightevaluatescore/divide
    2. boundarybox微adjust
    3. riskevaluatescore/divide
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
        calculationmultipletaskloss

        Args:
            predictions: modelpredictionoutput
            targets: truelabel

        Returns:
            total_loss: totalloss
        """
        total_loss = 0.0
        loss_components = {}

        # confidenceweightevaluatescore/divideloss
        if 'confidence_scores' in predictions and 'confidence_targets' in targets:
            conf_pred = predictions['confidence_scores']
            conf_target = targets['confidence_targets']

            if conf_pred.numel() > 0 and conf_target.numel() > 0:
                conf_loss = self.bce_loss(conf_pred, conf_target)
                total_loss += self.confidence_weight * conf_loss
                loss_components['confidence_loss'] = conf_loss.item()

        # boundarybox微adjustloss
        if 'bbox_deltas' in predictions and 'bbox_targets' in targets:
            bbox_pred = predictions['bbox_deltas']
            bbox_target = targets['bbox_targets']

            if bbox_pred.numel() > 0 and bbox_target.numel() > 0:
                bbox_loss = self.smooth_l1_loss(bbox_pred, bbox_target)
                total_loss += self.bbox_weight * bbox_loss
                loss_components['bbox_loss'] = bbox_loss.item()

        # riskevaluatescore/divideloss
        if 'risk_scores' in predictions and 'risk_targets' in targets:
            risk_pred = predictions['risk_scores']
            risk_target = targets['risk_targets']

            if risk_pred.numel() > 0 and risk_target.numel() > 0:
                risk_loss = self.bce_loss(risk_pred, risk_target)
                total_loss += self.risk_weight * risk_loss
                loss_components['risk_loss'] = risk_loss.item()

        # ifnothavehaveeffectofprediction，returnzeroloss
        if total_loss == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        return total_loss

    def compute_detection_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """
        calculationdetectionmutual关metric
        """
        metrics = {}

        # confidenceaccuracy
        if 'confidence_scores' in predictions and 'confidence_targets' in targets:
            conf_pred = predictions['confidence_scores']
            conf_target = targets['confidence_targets']

            if conf_pred.numel() > 0:
                conf_pred_binary = (conf_pred > 0.5).float()
                conf_accuracy = (conf_pred_binary == conf_target).float().mean()
                metrics['confidence_accuracy'] = conf_accuracy.item()

        # boundaryboxIoUimprove
        if 'bbox_deltas' in predictions:
            bbox_deltas = predictions['bbox_deltas']
            if bbox_deltas.numel() > 0:
                metrics['bbox_delta_norm'] = bbox_deltas.norm(dim=1).mean().item()

        return metrics