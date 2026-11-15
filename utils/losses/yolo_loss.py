import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox
from ultralytics.utils.ops import xywh2xyxy


class SigmoidFocalLoss(nn.Module):
    """
    [fixversion] branchhold软label (Soft Labels) of Focal Loss
    resolve决 target!=1 problem causing positive samples to be ignored
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        # pred_logits: [B, A, C]
        # target: [B, A, C] (Soft labels 0.0 ~ 1.0)

        pred_prob = torch.sigmoid(pred_logits)

        # calculation BCE Loss (notdo reduce)
        # binary_cross_entropy_with_logits value稳definepropertymoregood
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        # calculation pt (predictionprobabilityandtruelabelofconnectcloseprogramdegree)
        # pairfor软label y, pt optionalby/withclose似for/as p*y + (1-p)*(1-y)
        # when y=1 when, pt=p; when y=0 when, pt=1-p
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # calculation alpha becauseelement (flat衡positivenegativesample)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)

        # calculationadjustcontrolbecauseelement (Modulating Factor)
        modulating_factor = torch.pow((1.0 - p_t), self.gamma)

        # final Loss
        loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HighRecallAssigner(TaskAlignedAssigner):
    """
    松弛matchscore/divideconfiguredevice (Relaxed Matching)
    """

    def __init__(self, topk=13, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)


class CustomYOLOLoss(v8DetectionLoss):
    """
    adaptcustom YOLO-Fusion modelofloss function
    collectgenerate Focal Loss and High Recall Assigner
    """

    def __init__(self, model, hyp, alpha=0.25, gamma=2.0):
        super(v8DetectionLoss, self).__init__()

        self.hyp = hyp
        self.device = next(model.parameters()).device

        m = model.head

        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.stride = m.stride

        # usingfixafterof Focal Loss
        self.bce = SigmoidFocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        self.use_dfl = m.reg_max > 1

        self.bbox_loss = BboxLoss(m.reg_max).to(self.device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=self.device)

        self.assigner = HighRecallAssigner(topk=13,
                                           num_classes=self.nc,
                                           alpha=0.5,
                                           beta=6.0)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]

            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_distri):
        if self.use_dfl:
            b, a, c = pred_distri.shape
            pred_bboxes = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_distri.dtype))
        else:
            pred_bboxes = pred_distri

        return dist2bbox(pred_bboxes, anchor_points, xywh=False)

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
            if torch.__version__ >= '1.10.0':
                sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            else:
                sy, sx = torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)

        targets_raw = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets_raw, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls Loss
        if target_scores_sum > 0:
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        else:
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum()

        # Box Loss
        if fg_mask.sum() > 0:
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['cls']
        loss[2] *= self.hyp['dfl']

        return loss.sum() * batch_size, loss.detach()