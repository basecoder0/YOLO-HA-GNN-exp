# models/full_pipeline.py - finalgodependencyfixversion

import torch
import torch.nn as nn
import torchvision  # using torchvision replacegeneration ultralytics oftool
import time
from typing import Dict, List, Optional

from .yolo.yolo_with_fusion import YOLOWithFusion
from .gnn.hierarchical_gnn import HierarchicalGNN
from .risk_assessment.risk_calculator import RiskCalculator


class YOLOGNNFusion(nn.Module):
    """
    baseformultiplemodelstatefusionofcompletepipeline (go ultralytics dependencyversion)
    """

    def __init__(self,
                 num_classes: int = 3,
                 confidence_threshold: float = 0.01,  # defaulthigh召fallback
                 nms_threshold: float = 0.6,  # default宽松 NMS
                 use_risk_assessment: bool = False,
                 gnn_max_nodes: int = 100,
                 use_gnn: bool = False):
        super().__init__()

        self.num_classes = num_classes

        # willthresholdasclassattributepropertysave
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.use_risk_assessment = use_risk_assessment
        self.use_gnn = use_gnn
        self.gnn_max_nodes = gnn_max_nodes

        # --- 1. initialization YOLO fusionmodule ---
        # ⚠️ fixpositive：notpassinput confidence_threshold，becausefor/as YOLOWithFusion notconnectbear
        self.yolo_fusion = YOLOWithFusion(num_classes=num_classes)

        # --- 2. GNN module (optional) ---
        self.gnn_enhancement = None
        if use_gnn:
            self.gnn_enhancement = HierarchicalGNN(
                node_feature_dim=512,
                hidden_dim=128,
                num_layers=3,
                dropout=0.1,
                max_nodes=gnn_max_nodes
            )

        # --- 3. riskevaluation (optional) ---
        self.risk_calculator = None
        if use_risk_assessment:
            self.risk_calculator = RiskCalculator()

    def _xywh2xyxy(self, x):
        """willincentercoordinate (cx, cy, w, h) convertfor/as左above右below (x1, y1, x2, y2)"""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def custom_nms(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """
        manualimplement NMS，notdependency ultralytics
        Input: prediction [Batch, 4+Classes, Anchors]
        Output: List[Tensor] per image, Tensor shape [N, 6] (x1,y1,x2,y2,conf,cls)
        """
        # 1. checkinputformatandconvertplace
        # YOLOv8 outputusuallyis [Batch, 4+NC, Anchors]，ourneed [Batch, Anchors, 4+NC]
        if prediction.shape[-1] > prediction.shape[-2]:
            # assume Anchors isfinalonedimension (examplelike 8400)，andcompare 4+NC (examplelike 7) large
            prediction = prediction.transpose(-1, -2)

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

        for xi, x in enumerate(prediction):
            # x: [Anchors, 4+NC]
            # 拆score/divide box and cls
            box = x[:, :4]  # cx, cy, w, h
            cls = x[:, 4:]  # class scores

            # findtoeacha Anchor maximumofcategoryscore
            conf, j = cls.max(1, keepdim=True)

            # thresholdfilter
            mask = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float()), 1)[mask]

            if not x.shape[0]: continue

            # coordinate conversion xywh -> xyxy
            x[:, :4] = self._xywh2xyxy(x[:, :4])

            # limitdetectioncountamount
            if x.shape[0] > 3000:  # predict/prefilter，prevent NMS too慢
                x = x[x[:, 4].argsort(descending=True)[:3000]]

            # extractneedofamountperform NMS
            boxes, scores, idxs = x[:, :4], x[:, 4], x[:, 5]

            # Batched NMS (processingnotsamecategory)
            # 技巧：givenotsamecategoryofboxaddaboveoffsetamount，makegetnotsamecategoryofbox IoU for/as 0，frombut独立 NMS
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
        firsttowardspropagation
        """
        # 1. get YOLO originaloutput [B, 4+C, N]
        raw_preds = self.yolo_fusion({'rgb': rgb, 'thermal': thermal})

        # if YOLOWithFusion returnofislist (multiple尺degree)，usuallyat/in inference modebelow head willprocessing
        # if raw_preds is tuple/list，getfirstaelement素 (assumeiscombineandafterof)
        preds_for_nms = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds

        # 2. executecustom NMS
        detections = self.custom_nms(
            preds_for_nms,
            conf_thres=self.confidence_threshold,
            iou_thres=self.nms_threshold,
            max_det=300
        )

        # detections isonea list，lengthfor/as batch_size
        # eachaelement素is [N, 6] (x1, y1, x2, y2, conf, cls)

        # 3. 封installoutput
        output = {
            'detections': detections[0] if len(detections) > 0 else torch.empty(0, 6).to(rgb.device),
            'raw_preds': raw_preds
        }

        return output

    # compatiblepropertyconnectmouth
    def train_stage1(self):
        self.yolo_fusion.train()
        if self.gnn_enhancement: self.gnn_enhancement.eval()

    def train_stage2(self):
        self.yolo_fusion.eval()
        if self.gnn_enhancement: self.gnn_enhancement.train()