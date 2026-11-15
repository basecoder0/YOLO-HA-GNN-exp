import torch
import torch.nn as nn
from ultralytics import YOLO
from models.fusion.adaptive_fusion import DPAFM
import copy


class YOLOWithFusion(nn.Module):
    """
    [finalshapestate] dual stream YOLOv8 fusionmodel
    frameworkconstruct：Dual Backbone -> DPAFM Fusion -> Shared Neck (FPN+PAN) -> Detect Head
    """

    def __init__(self, model_path='weights/yolov8n.pt', num_classes=8):
        super().__init__()
        print(f"Loading base model from {model_path} for architecture reconstruction...")
        base_model = YOLO(model_path)
        m = base_model.model.model  # getinsidepart Sequential model

        # --- 1. dual streamskeletondry (Layers 0-9) ---
        # YOLOv8n Backbone containslayer 0-9 (0:Conv, ..., 4:P3, 6:P4, 9:SPPF-P5)
        self.rgb_backbone = nn.ModuleList(copy.deepcopy(m[:10]))
        self.thermal_backbone = nn.ModuleList(copy.deepcopy(m[:10]))

        # --- 2. fusionmodule (P3, P4, P5) ---
        # corresponding backbone ofoutputchannel: P3=64, P4=128, P5=256
        self.fusion_p3 = DPAFM(64)
        self.fusion_p4 = DPAFM(128)
        self.fusion_p5 = DPAFM(256)

        # --- 3. 颈partnetworknetwork (Layers 10-21) ---
        # ourdirectlycomplexusepredict/pretrainingof Neck layerparameter，butneedmanualmanagemanagedata流
        self.neck_layers = nn.ModuleList(copy.deepcopy(m[10:22]))

        # --- 4. detectionhead (Layer 22) ---
        self.head = copy.deepcopy(m[22])

        # resetdetectionheadofcategorycount
        if self.head.nc != num_classes:
            self._reset_head(num_classes)

    def _reset_head(self, num_classes):
        """
        resetdetectionheadofcategorycount
        fixpast tense marker 'Conv' object has no attribute 'in_channels' error
        """
        self.head.nc = num_classes
        # YOLOv8outputchannelcountcalculation: categorycount + fallbackreturnboxcoordinatecount(4 * reg_max)
        self.head.no = num_classes + self.head.reg_max * 4

        # traverse cv3 (score/divideclassbranch) ofeachonelayer (P3, P4, P5)
        for m in self.head.cv3:
            old_conv = m[-1]
            if isinstance(old_conv, nn.Conv2d):
                new_conv = nn.Conv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=num_classes,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=True
                )
                import math
                torch.nn.init.normal_(new_conv.weight, std=0.01)
                if new_conv.bias is not None:
                    torch.nn.init.constant_(new_conv.bias, -math.log((1 - 0.01) / 0.01))
                m[-1] = new_conv

        # 🔥 criticalfix：safety/full地update stride attributeproperty
        if hasattr(self.head, 'stride'):
            m = self.head
            stride_val = torch.tensor([8., 16., 32.])

            # ifitalreadyexist，firstdeletethisa Python attributeproperty，prevent register_buffer error
            del m.stride

            # thenweightnewnotevolumefor/as Buffer (make sureitability/canfollowmodel移motionto GPU)
            m.register_buffer('stride', stride_val)

    def forward_backbone(self, backbone, x):
        """executeskeletondrynetworknetworkandextract P3, P4, P5"""
        y = x
        p3, p4, p5 = None, None, None
        for i, layer in enumerate(backbone):
            y = layer(y)
            if i == 4:
                p3 = y
            elif i == 6:
                p4 = y
            elif i == 9:
                p5 = y
        return p3, p4, p5

    def forward(self, x):
        # 1. parseinput
        if isinstance(x, dict):
            rgb = x['rgb']
            thermal = x['thermal']
        else:
            rgb = x
            thermal = x

            # 2. Backbone firsttowards
        rgb_p3, rgb_p4, rgb_p5 = self.forward_backbone(self.rgb_backbone, rgb)
        th_p3, th_p4, th_p5 = self.forward_backbone(self.thermal_backbone, thermal)

        # 3. fusion
        f_p3 = self.fusion_p3(rgb_p3, th_p3)
        f_p4 = self.fusion_p4(rgb_p4, th_p4)
        f_p5 = self.fusion_p5(rgb_p5, th_p5)

        # 4. Neck firsttowards (standard YOLOv8n logic)
        # -----------------------------------------------------------
        # ⚠️ hereofindexbasefor standard yolov8n.yaml，ifyouofversionnotsame，indexmayoffset
        # but we will in the final step through“sizearrangesequence”tothroughcontainererror
        # -----------------------------------------------------------

        # Layer 10: Upsample
        x = self.neck_layers[0](f_p5)
        # Layer 11: Concat (P5_up + P4) -> resultshouldfor/as 40x40
        # if f_p4 and x sizepairnotabove，here就willdirectlyerror，thisalsoisonetypecheck
        if x.shape[2:] != f_p4.shape[2:]:
            # containererror：strongcontrolpair齐size (万one f_p4 alsoiserrorof)
            x = torch.nn.functional.interpolate(x, size=f_p4.shape[2:], mode='nearest')
        x = torch.cat([x, f_p4], dim=1)

        # Layer 12: C2f (P4_TopDown)
        out_1 = self.neck_layers[2](x)

        # Layer 13: Upsample
        x = self.neck_layers[3](out_1)
        # Layer 14: Concat (P4_up + P3) -> resultshouldfor/as 80x80
        if x.shape[2:] != f_p3.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=f_p3.shape[2:], mode='nearest')
        x = torch.cat([x, f_p3], dim=1)

        # Layer 15: C2f (P3_out) --> this is the highest resolutionfeature (80x80)
        out_2 = self.neck_layers[5](x)

        # Layer 16: Conv (Downsample)
        x = self.neck_layers[6](out_2)
        # Layer 17: Concat
        x = torch.cat([x, out_1], dim=1)

        # Layer 18: C2f (P4_out) --> thisisinwaitresolutionfeature (40x40)
        out_3 = self.neck_layers[8](x)

        # Layer 19: Conv (Downsample)
        x = self.neck_layers[9](out_3)
        # Layer 20: Concat
        x = torch.cat([x, f_p5], dim=1)

        # Layer 21: C2f (P5_out) --> thisislowresolutionfeature (20x20)
        out_4 = self.neck_layers[11](x)

        # -----------------------------------------------------------
        # 🔥🔥🔥 终extremefix：notthen信任changeamountname，only信任size 🔥🔥🔥
        # -----------------------------------------------------------
        outputs = [out_2, out_3, out_4]

        # printdebuginformation (onlyat/infirstwheelprint，confirmsize)
        if not hasattr(self, '_debug_printed'):
            print(f"\n🛑 [DEBUG] Head Inputs Shapes (Before Sort):")
            for i, o in enumerate(outputs):
                print(f"   Input {i}: {o.shape} (H={o.shape[2]})")
            self._debug_printed = True

        # strongcontrolper H (highdegree) reducesequencearrangelist -> [80x80, 40x40, 20x20]
        # corresponding Stride -> [8, 16, 32]
        outputs.sort(key=lambda x: x.shape[2], reverse=True)

        # thenminorconfirm
        if outputs[0].shape[2] < outputs[-1].shape[2]:
            raise RuntimeError("❌ arrangesequencefailed！Head input顺sequencestilliserrorof！")

        return self.head(outputs)