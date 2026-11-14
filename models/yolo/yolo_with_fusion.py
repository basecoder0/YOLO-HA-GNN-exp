# models/yolo/yolo_with_fusion.py
import torch
import torch.nn as nn
from ultralytics import YOLO
from models.fusion.adaptive_fusion import DPAFM
import copy
import math


class YOLOWithFusion(nn.Module):
    """
    [自适应最终版] 双流 YOLOv8 融合模型
    特性：
    1. 自动适配 yolov8n/s/m 不同规模的通道数 (不再硬编码)
    2. 包含《避坑指南》所有防御机制
    """

    def __init__(self, model_path='weights/yolov8s.pt', num_classes=8):
        super().__init__()
        print(f"🏗️ Loading base model from {model_path} ...")

        # 加载基础模型
        base_model = YOLO(model_path)
        m = base_model.model.model  # 获取内部 Sequential

        # --- 1. 双流骨干 ---
        self.rgb_backbone = nn.ModuleList(copy.deepcopy(m[:10]))
        self.thermal_backbone = nn.ModuleList(copy.deepcopy(m[:10]))

        # --- 🔥 核心修复: 自动检测通道数 🔥 ---
        # 必须动态获取，否则换模型必报错
        try:
            ch_p3 = m[4].cv2.conv.out_channels
            ch_p4 = m[6].cv2.conv.out_channels
            ch_p5 = m[9].cv2.conv.out_channels
        except AttributeError:
            # 兼容旧版权重
            ch_p3 = m[4].c2
            ch_p4 = m[6].c2
            ch_p5 = m[9].c2

        print(f"🔍 [Auto-Config] Detected Channels: P3={ch_p3}, P4={ch_p4}, P5={ch_p5}")

        # --- 2. 融合模块 (使用检测到的通道数) ---
        self.fusion_p3 = DPAFM(ch_p3)
        self.fusion_p4 = DPAFM(ch_p4)
        self.fusion_p5 = DPAFM(ch_p5)

        # --- 3. 颈部网络 ---
        self.neck_layers = nn.ModuleList(copy.deepcopy(m[10:22]))

        # --- 4. 检测头 ---
        self.head = copy.deepcopy(m[22])

        if self.head.nc != num_classes:
            self._reset_head(num_classes)

    def _reset_head(self, num_classes):
        """重置检测头 (包含初始化和Buffer修复)"""
        self.head.nc = num_classes
        self.head.no = num_classes + self.head.reg_max * 4

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
                torch.nn.init.normal_(new_conv.weight, std=0.01)
                if new_conv.bias is not None:
                    torch.nn.init.constant_(new_conv.bias, -math.log((1 - 0.01) / 0.01))
                m[-1] = new_conv

        if hasattr(self.head, 'stride'):
            m = self.head
            stride_val = torch.tensor([8., 16., 32.])
            if 'stride' in m.__dict__: del m.stride
            m.register_buffer('stride', stride_val)

    def forward_backbone(self, backbone, x):
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
        if isinstance(x, dict):
            rgb = x['rgb']
            thermal = x['thermal']
        else:
            rgb, thermal = x, x

        rgb_p3, rgb_p4, rgb_p5 = self.forward_backbone(self.rgb_backbone, rgb)
        th_p3, th_p4, th_p5 = self.forward_backbone(self.thermal_backbone, thermal)

        f_p3 = self.fusion_p3(rgb_p3, th_p3)
        f_p4 = self.fusion_p4(rgb_p4, th_p4)
        f_p5 = self.fusion_p5(rgb_p5, th_p5)

        x = self.neck_layers[0](f_p5)
        if x.shape[2:] != f_p4.shape[2:]: x = torch.nn.functional.interpolate(x, size=f_p4.shape[2:], mode='nearest')
        x = torch.cat([x, f_p4], dim=1)
        out_1 = self.neck_layers[2](x)

        x = self.neck_layers[3](out_1)
        if x.shape[2:] != f_p3.shape[2:]: x = torch.nn.functional.interpolate(x, size=f_p3.shape[2:], mode='nearest')
        x = torch.cat([x, f_p3], dim=1)
        out_2 = self.neck_layers[5](x)

        x = self.neck_layers[6](out_2)
        x = torch.cat([x, out_1], dim=1)
        out_3 = self.neck_layers[8](x)

        x = self.neck_layers[9](out_3)
        x = torch.cat([x, f_p5], dim=1)
        out_4 = self.neck_layers[11](x)

        outputs = [out_2, out_3, out_4]
        # 🔥 The "Shape" Trap 防御：按尺寸排序
        outputs.sort(key=lambda x: x.shape[2], reverse=True)

        return self.head(outputs)