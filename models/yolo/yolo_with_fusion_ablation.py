# models/yolo/yolo_with_fusion_ablation.py
import torch
import torch.nn as nn
from ultralytics import YOLO
from models.fusion.adaptive_fusion import DPAFM
import copy
import math


class YOLOWithFusionAblation(nn.Module):
    """
    [消融实验全能版] 双流 YOLOv8 融合模型
    支持模式 (fusion_mode):
        - 'rgb':    Baseline (仅使用 RGB，完全忽略 Thermal)
        - 'add':    + Thermal (RGB + Thermal 简单相加)
        - 'concat': + Thermal (RGB + Thermal 拼接)
        - 'dpafm':  + DPAFM (自适应注意力融合，本方案核心)
    """

    def __init__(self, model_path='weights/yolov8s.pt', num_classes=8, fusion_mode='dpafm'):
        super().__init__()
        print(f"🏗️ Loading Ablation Model from {model_path} ...")
        self.fusion_mode = fusion_mode
        print(f"🎛️ Ablation Mode: {self.fusion_mode}")

        # 加载基础模型
        base_model = YOLO(model_path)
        m = base_model.model.model

        # --- 1. 骨干网络 ---
        # RGB 分支总是需要的
        self.rgb_backbone = nn.ModuleList(copy.deepcopy(m[:10]))

        # Thermal 分支：只有在非 RGB 模式下才初始化，节省显存
        if self.fusion_mode != 'rgb':
            self.thermal_backbone = nn.ModuleList(copy.deepcopy(m[:10]))
        else:
            self.thermal_backbone = None

        # --- 自动检测通道数 ---
        try:
            ch_p3 = m[4].cv2.conv.out_channels
            ch_p4 = m[6].cv2.conv.out_channels
            ch_p5 = m[9].cv2.conv.out_channels
        except AttributeError:
            ch_p3 = m[4].c2
            ch_p4 = m[6].c2
            ch_p5 = m[9].c2

        print(f"🔍 [Auto-Config] Detected Channels: P3={ch_p3}, P4={ch_p4}, P5={ch_p5}")

        # --- 2. 融合模块 ---
        if self.fusion_mode == 'dpafm':
            self.fusion_p3 = DPAFM(ch_p3)
            self.fusion_p4 = DPAFM(ch_p4)
            self.fusion_p5 = DPAFM(ch_p5)
        elif self.fusion_mode == 'concat':
            # 拼接后降维 (2*C -> C)
            self.fusion_p3 = nn.Conv2d(ch_p3 * 2, ch_p3, 1, bias=False)
            self.fusion_p4 = nn.Conv2d(ch_p4 * 2, ch_p4, 1, bias=False)
            self.fusion_p5 = nn.Conv2d(ch_p5 * 2, ch_p5, 1, bias=False)
        else:
            # 'rgb' 或 'add' 模式不需要参数化层
            self.fusion_p3 = nn.Identity()
            self.fusion_p4 = nn.Identity()
            self.fusion_p5 = nn.Identity()

        # --- 3. 颈部网络 & 检测头 ---
        self.neck_layers = nn.ModuleList(copy.deepcopy(m[10:22]))
        self.head = copy.deepcopy(m[22])

        if self.head.nc != num_classes:
            self._reset_head(num_classes)

    def _reset_head(self, num_classes):
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
        # 1. 解析输入
        if isinstance(x, dict):
            rgb = x['rgb']
            thermal = x['thermal']
        else:
            rgb = x
            thermal = None  # 如果不是字典，假设只有RGB

        # 2. RGB 骨干推理 (所有模式都需要)
        rgb_p3, rgb_p4, rgb_p5 = self.forward_backbone(self.rgb_backbone, rgb)

        # 3. 融合逻辑
        if self.fusion_mode == 'rgb':
            # === Baseline: 仅使用 RGB ===
            f_p3, f_p4, f_p5 = rgb_p3, rgb_p4, rgb_p5
        else:
            # === 双模态: 需要 Thermal ===
            th_p3, th_p4, th_p5 = self.forward_backbone(self.thermal_backbone, thermal)

            if self.fusion_mode == 'dpafm':
                f_p3 = self.fusion_p3(rgb_p3, th_p3)
                f_p4 = self.fusion_p4(rgb_p4, th_p4)
                f_p5 = self.fusion_p5(rgb_p5, th_p5)
            elif self.fusion_mode == 'add':
                f_p3 = rgb_p3 + th_p3
                f_p4 = rgb_p4 + th_p4
                f_p5 = rgb_p5 + th_p5
            elif self.fusion_mode == 'concat':
                f_p3 = self.fusion_p3(torch.cat([rgb_p3, th_p3], dim=1))
                f_p4 = self.fusion_p4(torch.cat([rgb_p4, th_p4], dim=1))
                f_p5 = self.fusion_p5(torch.cat([rgb_p5, th_p5], dim=1))

        # --- Neck & Head (保持不变) ---
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
        outputs.sort(key=lambda x: x.shape[2], reverse=True)

        return self.head(outputs)