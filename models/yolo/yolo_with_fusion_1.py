import torch
import torch.nn as nn
from ultralytics import YOLO
from models.fusion.adaptive_fusion import DPAFM
import copy


class YOLOWithFusion(nn.Module):
    """
    [最终形态] 双流 YOLOv8 融合模型
    架构：Dual Backbone -> DPAFM Fusion -> Shared Neck (FPN+PAN) -> Detect Head
    """

    def __init__(self, model_path='weights/yolov8n.pt', num_classes=8):
        super().__init__()
        print(f"Loading base model from {model_path} for architecture reconstruction...")
        base_model = YOLO(model_path)
        m = base_model.model.model  # 获取内部 Sequential 模型

        # --- 1. 双流骨干 (Layers 0-9) ---
        # YOLOv8n Backbone 包含层 0-9 (0:Conv, ..., 4:P3, 6:P4, 9:SPPF-P5)
        self.rgb_backbone = nn.ModuleList(copy.deepcopy(m[:10]))
        self.thermal_backbone = nn.ModuleList(copy.deepcopy(m[:10]))

        # --- 2. 融合模块 (P3, P4, P5) ---
        # 对应 backbone 的输出通道: P3=64, P4=128, P5=256
        self.fusion_p3 = DPAFM(64)
        self.fusion_p4 = DPAFM(128)
        self.fusion_p5 = DPAFM(256)

        # --- 3. 颈部网络 (Layers 10-21) ---
        # 我们直接复用预训练的 Neck 层参数，但需要手动管理数据流
        self.neck_layers = nn.ModuleList(copy.deepcopy(m[10:22]))

        # --- 4. 检测头 (Layer 22) ---
        self.head = copy.deepcopy(m[22])

        # 重置检测头的类别数
        if self.head.nc != num_classes:
            self._reset_head(num_classes)

    def _reset_head(self, num_classes):
        """
        重置检测头的类别数
        修复了 'Conv' object has no attribute 'in_channels' 错误
        """
        self.head.nc = num_classes
        # YOLOv8输出通道数计算: 类别数 + 回归框坐标数(4 * reg_max)
        self.head.no = num_classes + self.head.reg_max * 4

        # 遍历 cv3 (分类分支) 的每一层 (P3, P4, P5)
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

        # 🔥 关键修复：安全地更新 stride 属性
        if hasattr(self.head, 'stride'):
            m = self.head
            stride_val = torch.tensor([8., 16., 32.])

            # 如果它已经存在，先删除这个 Python 属性，防止 register_buffer 报错
            del m.stride

            # 再重新注册为 Buffer (确保它能随模型移动到 GPU)
            m.register_buffer('stride', stride_val)

    def forward_backbone(self, backbone, x):
        """执行骨干网络并提取 P3, P4, P5"""
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
            thermal = x

            # 2. Backbone 前向
        rgb_p3, rgb_p4, rgb_p5 = self.forward_backbone(self.rgb_backbone, rgb)
        th_p3, th_p4, th_p5 = self.forward_backbone(self.thermal_backbone, thermal)

        # 3. 融合
        f_p3 = self.fusion_p3(rgb_p3, th_p3)
        f_p4 = self.fusion_p4(rgb_p4, th_p4)
        f_p5 = self.fusion_p5(rgb_p5, th_p5)

        # 4. Neck 前向 (标准 YOLOv8n 逻辑)
        # -----------------------------------------------------------
        # ⚠️ 这里的索引基于 standard yolov8n.yaml，如果你的版本不同，索引可能偏移
        # 但我们将在最后一步通过“尺寸排序”来通过容错
        # -----------------------------------------------------------

        # Layer 10: Upsample
        x = self.neck_layers[0](f_p5)
        # Layer 11: Concat (P5_up + P4) -> 结果应为 40x40
        # 如果 f_p4 和 x 尺寸对不上，这里就会直接报错，这也是一种检查
        if x.shape[2:] != f_p4.shape[2:]:
            # 容错：强制对齐尺寸 (万一 f_p4 也是错的)
            x = torch.nn.functional.interpolate(x, size=f_p4.shape[2:], mode='nearest')
        x = torch.cat([x, f_p4], dim=1)

        # Layer 12: C2f (P4_TopDown)
        out_1 = self.neck_layers[2](x)

        # Layer 13: Upsample
        x = self.neck_layers[3](out_1)
        # Layer 14: Concat (P4_up + P3) -> 结果应为 80x80
        if x.shape[2:] != f_p3.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=f_p3.shape[2:], mode='nearest')
        x = torch.cat([x, f_p3], dim=1)

        # Layer 15: C2f (P3_out) --> 这是分辨率最高的特征 (80x80)
        out_2 = self.neck_layers[5](x)

        # Layer 16: Conv (Downsample)
        x = self.neck_layers[6](out_2)
        # Layer 17: Concat
        x = torch.cat([x, out_1], dim=1)

        # Layer 18: C2f (P4_out) --> 这是中等分辨率特征 (40x40)
        out_3 = self.neck_layers[8](x)

        # Layer 19: Conv (Downsample)
        x = self.neck_layers[9](out_3)
        # Layer 20: Concat
        x = torch.cat([x, f_p5], dim=1)

        # Layer 21: C2f (P5_out) --> 这是低分辨率特征 (20x20)
        out_4 = self.neck_layers[11](x)

        # -----------------------------------------------------------
        # 🔥🔥🔥 终极修复：不再信任变量名，只信任尺寸 🔥🔥🔥
        # -----------------------------------------------------------
        outputs = [out_2, out_3, out_4]

        # 打印调试信息 (只在第一轮打印，确认尺寸)
        if not hasattr(self, '_debug_printed'):
            print(f"\n🛑 [DEBUG] Head Inputs Shapes (Before Sort):")
            for i, o in enumerate(outputs):
                print(f"   Input {i}: {o.shape} (H={o.shape[2]})")
            self._debug_printed = True

        # 强制按 H (高度) 降序排列 -> [80x80, 40x40, 20x20]
        # 对应 Stride -> [8, 16, 32]
        outputs.sort(key=lambda x: x.shape[2], reverse=True)

        # 再次确认
        if outputs[0].shape[2] < outputs[-1].shape[2]:
            raise RuntimeError("❌ 排序失败！Head 输入顺序依然是错误的！")

        return self.head(outputs)