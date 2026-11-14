import torch
import torch.nn as nn


class DPAFM(nn.Module):
    """
    [无痛融合版] DPAFM
    核心策略：Identity Mapping (恒等映射)
    训练初期：Output = RGB (100% 保留 RGB 预训练能力)
    训练后期：Output = RGB + beta * Thermal (自动学习红外补充)
    """

    def __init__(self, in_channels):
        super().__init__()

        # 1. 简单的特征提取 (针对 Thermal)
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # 生成一个 0~1 的注意力掩码
        )

        # 2. 🔥 核心：可学习的融合门控，初始化为 0
        # 这意味着刚开始训练时，Thermal 分支完全被屏蔽
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, f_rgb, f_thermal):
        # f_rgb: 主路 (Main Path) - 保持原样，不要乘任何东西！
        # f_thermal: 辅路 (Auxiliary Path)

        # 计算 Thermal 的有效信息
        thermal_feat = self.thermal_conv(f_thermal) * f_thermal

        # 融合公式：RGB + 0 * Thermal
        # 这样 Neck 接收到的就是纯正的 RGB 特征，Recall 瞬间回到 0.48
        f_fused = f_rgb + self.weight * thermal_feat

        return f_fused