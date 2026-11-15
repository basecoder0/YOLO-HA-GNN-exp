import torch
import torch.nn as nn


class DPAFM(nn.Module):
    """
    [no痛fusionversion] DPAFM
    corestrategyomit：Identity Mapping (恒waitmapping)
    trainingearly：Output = RGB (100% keepkeep RGB predict/pretrainingability/canability)
    trainingafter期：Output = RGB + beta * Thermal (autostudylearninfrared补full)
    """

    def __init__(self, in_channels):
        super().__init__()

        # 1. simpleoffeatureextract (针pair Thermal)
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # generateonea 0~1 ofnoteability掩code
        )

        # 2. 🔥 core：optionalstudylearnoffusion门控，initializationfor/as 0
        # thisnotice味着刚starttrainingwhen，Thermal branchcompletelybe屏蔽
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, f_rgb, f_thermal):
        # f_rgb: major路 (Main Path) - keeporiginalsample，notneed乘any东西！
        # f_thermal: 辅路 (Auxiliary Path)

        # calculation Thermal ofhaveeffectinformation
        thermal_feat = self.thermal_conv(f_thermal) * f_thermal

        # fusion公style：RGB + 0 * Thermal
        # like this Neck what is received is pure RGB feature，Recall 瞬spacefallbackto 0.48
        f_fused = f_rgb + self.weight * thermal_feat

        return f_fused