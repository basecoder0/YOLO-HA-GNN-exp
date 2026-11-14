import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class LiDARBEVGenerator(nn.Module):
    """
    LiDAR点云到BEV特征图生成器
    将N×4点云转换为多通道BEV特征图
    """

    def __init__(self,
                 bev_size: Tuple[int, int] = (128, 128),  # BEV网格大小 [宽, 高]
                 bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),  # [x_min, x_max, y_min, y_max]
                 num_height_slices: int = 5,  # 高度切片数量
                 max_points: int = 50000,  # 最大处理点数
                 output_channels: int = 5):  # 输出通道数
        super().__init__()

        self.bev_size = bev_size
        self.bev_range = bev_range
        self.num_height_slices = num_height_slices
        self.max_points = max_points
        self.output_channels = output_channels

        # 网格分辨率
        self.grid_resolution = (
            (bev_range[1] - bev_range[0]) / bev_size[0],
            (bev_range[3] - bev_range[2]) / bev_size[1]
        )

        # BEV后处理网络（将原始BEV转换为高级特征）
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(num_height_slices + 1, 32, 3, padding=1),  # 输入: 高度切片 + 密度
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 1),  # 输出指定通道数
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        将点云转换为BEV特征图

        Args:
            points: [B, N, 4] 或 [N, 4] 点云，格式为 (x, y, z, intensity)

        Returns:
            BEV特征图: [B, C, H, W] 或 [C, H, W]
        """
        if points.dim() == 2:
            # 单样本: [N, 4] -> [1, N, 4]
            points = points.unsqueeze(0)
            batch_size = 1
            single_sample = True
        else:
            batch_size = points.shape[0]
            single_sample = False

        # 确保点云数量不超过最大值
        if points.shape[1] > self.max_points:
            # 随机采样
            indices = torch.randperm(points.shape[1])[:self.max_points]
            points = points[:, indices]

        device = points.device
        batch_bev_features = []

        for b in range(batch_size):
            batch_points = points[b]  # [N, 4]

            # 1. 转换为BEV网格坐标
            x_coords = batch_points[:, 0]  # x
            y_coords = batch_points[:, 1]  # y
            z_coords = batch_points[:, 2]  # z
            intensities = batch_points[:, 3]  # 强度

            # 转换为网格索引
            grid_x = torch.floor((x_coords - self.bev_range[0]) / self.grid_resolution[0])
            grid_y = torch.floor((y_coords - self.bev_range[2]) / self.grid_resolution[1])

            # 确保在网格范围内
            valid_mask = (grid_x >= 0) & (grid_x < self.bev_size[0]) & \
                         (grid_y >= 0) & (grid_y < self.bev_size[1])

            if not valid_mask.any():
                # 如果没有有效点，创建零特征
                bev_map = torch.zeros((self.num_height_slices + 1, self.bev_size[1], self.bev_size[0]),
                                      device=device)
                batch_bev_features.append(bev_map.unsqueeze(0))
                continue

            grid_x = grid_x[valid_mask].long()
            grid_y = grid_y[valid_mask].long()
            z_coords = z_coords[valid_mask]
            intensities = intensities[valid_mask]

            # 2. 创建多通道BEV特征
            bev_channels = self.num_height_slices + 1  # 高度切片 + 密度

            # 初始化BEV图
            bev_map = torch.zeros(bev_channels, self.bev_size[1], self.bev_size[0], device=device)

            # 通道0: 点云密度
            unique_coords, counts = torch.unique(torch.stack([grid_y, grid_x], dim=1), dim=0, return_counts=True)
            bev_map[0, unique_coords[:, 0], unique_coords[:, 1]] = torch.log1p(counts.float()) / torch.log1p(
                torch.tensor(100.0, device=device))

            # 通道1-5: 高度切片特征
            # 归一化z坐标到[0, 1]范围
            z_min, z_max = -5.0, 5.0  # R-LiViT中z范围大约是[-7.6, 0]
            z_normalized = (z_coords - z_min) / (z_max - z_min)
            z_normalized = torch.clamp(z_normalized, 0.0, 1.0)

            # 分配高度切片
            for i in range(self.num_height_slices):
                height_threshold = (i + 1) / self.num_height_slices
                in_slice = z_normalized <= height_threshold

                if in_slice.any():
                    slice_coords_x = grid_x[in_slice]
                    slice_coords_y = grid_y[in_slice]
                    slice_intensities = intensities[in_slice]

                    # 使用最大强度作为该切片的特征
                    unique_slice_coords, indices = torch.unique(
                        torch.stack([slice_coords_y, slice_coords_x], dim=1),
                        dim=0,
                        return_inverse=True
                    )
                    max_intensities = torch.zeros(unique_slice_coords.shape[0], device=device)

                    for idx in range(len(unique_slice_coords)):
                        mask = indices == idx
                        if mask.any():
                            max_intensities[idx] = slice_intensities[mask].max()

                    bev_map[i + 1, unique_slice_coords[:, 0], unique_slice_coords[:, 1]] = \
                        max_intensities / 255.0  # 归一化强度到[0, 1]

            # 3. 通过BEV编码器
            bev_map = self.bev_encoder(bev_map.unsqueeze(0))  # [1, C, H, W]
            batch_bev_features.append(bev_map)

        # 合并批次
        if single_sample:
            return batch_bev_features[0].squeeze(0)  # [C, H, W]
        else:
            return torch.cat(batch_bev_features, dim=0)  # [B, C, H, W]

    @torch.no_grad()
    def visualize_bev(self, points: torch.Tensor) -> np.ndarray:
        """可视化BEV特征（用于调试）"""
        bev_features = self.forward(points)

        if bev_features.dim() == 4:
            bev_features = bev_features[0]  # 取第一个批次

        # 将每个通道归一化到[0, 1]
        bev_norm = bev_features.cpu().numpy()
        for i in range(bev_norm.shape[0]):
            channel_min = bev_norm[i].min()
            channel_max = bev_norm[i].max()
            if channel_max > channel_min:
                bev_norm[i] = (bev_norm[i] - channel_min) / (channel_max - channel_min)

        return bev_norm


class MockLiDARProcessor(nn.Module):
    """
    模拟LiDAR处理器（用于测试或当没有真实LiDAR数据时）
    """

    def __init__(self, output_channels: int = 5, bev_size: Tuple[int, int] = (128, 128)):
        super().__init__()

        self.output_channels = output_channels
        self.bev_size = bev_size

        # 简单的卷积网络模拟BEV特征
        self.processor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(bev_size)
        )

    def forward(self, dummy_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成模拟的BEV特征

        Args:
            dummy_input: 可选输入，形状应为[B, 1, H, W]或[B, C, H, W]

        Returns:
            模拟BEV特征: [B, C, H, W]
        """
        if dummy_input is not None:
            # 如果有输入，确保它是正确的形状
            if dummy_input.dim() == 3:
                dummy_input = dummy_input.unsqueeze(1)  # [B, 1, H, W]
            elif dummy_input.dim() == 4 and dummy_input.shape[1] != 1:
                # 如果是多通道，取平均作为单通道
                dummy_input = dummy_input.mean(dim=1, keepdim=True)

            # 调整大小到BEV尺寸
            if dummy_input.shape[2:] != self.bev_size:
                dummy_input = nn.functional.interpolate(
                    dummy_input,
                    size=self.bev_size,
                    mode='bilinear',
                    align_corners=False
                )
        else:
            # 生成随机模拟数据
            batch_size = 1 if not torch.is_tensor(dummy_input) else dummy_input.shape[0]
            dummy_input = torch.randn(batch_size, 1, self.bev_size[0], self.bev_size[1])

        return self.processor(dummy_input)


def load_lidar_bin_file(bin_path: str, max_points: int = 100000) -> torch.Tensor:
    """
    从.bin文件加载LiDAR点云

    Args:
        bin_path: .bin文件路径
        max_points: 最大加载点数

    Returns:
        点云张量: [N, 4] (x, y, z, intensity)
    """
    try:
        # 读取二进制文件
        points = np.fromfile(bin_path, dtype=np.float32)

        # 重塑为N×4
        if len(points) % 4 == 0:
            points = points.reshape(-1, 4)
        else:
            # 如果不是4的倍数，尝试其他格式
            if len(points) % 3 == 0:
                points = points.reshape(-1, 3)
                # 添加强度通道（设为1.0）
                intensity = np.ones((points.shape[0], 1), dtype=np.float32)
                points = np.concatenate([points, intensity], axis=1)
            else:
                raise ValueError(f"Unexpected point cloud format in {bin_path}")

        # 限制点数
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[indices]

        return torch.from_numpy(points).float()

    except Exception as e:
        print(f"Error loading LiDAR file {bin_path}: {e}")
        # 返回空点云
        return torch.zeros((100, 4), dtype=torch.float32)