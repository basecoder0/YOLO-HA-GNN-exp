import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class LiDARBEVGenerator(nn.Module):
    """
    LiDARpoint cloudtoBEVfeatureimagegeneratedevice
    willN×4point cloudconvertfor/asmultiplechannelBEVfeatureimage
    """

    def __init__(self,
                 bev_size: Tuple[int, int] = (128, 128),  # BEVnetworkstrictlargesmall [宽, high]
                 bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),  # [x_min, x_max, y_min, y_max]
                 num_height_slices: int = 5,  # highdegreeslicecountamount
                 max_points: int = 50000,  # maximumprocessingpointcount
                 output_channels: int = 5):  # outputchannelcount
        super().__init__()

        self.bev_size = bev_size
        self.bev_range = bev_range
        self.num_height_slices = num_height_slices
        self.max_points = max_points
        self.output_channels = output_channels

        # networkstrictresolution
        self.grid_resolution = (
            (bev_range[1] - bev_range[0]) / bev_size[0],
            (bev_range[3] - bev_range[2]) / bev_size[1]
        )

        # BEVafterprocessingnetworknetwork（willoriginalBEVconvertfor/ashighlevelfeature）
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(num_height_slices + 1, 32, 3, padding=1),  # input: highdegreeslice + densedegree
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 1),  # outputpointdefinechannelcount
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        willpoint cloudconvertfor/asBEVfeatureimage

        Args:
            points: [B, N, 4] or [N, 4] point cloud，formatfor/as (x, y, z, intensity)

        Returns:
            BEVfeatureimage: [B, C, H, W] or [C, H, W]
        """
        if points.dim() == 2:
            # singlesample: [N, 4] -> [1, N, 4]
            points = points.unsqueeze(0)
            batch_size = 1
            single_sample = True
        else:
            batch_size = points.shape[0]
            single_sample = False

        # make surepoint cloudcountamountnotexceedmaximumvalue
        if points.shape[1] > self.max_points:
            # random采sample
            indices = torch.randperm(points.shape[1])[:self.max_points]
            points = points[:, indices]

        device = points.device
        batch_bev_features = []

        for b in range(batch_size):
            batch_points = points[b]  # [N, 4]

            # 1. convertfor/asBEVnetworkstrictcoordinate
            x_coords = batch_points[:, 0]  # x
            y_coords = batch_points[:, 1]  # y
            z_coords = batch_points[:, 2]  # z
            intensities = batch_points[:, 3]  # intensity

            # convertfor/asnetworkstrictindex
            grid_x = torch.floor((x_coords - self.bev_range[0]) / self.grid_resolution[0])
            grid_y = torch.floor((y_coords - self.bev_range[2]) / self.grid_resolution[1])

            # make sureat/innetworkstrictrangeinside
            valid_mask = (grid_x >= 0) & (grid_x < self.bev_size[0]) & \
                         (grid_y >= 0) & (grid_y < self.bev_size[1])

            if not valid_mask.any():
                # ifnothavehaveeffectpoint，createzerofeature
                bev_map = torch.zeros((self.num_height_slices + 1, self.bev_size[1], self.bev_size[0]),
                                      device=device)
                batch_bev_features.append(bev_map.unsqueeze(0))
                continue

            grid_x = grid_x[valid_mask].long()
            grid_y = grid_y[valid_mask].long()
            z_coords = z_coords[valid_mask]
            intensities = intensities[valid_mask]

            # 2. createmultiplechannelBEVfeature
            bev_channels = self.num_height_slices + 1  # highdegreeslice + densedegree

            # initializationBEVimage
            bev_map = torch.zeros(bev_channels, self.bev_size[1], self.bev_size[0], device=device)

            # channel0: point clouddensedegree
            unique_coords, counts = torch.unique(torch.stack([grid_y, grid_x], dim=1), dim=0, return_counts=True)
            bev_map[0, unique_coords[:, 0], unique_coords[:, 1]] = torch.log1p(counts.float()) / torch.log1p(
                torch.tensor(100.0, device=device))

            # channel1-5: highdegreeslicefeature
            # normalizezcoordinateto[0, 1]range
            z_min, z_max = -5.0, 5.0  # R-LiViTinzrangelargeapproximatelyis[-7.6, 0]
            z_normalized = (z_coords - z_min) / (z_max - z_min)
            z_normalized = torch.clamp(z_normalized, 0.0, 1.0)

            # score/divideconfigurehighdegreeslice
            for i in range(self.num_height_slices):
                height_threshold = (i + 1) / self.num_height_slices
                in_slice = z_normalized <= height_threshold

                if in_slice.any():
                    slice_coords_x = grid_x[in_slice]
                    slice_coords_y = grid_y[in_slice]
                    slice_intensities = intensities[in_slice]

                    # use maximum intensity as the value for that slicefeature
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
                        max_intensities / 255.0  # normalizeintensityto[0, 1]

            # 3. throughBEVencodingdevice
            bev_map = self.bev_encoder(bev_map.unsqueeze(0))  # [1, C, H, W]
            batch_bev_features.append(bev_map)

        # combineandbatchminor
        if single_sample:
            return batch_bev_features[0].squeeze(0)  # [C, H, W]
        else:
            return torch.cat(batch_bev_features, dim=0)  # [B, C, H, W]

    @torch.no_grad()
    def visualize_bev(self, points: torch.Tensor) -> np.ndarray:
        """visualizationBEVfeature（fordebug）"""
        bev_features = self.forward(points)

        if bev_features.dim() == 4:
            bev_features = bev_features[0]  # getfirstabatchminor

        # normalize each channel to[0, 1]
        bev_norm = bev_features.cpu().numpy()
        for i in range(bev_norm.shape[0]):
            channel_min = bev_norm[i].min()
            channel_max = bev_norm[i].max()
            if channel_max > channel_min:
                bev_norm[i] = (bev_norm[i] - channel_min) / (channel_max - channel_min)

        return bev_norm


class MockLiDARProcessor(nn.Module):
    """
    simulateLiDARprocessingdevice（fortestorwhennothavetrueLiDARdatawhen）
    """

    def __init__(self, output_channels: int = 5, bev_size: Tuple[int, int] = (128, 128)):
        super().__init__()

        self.output_channels = output_channels
        self.bev_size = bev_size

        # simpleofconvolutionnetworknetworksimulateBEVfeature
        self.processor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(bev_size)
        )

    def forward(self, dummy_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        generatesimulateofBEVfeature

        Args:
            dummy_input: optionalinput，shapeshouldfor/as[B, 1, H, W]or[B, C, H, W]

        Returns:
            simulateBEVfeature: [B, C, H, W]
        """
        if dummy_input is not None:
            # ifhaveinput，make sureitispositivesureofshape
            if dummy_input.dim() == 3:
                dummy_input = dummy_input.unsqueeze(1)  # [B, 1, H, W]
            elif dummy_input.dim() == 4 and dummy_input.shape[1] != 1:
                # ifismultiplechannel，getaverageassinglechannel
                dummy_input = dummy_input.mean(dim=1, keepdim=True)

            # adjustmentlargesmalltoBEVsize
            if dummy_input.shape[2:] != self.bev_size:
                dummy_input = nn.functional.interpolate(
                    dummy_input,
                    size=self.bev_size,
                    mode='bilinear',
                    align_corners=False
                )
        else:
            # generaterandomsimulatedata
            batch_size = 1 if not torch.is_tensor(dummy_input) else dummy_input.shape[0]
            dummy_input = torch.randn(batch_size, 1, self.bev_size[0], self.bev_size[1])

        return self.processor(dummy_input)


def load_lidar_bin_file(bin_path: str, max_points: int = 100000) -> torch.Tensor:
    """
    from.binfileloadingLiDARpoint cloud

    Args:
        bin_path: .binfilepath
        max_points: maximumloadingpointcount

    Returns:
        point cloudtensoramount: [N, 4] (x, y, z, intensity)
    """
    try:
        # read二advancecontrolfile
        points = np.fromfile(bin_path, dtype=np.float32)

        # weight塑for/asN×4
        if len(points) % 4 == 0:
            points = points.reshape(-1, 4)
        else:
            # ifnotis4of倍count，tryitsotherformat
            if len(points) % 3 == 0:
                points = points.reshape(-1, 3)
                # addintensitychannel（setfor/as1.0）
                intensity = np.ones((points.shape[0], 1), dtype=np.float32)
                points = np.concatenate([points, intensity], axis=1)
            else:
                raise ValueError(f"Unexpected point cloud format in {bin_path}")

        # limitpointcount
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[indices]

        return torch.from_numpy(points).float()

    except Exception as e:
        print(f"Error loading LiDAR file {bin_path}: {e}")
        # returnemptypoint cloud
        return torch.zeros((100, 4), dtype=torch.float32)