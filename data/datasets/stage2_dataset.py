import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from data.datasets.r_livit_dataset import RLiViTDataset, RLivitDataLoader
from utils.risk_ground_truth import RiskGroundTruthCalculator


class Stage2Dataset(Dataset):
    """
    第二阶段训练数据集
    集成：1) 第一阶段特征 2) LiDAR数据 3) 风险真值
    """

    def __init__(self,
                 dataset_path: str,
                 stage1_features_dir: str,
                 risk_labels_path: str,
                 split: str = 'train',
                 modalities: List[str] = ['rgb', 'thermal', 'lidar'],
                 use_lidar: bool = True,
                 lidar_config: Dict = None,
                 max_samples: int = -1,
                 transform=None):

        self.dataset_path = dataset_path
        self.stage1_features_dir = stage1_features_dir
        self.risk_labels_path = risk_labels_path
        self.split = split
        self.modalities = modalities
        self.use_lidar = use_lidar
        self.max_samples = max_samples
        self.transform = transform

        # LiDAR配置
        if lidar_config is None:
            lidar_config = {
                'bev_size': (128, 128),
                'bev_range': (-50, 50, -50, 50),
                'num_height_slices': 5,
                'max_points': 50000,
                'output_channels': 5
            }
        self.lidar_config = lidar_config

        # 加载基础数据集（用于获取LiDAR和原始图像）
        self.base_dataset = RLiViTDataset(
            dataset_path=dataset_path,
            split=split,
            modalities=['rgb', 'thermal'],
            use_lidar=use_lidar,
            lidar_config=lidar_config,
            transform=transform
        )

        # 加载第一阶段特征
        self.feature_samples = self._load_stage1_features()

        # 加载风险标签
        self.risk_labels = self._load_risk_labels()

        # 风险计算器（用于实时计算风险真值）
        self.risk_calculator = RiskGroundTruthCalculator()

        # 创建样本映射
        self.samples = self._create_sample_mapping()

        print(f"📊 Stage2 Dataset - {split}:")
        print(f"  Base dataset: {len(self.base_dataset)} samples")
        print(f"  Feature samples: {len(self.feature_samples)} samples")
        print(f"  Risk labels: {len(self.risk_labels)} samples")
        print(f"  Final samples: {len(self.samples)} samples")
        print(f"  Use LiDAR: {use_lidar}")

    def _load_stage1_features(self) -> List[Dict]:
        """加载第一阶段提取的特征"""
        features = []
        features_dir = Path(self.stage1_features_dir) / self.split

        if not features_dir.exists():
            print(f"⚠️  Stage1 features directory not found: {features_dir}")
            print("⚠️  Will use simulated features for testing")
            return self._create_simulated_features()

        feature_files = list(features_dir.glob("*.pkl"))

        if self.max_samples > 0:
            feature_files = feature_files[:self.max_samples]

        print(f"📁 Loading {len(feature_files)} feature files from {features_dir}")

        for feature_file in feature_files:
            try:
                with open(feature_file, 'rb') as f:
                    feature_data = pickle.load(f)

                sample_id = feature_file.stem

                # 确保特征数据包含必要的信息
                if 'detections' not in feature_data:
                    feature_data['detections'] = np.zeros((0, 6))
                if 'node_features' not in feature_data:
                    feature_data['node_features'] = np.zeros((0, 256))
                if 'scene_features' not in feature_data:
                    feature_data['scene_features'] = np.zeros((256,))

                features.append({
                    'id': sample_id,
                    'features': feature_data,
                    'num_detections': len(feature_data['detections'])
                })

            except Exception as e:
                print(f"Error loading feature file {feature_file}: {e}")

        return features

    def _create_simulated_features(self) -> List[Dict]:
        """创建模拟特征（用于测试）"""
        print("🔧 Creating simulated stage1 features for testing...")

        features = []
        # 使用基础数据集中的样本ID
        for i, sample in enumerate(self.base_dataset.samples[:100]):  # 限制100个样本
            sample_id = sample['id']

            # 模拟检测结果
            num_detections = np.random.randint(1, 10)
            detections = np.zeros((num_detections, 6))

            for j in range(num_detections):
                # [x1, y1, x2, y2, confidence, class_id]
                x1 = np.random.randint(0, 1200)
                y1 = np.random.randint(0, 700)
                x2 = x1 + np.random.randint(20, 200)
                y2 = y1 + np.random.randint(20, 200)
                conf = np.random.uniform(0.3, 0.95)
                class_id = np.random.choice([0, 1, 2])
                detections[j] = [x1, y1, x2, y2, conf, class_id]

            # 模拟节点特征
            node_features = np.random.randn(num_detections, 256)

            # 模拟场景特征
            scene_features = np.random.randn(256)

            features.append({
                'id': sample_id,
                'features': {
                    'detections': detections,
                    'node_features': node_features,
                    'scene_features': scene_features,
                    'sequence_id': sample['sequence_id'],
                    'frame_id': sample['frame_id']
                },
                'num_detections': num_detections
            })

        return features

    def _load_risk_labels(self) -> Dict:
        """加载风险标签"""
        risk_file = Path(self.risk_labels_path) / f'risk_labels_{self.split}.json'

        if not risk_file.exists():
            print(f"⚠️  Risk labels file not found: {risk_file}")
            print("⚠️  Will calculate risk labels on the fly")
            return {}

        try:
            with open(risk_file, 'r', encoding='utf-8') as f:
                risk_labels = json.load(f)

            print(f"✅ Loaded risk labels for {len(risk_labels)} samples")
            return risk_labels

        except Exception as e:
            print(f"Error loading risk labels: {e}")
            return {}

    def _create_sample_mapping(self) -> List[Dict]:
        """创建样本映射，将特征、基础数据和风险标签关联起来"""
        samples = []

        # 使用特征样本作为基础
        for feature_sample in self.feature_samples:
            sample_id = feature_sample['id']

            # 查找对应的基础数据样本
            base_sample_idx = -1
            for i, base_sample in enumerate(self.base_dataset.samples):
                if base_sample['id'] == sample_id:
                    base_sample_idx = i
                    break

            if base_sample_idx == -1:
                # 如果没有找到对应的基础样本，跳过
                continue

            # 获取风险标签
            risk_info = self.risk_labels.get(sample_id, {})

            sample = {
                'id': sample_id,
                'feature_data': feature_sample['features'],
                'base_sample_idx': base_sample_idx,
                'risk_labels': risk_info.get('risk_scores', []),
                'num_targets': risk_info.get('num_targets', feature_sample['num_detections'])
            }

            samples.append(sample)

        # 限制样本数量
        if self.max_samples > 0 and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        return samples

    def _calculate_risk_ground_truth(self, sample_id: str, detections: np.ndarray) -> np.ndarray:
        """计算风险真值（如果预计算的标签不存在）"""
        # 首先尝试使用预计算的标签
        if sample_id in self.risk_labels:
            risk_scores = np.array(self.risk_labels[sample_id].get('risk_scores', []))
            if len(risk_scores) > 0:
                return risk_scores

        # 如果没有预计算的标签，实时计算
        detection_list = []
        for det in detections:
            detection_list.append({
                'bbox': det[:4].tolist(),
                'class_id': int(det[5]),
                'confidence': float(det[4]),
                'occlusion': 0.0  # 假设没有遮挡信息
            })

        risk_scores = self.risk_calculator.calculate_risk_for_frame(
            detections=detection_list,
            lidar_data=None,
            frame_info={'sample_id': sample_id}
        )

        return risk_scores

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        sample_id = sample['id']

        data = {
            'sample_id': sample_id,
            'sequence_id': sample['feature_data'].get('sequence_id', 'unknown'),
            'frame_id': sample['feature_data'].get('frame_id', 'unknown'),
            'num_targets': sample['num_targets']
        }

        # 1. 第一阶段特征
        feature_data = sample['feature_data']

        # 检测结果
        detections = feature_data.get('detections', np.zeros((0, 6)))
        data['detections'] = torch.from_numpy(detections).float()

        # 节点特征（GNN输入）
        node_features = feature_data.get('node_features', np.zeros((0, 256)))
        data['node_features'] = torch.from_numpy(node_features).float()

        # 场景特征
        scene_features = feature_data.get('scene_features', np.zeros((256,)))
        data['scene_features'] = torch.from_numpy(scene_features).float()

        # 2. LiDAR特征（从基础数据集获取）
        base_sample = self.base_dataset[sample['base_sample_idx']]
        if 'lidar' in base_sample and torch.is_tensor(base_sample['lidar']):
            data['lidar_features'] = base_sample['lidar'].float()
        else:
            # 创建空的LiDAR特征
            data['lidar_features'] = torch.zeros(
                (self.lidar_config['output_channels'],
                 self.lidar_config['bev_size'][1],
                 self.lidar_config['bev_size'][0])
            ).float()

        # 3. 风险真值
        risk_scores = self._calculate_risk_ground_truth(sample_id, detections)
        data['risk_ground_truth'] = torch.from_numpy(risk_scores).float()

        # 4. 原始图像（可选，用于可视化）
        if 'rgb' in base_sample:
            data['rgb_image'] = base_sample['rgb'].float()
        if 'thermal' in base_sample:
            data['thermal_image'] = base_sample['thermal'].float()

        # 5. 原始边界框（用于验证）
        if 'bboxes' in base_sample:
            data['original_bboxes'] = torch.from_numpy(base_sample['bboxes']).float()
        if 'labels' in base_sample:
            data['original_labels'] = torch.from_numpy(base_sample['labels']).float()

        return data


def collate_stage2_batch(batch: List[Dict]) -> Dict:
    """
    第二阶段数据批处理函数
    处理可变长度的检测结果和节点特征
    """
    collated = {}

    # 获取所有键
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        if key in ['detections', 'node_features', 'risk_ground_truth']:
            # 对于可变长度的张量，保持为列表
            collated[key] = values

        elif key in ['lidar_features', 'scene_features', 'rgb_image', 'thermal_image']:
            # 对于固定形状的张量，堆叠
            if all(torch.is_tensor(v) for v in values):
                try:
                    collated[key] = torch.stack(values)
                except RuntimeError as e:
                    print(f"Warning: Could not stack {key}: {e}")
                    collated[key] = values

        elif key in ['sample_id', 'sequence_id', 'frame_id']:
            # 字符串ID
            collated[key] = values

        elif key in ['num_targets']:
            # 数值
            collated[key] = torch.tensor(values, dtype=torch.int32)

        elif torch.is_tensor(values[0]):
            # 其他张量
            try:
                collated[key] = torch.stack(values)
            except:
                collated[key] = values

        else:
            # 其他类型
            collated[key] = values

    return collated


class Stage2DataLoader:
    """第二阶段数据加载器包装类"""

    @staticmethod
    def create(dataset_path: str,
               stage1_features_dir: str,
               risk_labels_path: str,
               split: str = 'train',
               batch_size: int = 4,
               num_workers: int = 4,
               shuffle: bool = True,
               max_samples: int = -1):
        # 创建数据集
        dataset = Stage2Dataset(
            dataset_path=dataset_path,
            stage1_features_dir=stage1_features_dir,
            risk_labels_path=risk_labels_path,
            split=split,
            use_lidar=True,
            max_samples=max_samples
        )

        # 创建数据加载器
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_stage2_batch,
            pin_memory=True
        )

        return loader, dataset


def test_stage2_dataset():
    """测试第二阶段数据集"""
    print("🧪 Testing Stage2 Dataset...")

    # 测试配置
    dataset_path = "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT"
    stage1_features_dir = "./stage1_features"  # 假设第一阶段特征保存在这里
    risk_labels_path = "./risk_labels"

    # 创建数据集
    dataset = Stage2Dataset(
        dataset_path=dataset_path,
        stage1_features_dir=stage1_features_dir,
        risk_labels_path=risk_labels_path,
        split='train',
        use_lidar=True,
        max_samples=10  # 只测试10个样本
    )

    print(f"\n📋 Dataset info:")
    print(f"  Total samples: {len(dataset)}")

    if len(dataset) > 0:
        # 测试一个样本
        sample = dataset[0]
        print(f"\n🔍 Sample 0:")
        print(f"  ID: {sample['sample_id']}")
        print(f"  Num detections: {len(sample['detections'])}")
        print(f"  Node features shape: {sample['node_features'].shape}")
        print(f"  Scene features shape: {sample['scene_features'].shape}")
        print(f"  LiDAR features shape: {sample['lidar_features'].shape}")
        print(f"  Risk ground truth shape: {sample['risk_ground_truth'].shape}")
        print(
            f"  Risk values: {sample['risk_ground_truth'][:3].tolist() if len(sample['risk_ground_truth']) > 3 else sample['risk_ground_truth'].tolist()}")

        # 测试批处理
        batch = [dataset[i] for i in range(min(3, len(dataset)))]
        collated = collate_stage2_batch(batch)

        print(f"\n📦 Collated batch:")
        for key, value in collated.items():
            if isinstance(value, list):
                print(f"  {key}: list with {len(value)} elements")
            elif torch.is_tensor(value):
                print(f"  {key}: tensor with shape {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")


if __name__ == "__main__":
    test_stage2_dataset()