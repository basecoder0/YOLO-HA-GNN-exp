# utils/data_utils.py (更新Stage2Dataset)
import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class Stage2Dataset(Dataset):
    """第二阶段训练数据集 - 适配R-LiViT结构"""

    def __init__(self,
                 dataset_path: str,
                 features_dir: str,
                 risk_labels_dir: str,
                 split: str = 'train',  # 'train' 或 'test'
                 max_samples: int = -1):
        """
        初始化第二阶段数据集

        Args:
            dataset_path: R-LiViT数据集路径
            features_dir: 第一阶段特征保存目录
            risk_labels_dir: 风险标签目录
            split: 数据集划分
            max_samples: 最大样本数（-1表示全部）
        """
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.features_dir = Path(features_dir)
        self.risk_labels_dir = Path(risk_labels_dir)
        self.split = split
        self.max_samples = max_samples

        # 收集样本
        self.samples = self._collect_samples()

        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"📊 第二阶段 {split} 数据集: {len(self.samples)} 个样本")

    def _collect_samples(self):
        """收集样本，根据数据集划分"""
        samples = []

        # 读取数据集划分文件
        split_file = self.dataset_path / "R-LiViT_RGB-T" / f"{self.split}.txt"

        if not split_file.exists():
            print(f"⚠️ 划分文件不存在: {split_file}")
            # 如果没有划分文件，使用所有特征文件
            feature_files = list(self.features_dir.glob("*_features.pt"))
            for feature_file in feature_files:
                sample_id = feature_file.stem.replace("_features", "")
                samples.append({
                    'sample_id': sample_id,
                    'feature_path': feature_file
                })
        else:
            # 读取划分文件中的序列
            with open(split_file, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]

            # 根据你的数据结构，我们需要映射序列到具体样本
            # 这里假设每个序列有12个时间戳（从你的数据集分析得知）
            for seq in sequences:
                for timestamp in ['02', '06', '10', '14', '18', '22', '26', '30', '34', '38']:
                    # 构建样本ID - 需要根据你的实际命名规则调整
                    sample_id = f"{seq}_{timestamp}"

                    # 检查特征文件是否存在
                    feature_path = self.features_dir / f"sample_{sample_id}_features.pt"

                    # 检查风险标签是否存在
                    risk_path = self.risk_labels_dir / f"{sample_id}_risk.npy"

                    if feature_path.exists() and risk_path.exists():
                        samples.append({
                            'sample_id': sample_id,
                            'feature_path': feature_path,
                            'risk_path': risk_path,
                            'sequence': seq,
                            'timestamp': timestamp
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # 加载特征
            features = torch.load(sample['feature_path'], map_location='cpu')

            # 加载风险标签
            risk_label = np.load(sample['risk_path'])
            risk_label = torch.tensor(risk_label, dtype=torch.float32)

            # 准备数据
            data = {
                'scene_features': features['scene_features'],  # [256]
                'sample_id': sample['sample_id'],
                'risk_label': risk_label
            }

            # 如果有节点特征和检测结果，也包含进来
            if 'node_features' in features and features['node_features'] is not None:
                data['node_features'] = features['node_features']

            if 'stage1_detections' in features and features['stage1_detections'] is not None:
                data['stage1_detections'] = features['stage1_detections']

            return data

        except Exception as e:
            print(f"❌ 加载样本 {sample['sample_id']} 失败: {e}")
            # 返回空样本
            return {
                'scene_features': torch.zeros(256),
                'sample_id': sample['sample_id'],
                'risk_label': torch.tensor(0.0, dtype=torch.float32)
            }

    def get_sample_info(self, idx):
        """获取样本的详细信息"""
        sample = self.samples[idx]
        info_path = self.features_dir / f"sample_{sample['sample_id']}_meta.json"

        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return None