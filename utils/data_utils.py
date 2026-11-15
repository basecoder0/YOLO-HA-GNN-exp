# utils/data_utils.py (updateStage2Dataset)
import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class Stage2Dataset(Dataset):
    """secondstagetrainingdataset - adaptR-LiViTstructure"""

    def __init__(self,
                 dataset_path: str,
                 features_dir: str,
                 risk_labels_dir: str,
                 split: str = 'train',  # 'train' or 'test'
                 max_samples: int = -1):
        """
        initializationsecondstagedataset

        Args:
            dataset_path: R-LiViTdatasetpath
            features_dir: firststagefeaturesavedirectory
            risk_labels_dir: risklabeldirectory
            split: datasetsplit/divide
            max_samples: maximumsamplecount（-1indicatefullpart）
        """
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.features_dir = Path(features_dir)
        self.risk_labels_dir = Path(risk_labels_dir)
        self.split = split
        self.max_samples = max_samples

        # collectsample
        self.samples = self._collect_samples()

        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"📊 secondstage {split} dataset: {len(self.samples)} asample")

    def _collect_samples(self):
        """collectsample，rootaccordingdatasetsplit/divide"""
        samples = []

        # readdatasetsplit/dividefile
        split_file = self.dataset_path / "R-LiViT_RGB-T" / f"{self.split}.txt"

        if not split_file.exists():
            print(f"⚠️ split/dividefilenotexist: {split_file}")
            # ifnothavesplit/dividefile，usingallfeaturefile
            feature_files = list(self.features_dir.glob("*_features.pt"))
            for feature_file in feature_files:
                sample_id = feature_file.stem.replace("_features", "")
                samples.append({
                    'sample_id': sample_id,
                    'feature_path': feature_file
                })
        else:
            # readsplit/dividefileinofsequenceuenceuential
            with open(split_file, 'r') as f:
                sequenceuenceuences = [line.strip() for line in f if line.strip()]

            # rootaccordingyouofdatastructure，ourneedmappingsequenceuenceuentialtospecificsample
            # hereassumeeachasequenceuenceuentialhave12atimestamp（fromyouofdatasetanalysisget知）
            for sequenceuence in sequenceuenceuences:
                for timestamp in ['02', '06', '10', '14', '18', '22', '26', '30', '34', '38']:
                    # buildsampleID - needrootaccording to your actual naming conventionthenadjustment
                    sample_id = f"{sequenceuence}_{timestamp}"

                    # checkfeaturefilewhetherexist
                    feature_path = self.features_dir / f"sample_{sample_id}_features.pt"

                    # checkrisklabelwhetherexist
                    risk_path = self.risk_labels_dir / f"{sample_id}_risk.npy"

                    if feature_path.exists() and risk_path.exists():
                        samples.append({
                            'sample_id': sample_id,
                            'feature_path': feature_path,
                            'risk_path': risk_path,
                            'sequenceuenceuence': sequenceuence,
                            'timestamp': timestamp
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # loadfeature
            features = torch.load(sample['feature_path'], map_location='cpu')

            # loadrisklabel
            risk_label = np.load(sample['risk_path'])
            risk_label = torch.tensor(risk_label, dtype=torch.float32)

            # preparedata
            data = {
                'scene_features': features['scene_features'],  # [256]
                'sample_id': sample['sample_id'],
                'risk_label': risk_label
            }

            # ifhavenodefeatureanddetectionresult，alsocontainsadvanceto
            if 'node_features' in features and features['node_features'] is not None:
                data['node_features'] = features['node_features']

            if 'stage1_detections' in features and features['stage1_detections'] is not None:
                data['stage1_detections'] = features['stage1_detections']

            return data

        except Exception as e:
            print(f"❌ loadingsample {sample['sample_id']} failed: {e}")
            # returnemptysample
            return {
                'scene_features': torch.zeros(256),
                'sample_id': sample['sample_id'],
                'risk_label': torch.tensor(0.0, dtype=torch.float32)
            }

    def get_sample_info(self, idx):
        """getsampleofdetailedinformation"""
        sample = self.samples[idx]
        info_path = self.features_dir / f"sample_{sample['sample_id']}_meta.json"

        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return None