import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import math


class RiskGroundTruthCalculator:
    """
    风险真值计算器
    根据设计文档中的风险评估公式计算风险真值
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = {
                # 目标类型权重
                'class_weights': {
                    0: 0.7,  # car/vehicle (默认值，可根据需要调整)
                    1: 0.8,  # bicycle (非机动车)
                    2: 1.0  # person (行人，最高风险)
                },

                # 距离风险参数
                'distance_threshold': 50.0,  # 米，最大有效距离
                'min_distance': 2.0,  # 米，最小安全距离

                # 速度风险参数
                'max_speed': 60.0,  # km/h，最大速度
                'danger_speed_threshold': 30.0,  # km/h，危险速度阈值

                # 遮挡风险参数
                'severe_occlusion_threshold': 0.5,  # 严重遮挡阈值

                # 时间参数（帧率）
                'fps': 2.0,  # 帧率（R-LiViT中约2-5fps）

                # 冲突对权重
                'conflict_pair_weights': {
                    'person_vehicle': 1.0,  # 人-车冲突
                    'bicycle_vehicle': 0.8,  # 非机动车-车冲突
                    'person_bicycle': 0.6,  # 人-非机动车冲突
                    'vehicle_vehicle': 0.3,  # 车-车冲突
                    'same_class': 0.1  # 同类别冲突
                }
            }
        self.config = config

        print("✅ Risk ground truth calculator initialized")

    def calculate_risk_for_frame(self,
                                 detections: List[Dict],
                                 lidar_data: Optional[Dict] = None,
                                 frame_info: Optional[Dict] = None) -> np.ndarray:
        """
        计算单帧图像中所有目标的风险真值

        Args:
            detections: 目标检测列表，每个元素包含：
                - 'bbox': [x1, y1, x2, y2]
                - 'class_id': 类别ID
                - 'confidence': 置信度
                - 'occlusion': 遮挡程度 (0-1)
            lidar_data: LiDAR数据，包含：
                - 'points': 点云数据 [N, 4]
                - 'velocities': 速度信息（如果有）
                - 'positions': 3D位置（如果有）
            frame_info: 帧信息，包含：
                - 'frame_id': 帧ID
                - 'sequence_id': 序列ID
                - 'timestamp': 时间戳

        Returns:
            risk_scores: [M] 每个目标的风险得分 (0-1)
        """
        if not detections:
            return np.array([])

        n_targets = len(detections)

        # 1. 计算每个目标的个体风险
        individual_risks = np.zeros(n_targets)

        for i, det in enumerate(detections):
            # 基础风险
            base_risk = self._calculate_base_risk(det)

            # 运动风险（需要LiDAR或估计）
            motion_risk = self._calculate_motion_risk(det, lidar_data, i)

            # 环境风险
            env_risk = self._calculate_environment_risk(det, frame_info)

            # 组合风险
            individual_risks[i] = base_risk * motion_risk * env_risk

        # 2. 计算目标间的交互风险
        if n_targets > 1:
            interaction_risks = self._calculate_interaction_risks(detections, lidar_data)
            # 将交互风险整合到个体风险中
            for i in range(n_targets):
                if interaction_risks[i] > individual_risks[i]:
                    individual_risks[i] = interaction_risks[i]

        # 3. 归一化到[0, 1]范围
        if individual_risks.max() > 0:
            individual_risks = individual_risks / individual_risks.max()

        return individual_risks

    def _calculate_base_risk(self, detection: Dict) -> float:
        """计算基础风险"""
        class_id = detection.get('class_id', 0)
        occlusion = detection.get('occlusion', 0.0)
        confidence = detection.get('confidence', 0.5)

        # 1. 目标类型权重
        type_weight = self.config['class_weights'].get(class_id, 0.5)

        # 2. 可见度系数（遮挡程度）
        visibility_coef = 1.0 - occlusion  # 遮挡越严重，可见度越低，风险越高

        # 3. 检测置信度影响
        # 低置信度可能表示目标模糊或部分可见，风险较高
        confidence_risk = 1.2 - confidence  # 置信度越低，风险越高（最大1.2，最小0.2）
        confidence_risk = np.clip(confidence_risk, 0.2, 1.0)

        # 4. 目标尺寸影响（小目标风险更高）
        bbox = detection.get('bbox', [0, 0, 1, 1])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # 假设图像尺寸为1280x720
        normalized_area = area / (1280 * 720)
        size_risk = 1.0 / (1.0 + np.exp(10 * (normalized_area - 0.05)))  # 小目标风险高

        # 组合基础风险
        base_risk = type_weight * visibility_coef * confidence_risk * size_risk

        return np.clip(base_risk, 0.0, 1.0)

    def _calculate_motion_risk(self,
                               detection: Dict,
                               lidar_data: Optional[Dict],
                               target_idx: int) -> float:
        """计算运动风险"""
        # 如果没有LiDAR数据，使用估计值
        if lidar_data is None or 'velocities' not in lidar_data:
            return self._estimate_motion_risk(detection)

        # 如果有LiDAR速度信息
        velocities = lidar_data.get('velocities', [])
        if target_idx < len(velocities):
            speed = velocities[target_idx]  # 假设速度单位是km/h
        else:
            speed = 0.0

        # 速度风险（非线性）
        speed_ratio = min(speed / self.config['max_speed'], 1.0)
        speed_risk = np.exp(2 * speed_ratio) / np.exp(2)  # 指数增长

        # 加速度风险（如果有）
        accelerations = lidar_data.get('accelerations', [])
        if target_idx < len(accelerations):
            accel = abs(accelerations[target_idx])
            accel_risk = min(accel / 5.0, 1.0)  # 假设5m/s²为最大加速度
        else:
            accel_risk = 0.0

        # 运动方向风险（如果有）
        # 可根据运动方向与道路方向的夹角计算

        motion_risk = 0.7 * speed_risk + 0.3 * accel_risk

        return np.clip(motion_risk, 0.1, 1.0)

    def _estimate_motion_risk(self, detection: Dict) -> float:
        """在没有LiDAR时估计运动风险"""
        class_id = detection.get('class_id', 0)

        # 根据类别估计典型速度
        typical_speeds = {
            0: 40.0,  # 车辆：40 km/h
            1: 15.0,  # 自行车：15 km/h
            2: 5.0  # 行人：5 km/h
        }

        speed = typical_speeds.get(class_id, 20.0)
        speed_ratio = min(speed / self.config['max_speed'], 1.0)
        speed_risk = np.exp(2 * speed_ratio) / np.exp(2)

        return np.clip(speed_risk, 0.2, 1.0)

    def _calculate_environment_risk(self,
                                    detection: Dict,
                                    frame_info: Optional[Dict]) -> float:
        """计算环境风险"""
        # 如果没有帧信息，返回中等风险
        if frame_info is None:
            return 0.5

        # 可以根据以下因素计算：
        # 1. 时间（夜晚风险更高）
        # 2. 天气（雨雪天气风险更高）
        # 3. 交通密度（高密度风险更高）
        # 4. 道路类型（交叉口风险更高）

        # 这里简化处理，返回固定值
        return 0.5

    def _calculate_interaction_risks(self,
                                     detections: List[Dict],
                                     lidar_data: Optional[Dict]) -> np.ndarray:
        """计算目标间的交互风险"""
        n_targets = len(detections)
        interaction_risks = np.zeros(n_targets)

        if n_targets < 2:
            return interaction_risks

        # 获取所有目标的边界框和类别
        bboxes = [det['bbox'] for det in detections]
        class_ids = [det['class_id'] for det in detections]

        # 计算每对目标之间的风险
        for i in range(n_targets):
            max_pair_risk = 0.0

            for j in range(n_targets):
                if i == j:
                    continue

                # 计算两个目标之间的风险
                pair_risk = self._calculate_pair_risk(
                    detections[i], detections[j],
                    bboxes[i], bboxes[j],
                    class_ids[i], class_ids[j],
                    lidar_data
                )

                if pair_risk > max_pair_risk:
                    max_pair_risk = pair_risk

            interaction_risks[i] = max_pair_risk

        return interaction_risks

    def _calculate_pair_risk(self,
                             det1: Dict,
                             det2: Dict,
                             bbox1: List,
                             bbox2: List,
                             class1: int,
                             class2: int,
                             lidar_data: Optional[Dict]) -> float:
        """计算两个目标之间的风险"""

        # 1. 计算图像空间距离
        # 边界框中心点
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2

        # 欧氏距离（图像空间）
        img_distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        normalized_distance = img_distance / np.sqrt(1280 ** 2 + 720 ** 2)

        # 距离风险（距离越近风险越高）
        distance_risk = 1.0 / (1.0 + np.exp(10 * (normalized_distance - 0.1)))

        # 2. 冲突对类型权重
        conflict_type = self._get_conflict_type(class1, class2)
        type_weight = self.config['conflict_pair_weights'].get(conflict_type, 0.5)

        # 3. 相对运动风险（如果有LiDAR）
        relative_motion_risk = 0.0
        if lidar_data is not None and 'velocities' in lidar_data:
            velocities = lidar_data['velocities']
            # 这里简化处理，实际需要更复杂的相对运动计算
            if len(velocities) > max(class1, class2):
                v1 = velocities[class1]
                v2 = velocities[class2]
                relative_speed = abs(v1 - v2)
                relative_motion_risk = min(relative_speed / 30.0, 1.0)

        # 4. 轨迹冲突风险（预测是否会碰撞）
        # 这里简化，可以根据目标类别和位置估计
        trajectory_risk = 0.0
        if normalized_distance < 0.2:  # 距离较近
            if class1 != class2:  # 不同类别更可能冲突
                trajectory_risk = 0.7
            else:
                trajectory_risk = 0.3

        # 组合风险
        pair_risk = type_weight * (0.5 * distance_risk +
                                   0.3 * relative_motion_risk +
                                   0.2 * trajectory_risk)

        return np.clip(pair_risk, 0.0, 1.0)

    def _get_conflict_type(self, class1: int, class2: int) -> str:
        """获取冲突对类型"""
        class_names = {
            0: 'vehicle',
            1: 'bicycle',
            2: 'person'
        }

        name1 = class_names.get(class1, 'vehicle')
        name2 = class_names.get(class2, 'vehicle')

        # 排序以确保一致性
        if (name1, name2) in [('person', 'vehicle'), ('vehicle', 'person')]:
            return 'person_vehicle'
        elif (name1, name2) in [('bicycle', 'vehicle'), ('vehicle', 'bicycle')]:
            return 'bicycle_vehicle'
        elif (name1, name2) in [('person', 'bicycle'), ('bicycle', 'person')]:
            return 'person_bicycle'
        elif name1 == 'vehicle' and name2 == 'vehicle':
            return 'vehicle_vehicle'
        else:
            return 'same_class'


def generate_risk_labels_for_dataset(dataset_path: str,
                                     output_dir: str,
                                     split: str = 'train',
                                     num_samples: int = -1):
    """
    为整个数据集生成风险标签

    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录
        split: 数据划分
        num_samples: 生成样本数量（-1表示全部）
    """
    import os
    import json
    from tqdm import tqdm
    from data.datasets.r_livit_dataset import RLiViTDataset

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    dataset = RLiViTDataset(
        dataset_path=dataset_path,
        split=split,
        modalities=['rgb', 'thermal'],
        use_lidar=False  # 暂时不使用LiDAR
    )

    # 初始化风险计算器
    risk_calculator = RiskGroundTruthCalculator()

    # 确定要处理的样本数量
    total_samples = len(dataset)
    if num_samples > 0:
        total_samples = min(total_samples, num_samples)

    print(f"📊 Generating risk labels for {total_samples} samples...")

    all_risk_labels = {}

    for idx in tqdm(range(total_samples), desc="Generating risk labels"):
        try:
            # 加载数据
            data = dataset[idx]

            # 构建检测列表
            detections = []
            bboxes = data['bboxes']
            labels = data['labels']
            occlusion = data['occlusion']

            for i in range(len(bboxes)):
                det = {
                    'bbox': bboxes[i].tolist(),
                    'class_id': int(labels[i]),
                    'confidence': 0.8,  # 使用默认置信度
                    'occlusion': float(occlusion[i]) if i < len(occlusion) else 0.0
                }
                detections.append(det)

            # 计算风险真值
            risk_scores = risk_calculator.calculate_risk_for_frame(
                detections=detections,
                lidar_data=None,  # 暂时不用LiDAR
                frame_info={
                    'frame_id': data['frame_id'],
                    'sequence_id': data['sequence_id']
                }
            )

            # 保存结果
            sample_id = data['sample_id']
            all_risk_labels[sample_id] = {
                'risk_scores': risk_scores.tolist(),
                'num_targets': len(detections),
                'frame_id': data['frame_id'],
                'sequence_id': data['sequence_id']
            }

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")

    # 保存到文件
    output_file = os.path.join(output_dir, f'risk_labels_{split}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_risk_labels, f, indent=2, ensure_ascii=False)

    print(f"✅ Risk labels saved to {output_file}")
    print(f"   Total samples with risk labels: {len(all_risk_labels)}")

    # 统计信息
    total_risks = []
    for sample_id, labels in all_risk_labels.items():
        total_risks.extend(labels['risk_scores'])

    if total_risks:
        print(f"📈 Risk statistics:")
        print(f"   Mean: {np.mean(total_risks):.4f}")
        print(f"   Std: {np.std(total_risks):.4f}")
        print(f"   Min: {np.min(total_risks):.4f}")
        print(f"   Max: {np.max(total_risks):.4f}")
        print(f"   Distribution: <0.2: {(np.array(total_risks) < 0.2).sum()}, "
              f"0.2-0.5: {((np.array(total_risks) >= 0.2) & (np.array(total_risks) < 0.5)).sum()}, "
              f"0.5-0.8: {((np.array(total_risks) >= 0.5) & (np.array(total_risks) < 0.8)).sum()}, "
              f">=0.8: {(np.array(total_risks) >= 0.8).sum()}")


def test_risk_calculator():
    """测试风险计算器"""
    print("🧪 Testing Risk Ground Truth Calculator...")

    # 创建测试数据
    test_detections = [
        {
            'bbox': [100, 100, 200, 200],  # 车辆
            'class_id': 0,
            'confidence': 0.9,
            'occlusion': 0.1
        },
        {
            'bbox': [300, 300, 350, 400],  # 行人（部分遮挡）
            'class_id': 2,
            'confidence': 0.6,
            'occlusion': 0.6
        },
        {
            'bbox': [500, 200, 600, 300],  # 自行车
            'class_id': 1,
            'confidence': 0.8,
            'occlusion': 0.3
        }
    ]

    # 初始化计算器
    calculator = RiskGroundTruthCalculator()

    # 计算风险
    risk_scores = calculator.calculate_risk_for_frame(
        detections=test_detections,
        lidar_data=None,
        frame_info={'frame_id': '001', 'sequence_id': '000'}
    )

    print(f"Test detections: {len(test_detections)} targets")
    print(f"Risk scores: {risk_scores}")

    # 解释结果
    class_names = {0: 'Vehicle', 1: 'Bicycle', 2: 'Person'}
    for i, (det, risk) in enumerate(zip(test_detections, risk_scores)):
        print(f"  Target {i} ({class_names[det['class_id']]}): "
              f"risk={risk:.4f}, occlusion={det['occlusion']:.2f}")


if __name__ == "__main__":
    # 测试风险计算器
    test_risk_calculator()

    # 如果需要生成整个数据集的风险标签，取消下面的注释
    # generate_risk_labels_for_dataset(
    #     dataset_path="D:/BaiduNetdiskDownload/R-LiViT/R-LiViT",
    #     output_dir="./risk_labels",
    #     split='train',
    #     num_samples=100  # 先测试100个样本
    # )