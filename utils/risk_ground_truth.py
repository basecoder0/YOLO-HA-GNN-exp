import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import math


class RiskGroundTruthCalculator:
    """
    risktruevaluecalculationdevice
    rootaccordingsetcount文档inofriskevaluation公stylecalculationrisktruevalue
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = {
                # targetclasstypeweight
                'class_weights': {
                    0: 0.7,  # car/vehicle (defaultvalue，optionalrootaccordingneedadjustment)
                    1: 0.8,  # bicycle (nonmachinemotionvehicle)
                    2: 1.0  # person (lineperson，mosthighrisk)
                },

                # distanceriskparameter
                'distance_threshold': 50.0,  # 米，maximumhaveeffectdistance
                'min_distance': 2.0,  # 米，mostsmallsafety/fulldistance

                # velocityriskparameter
                'max_speed': 60.0,  # km/h，maximumvelocity
                'danger_speed_threshold': 30.0,  # km/h，危riskvelocitythreshold

                # coverblockriskparameter
                'severe_occlusion_threshold': 0.5,  # strictweightcoverblockthreshold

                # timeparameter（framerate）
                'fps': 2.0,  # framerate（R-LiViTinapproximately2-5fps）

                # conflict pairweight
                'conflict_pair_weights': {
                    'person_vehicle': 1.0,  # person-vehicleconflictbreak
                    'bicycle_vehicle': 0.8,  # nonmachinemotionvehicle-vehicleconflictbreak
                    'person_bicycle': 0.6,  # person-nonmachinemotionvehicleconflictbreak
                    'vehicle_vehicle': 0.3,  # vehicle-vehicleconflictbreak
                    'same_class': 0.1  # samecategoryconflictbreak
                }
            }
        self.config = config

        print("✅ Risk ground truth calculator initialized")

    def calculate_risk_for_frame(self,
                                 detections: List[Dict],
                                 lidar_data: Optional[Dict] = None,
                                 frame_info: Optional[Dict] = None) -> np.ndarray:
        """
        calculationsingleframeimageinalltargetofrisktruevalue

        Args:
            detections: object detectionlist，eachaelement素contains：
                - 'bbox': [x1, y1, x2, y2]
                - 'class_id': categoryID
                - 'confidence': confidence
                - 'occlusion': coverblockprogramdegree (0-1)
            lidar_data: LiDARdata，contains：
                - 'points': point clouddata [N, 4]
                - 'velocities': velocityinformation（ifhave）
                - 'positions': 3Ddigitplace（ifhave）
            frame_info: frameinformation，contains：
                - 'frame_id': frameID
                - 'sequenceuenceuence_id': sequenceuenceuentialID
                - 'timestamp': timestamp

        Returns:
            risk_scores: [M] eachatargetofriskgetscore/divide (0-1)
        """
        if not detections:
            return np.array([])

        n_targets = len(detections)

        # 1. calculationeachatargetofabodyrisk
        individual_risks = np.zeros(n_targets)

        for i, det in enumerate(detections):
            # basicrisk
            base_risk = self._calculate_base_risk(det)

            # motionrisk（needLiDARorestimatecount）
            motion_risk = self._calculate_motion_risk(det, lidar_data, i)

            # environmentrisk
            env_risk = self._calculate_environment_risk(det, frame_info)

            # groupcombinerisk
            individual_risks[i] = base_risk * motion_risk * env_risk

        # 2. calculationtargetspaceofinteractmutualrisk
        if n_targets > 1:
            interaction_risks = self._calculate_interaction_risks(detections, lidar_data)
            # willinteractmutualriskintegercombinetoabodyriskin
            for i in range(n_targets):
                if interaction_risks[i] > individual_risks[i]:
                    individual_risks[i] = interaction_risks[i]

        # 3. normalizeto[0, 1]range
        if individual_risks.max() > 0:
            individual_risks = individual_risks / individual_risks.max()

        return individual_risks

    def _calculate_base_risk(self, detection: Dict) -> float:
        """calculationbasicrisk"""
        class_id = detection.get('class_id', 0)
        occlusion = detection.get('occlusion', 0.0)
        confidence = detection.get('confidence', 0.5)

        # 1. targetclasstypeweight
        type_weight = self.config['class_weights'].get(class_id, 0.5)

        # 2. visibledegreecoefficient（coverblockprogramdegree）
        visibility_coef = 1.0 - occlusion  # coverblockmore/greaterstrictweight，visibledegreemore/greaterlow，riskmore/greaterhigh

        # 3. detectionconfidenceshadowresponse
        # lowconfidencemay indicate blurry or partially visible target，riskcomparehigh
        confidence_risk = 1.2 - confidence  # confidencemore/greaterlow，riskmore/greaterhigh（maximum1.2，mostsmall0.2）
        confidence_risk = np.clip(confidence_risk, 0.2, 1.0)

        # 4. targetsizeshadowresponse（smalltargetriskmorehigh）
        bbox = detection.get('bbox', [0, 0, 1, 1])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # assumeimagesizefor/as1280x720
        normalized_area = area / (1280 * 720)
        size_risk = 1.0 / (1.0 + np.exp(10 * (normalized_area - 0.05)))  # smalltargetriskhigh

        # groupcombinebasicrisk
        base_risk = type_weight * visibility_coef * confidence_risk * size_risk

        return np.clip(base_risk, 0.0, 1.0)

    def _calculate_motion_risk(self,
                               detection: Dict,
                               lidar_data: Optional[Dict],
                               target_idx: int) -> float:
        """calculationmotionrisk"""
        # ifnothaveLiDARdata，usingestimatecountvalue
        if lidar_data is None or 'velocities' not in lidar_data:
            return self._estimate_motion_risk(detection)

        # ifhaveLiDARvelocityinformation
        velocities = lidar_data.get('velocities', [])
        if target_idx < len(velocities):
            speed = velocities[target_idx]  # assumevelocitysingledigitiskm/h
        else:
            speed = 0.0

        # velocityrisk（nonlineproperty）
        speed_ratio = min(speed / self.config['max_speed'], 1.0)
        speed_risk = np.exp(2 * speed_ratio) / np.exp(2)  # pointcountincrease长

        # addvelocityrisk（ifhave）
        accelerations = lidar_data.get('accelerations', [])
        if target_idx < len(accelerations):
            accel = abs(accelerations[target_idx])
            accel_risk = min(accel / 5.0, 1.0)  # assume5m/s²for/asmaximumaddvelocity
        else:
            accel_risk = 0.0

        # motiondirectionrisk（ifhave）
        # optionalrootaccording to the angle between motion direction and road directioncalculation

        motion_risk = 0.7 * speed_risk + 0.3 * accel_risk

        return np.clip(motion_risk, 0.1, 1.0)

    def _estimate_motion_risk(self, detection: Dict) -> float:
        """at/innothaveLiDARwhenestimatecountmotionrisk"""
        class_id = detection.get('class_id', 0)

        # rootaccordingcategoryestimatecount典typevelocity
        typical_speeds = {
            0: 40.0,  # vehicle辆：40 km/h
            1: 15.0,  # selflinevehicle：15 km/h
            2: 5.0  # lineperson：5 km/h
        }

        speed = typical_speeds.get(class_id, 20.0)
        speed_ratio = min(speed / self.config['max_speed'], 1.0)
        speed_risk = np.exp(2 * speed_ratio) / np.exp(2)

        return np.clip(speed_risk, 0.2, 1.0)

    def _calculate_environment_risk(self,
                                    detection: Dict,
                                    frame_info: Optional[Dict]) -> float:
        """calculationenvironmentrisk"""
        # ifnothaveframeinformation，returninwaitrisk
        if frame_info is None:
            return 0.5

        # optionalby/withrootaccordingby/withbelowbecause素calculation：
        # 1. time（夜晚riskmorehigh）
        # 2. weather（rainsnowweatherriskmorehigh）
        # 3. interactpassdensedegree（highdensedegreeriskmorehigh）
        # 4. roadclasstype（interact叉mouthriskmorehigh）

        # heresimplifyprocessing，returnfixedvalue
        return 0.5

    def _calculate_interaction_risks(self,
                                     detections: List[Dict],
                                     lidar_data: Optional[Dict]) -> np.ndarray:
        """calculationtargetspaceofinteractmutualrisk"""
        n_targets = len(detections)
        interaction_risks = np.zeros(n_targets)

        if n_targets < 2:
            return interaction_risks

        # getboundaries of all targetsboxandcategory
        bboxes = [det['bbox'] for det in detections]
        class_ids = [det['class_id'] for det in detections]

        # calculationeachpairtargetofspaceofrisk
        for i in range(n_targets):
            max_pair_risk = 0.0

            for j in range(n_targets):
                if i == j:
                    continue

                # calculation两atargetofspaceofrisk
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
        """calculation两atargetofspaceofrisk"""

        # 1. calculationimageemptyspacedistance
        # boundaryboxincenterpoint
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2

        # 欧氏distance（imageemptyspace）
        img_distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        normalized_distance = img_distance / np.sqrt(1280 ** 2 + 720 ** 2)

        # distancerisk（distancemore/greatercloseriskmore/greaterhigh）
        distance_risk = 1.0 / (1.0 + np.exp(10 * (normalized_distance - 0.1)))

        # 2. conflict pairclasstypeweight
        conflict_type = self._get_conflict_type(class1, class2)
        type_weight = self.config['conflict_pair_weights'].get(conflict_type, 0.5)

        # 3. relativemotionrisk（ifhaveLiDAR）
        relative_motion_risk = 0.0
        if lidar_data is not None and 'velocities' in lidar_data:
            velocities = lidar_data['velocities']
            # heresimplifyprocessing，actualneedmore complex relative motioncalculation
            if len(velocities) > max(class1, class2):
                v1 = velocities[class1]
                v2 = velocities[class2]
                relative_speed = abs(v1 - v2)
                relative_motion_risk = min(relative_speed / 30.0, 1.0)

        # 4. 轨迹conflictbreakrisk（predictionwhetherwill碰撞）
        # heresimplify，optionalby/withrootaccordingtargetcategoryanddigitplaceestimatecount
        trajectory_risk = 0.0
        if normalized_distance < 0.2:  # distancecompareclose
            if class1 != class2:  # notsamecategorymoremayconflictbreak
                trajectory_risk = 0.7
            else:
                trajectory_risk = 0.3

        # groupcombinerisk
        pair_risk = type_weight * (0.5 * distance_risk +
                                   0.3 * relative_motion_risk +
                                   0.2 * trajectory_risk)

        return np.clip(pair_risk, 0.0, 1.0)

    def _get_conflict_type(self, class1: int, class2: int) -> str:
        """getconflict pairclasstype"""
        class_names = {
            0: 'vehicle',
            1: 'bicycle',
            2: 'person'
        }

        name1 = class_names.get(class1, 'vehicle')
        name2 = class_names.get(class2, 'vehicle')

        # arrangesequenceby/withmake sureoneconsistentproperty
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
    for/asintegeradatasetgeneraterisklabel

    Args:
        dataset_path: datasetpath
        output_dir: outputdirectory
        split: datasplit/divide
        num_samples: generatesamplecountamount（-1indicatefullpart）
    """
    import os
    import json
    from tqdm import tqdm
    from data.datasets.r_livit_dataset import RLiViTDataset

    # createoutputdirectory
    os.makedirs(output_dir, exist_ok=True)

    # loaddataset
    dataset = RLiViTDataset(
        dataset_path=dataset_path,
        split=split,
        modalities=['rgb', 'thermal'],
        use_lidar=False  # temporarilywhennotusingLiDAR
    )

    # initializationriskcalculationdevice
    risk_calculator = RiskGroundTruthCalculator()

    # suredefineneedprocessingofsamplecountamount
    total_samples = len(dataset)
    if num_samples > 0:
        total_samples = min(total_samples, num_samples)

    print(f"📊 Generating risk labels for {total_samples} samples...")

    all_risk_labels = {}

    for idx in tqdm(range(total_samples), desc="Generating risk labels"):
        try:
            # loaddata
            data = dataset[idx]

            # builddetectionlist
            detections = []
            bboxes = data['bboxes']
            labels = data['labels']
            occlusion = data['occlusion']

            for i in range(len(bboxes)):
                det = {
                    'bbox': bboxes[i].tolist(),
                    'class_id': int(labels[i]),
                    'confidence': 0.8,  # usingdefaultconfidence
                    'occlusion': float(occlusion[i]) if i < len(occlusion) else 0.0
                }
                detections.append(det)

            # calculationrisktruevalue
            risk_scores = risk_calculator.calculate_risk_for_frame(
                detections=detections,
                lidar_data=None,  # temporarilywhennotuseLiDAR
                frame_info={
                    'frame_id': data['frame_id'],
                    'sequenceuenceuence_id': data['sequenceuenceuence_id']
                }
            )

            # saveresult
            sample_id = data['sample_id']
            all_risk_labels[sample_id] = {
                'risk_scores': risk_scores.tolist(),
                'num_targets': len(detections),
                'frame_id': data['frame_id'],
                'sequenceuenceuence_id': data['sequenceuenceuence_id']
            }

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")

    # savetofile
    output_file = os.path.join(output_dir, f'risk_labels_{split}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_risk_labels, f, indent=2, ensure_ascii=False)

    print(f"✅ Risk labels saved to {output_file}")
    print(f"   Total samples with risk labels: {len(all_risk_labels)}")

    # statisticsinformation
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
    """testrisk calculationdevice"""
    print("🧪 Testing Risk Ground Truth Calculator...")

    # createtestdata
    test_detections = [
        {
            'bbox': [100, 100, 200, 200],  # vehicle辆
            'class_id': 0,
            'confidence': 0.9,
            'occlusion': 0.1
        },
        {
            'bbox': [300, 300, 350, 400],  # lineperson（partialcoverblock）
            'class_id': 2,
            'confidence': 0.6,
            'occlusion': 0.6
        },
        {
            'bbox': [500, 200, 600, 300],  # selflinevehicle
            'class_id': 1,
            'confidence': 0.8,
            'occlusion': 0.3
        }
    ]

    # initializationcalculationdevice
    calculator = RiskGroundTruthCalculator()

    # calculationrisk
    risk_scores = calculator.calculate_risk_for_frame(
        detections=test_detections,
        lidar_data=None,
        frame_info={'frame_id': '001', 'sequenceuenceuence_id': '000'}
    )

    print(f"Test detections: {len(test_detections)} targets")
    print(f"Risk scores: {risk_scores}")

    # resolveexplainresult
    class_names = {0: 'Vehicle', 1: 'Bicycle', 2: 'Person'}
    for i, (det, risk) in enumerate(zip(test_detections, risk_scores)):
        print(f"  Target {i} ({class_names[det['class_id']]}): "
              f"risk={risk:.4f}, occlusion={det['occlusion']:.2f}")


if __name__ == "__main__":
    # testriskcalculationdevice
    test_risk_calculator()

    # ifneedgenerateintegeradatasetofrisklabel，geteliminatebelowfaceofnoteexplain
    # generate_risk_labels_for_dataset(
    #     dataset_path="D:/BaiduNetdiskDownload/R-LiViT/R-LiViT",
    #     output_dir="./risk_labels",
    #     split='train',
    #     num_samples=100  # firsttest100asample
    # )