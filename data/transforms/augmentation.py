import torch
import numpy as np
import cv2
import random
from typing import Dict, Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MultiModalTransform:
    """
    multiplemodelstatedatatransformationbaseclass
    processingRGB、thermalimagingandLiDARdata
    """

    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size

    def __call__(self, data: Dict) -> Dict:
        """application/applytransformation"""
        raise NotImplementedError


class TrainTransform(MultiModalTransform):
    """
    trainingdatatransformation - fixdimension顺sequenceandnormalizeversion (alreadyremovednotcompatibleof CoarseDropout)
    """

    def __init__(self,
                 input_size: Tuple[int, int] = (640, 640),
                 use_augmentation: bool = True,
                 normalize_rgb: bool = True,
                 normalize_thermal: bool = True):

        # make sureinput_sizeiselementgroup
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        elif not isinstance(input_size, (tuple, list)):
            input_size = (640, 640)
        else:
            input_size = tuple(input_size)

        super().__init__(input_size)

        self.use_augmentation = use_augmentation
        self.normalize_rgb = normalize_rgb
        self.normalize_thermal = normalize_thermal

        # RGBimageenhancetransformation
        if use_augmentation:
            self.rgb_augmentations = A.Compose([
                # 1. fewwhattransformation
                A.RandomResizedCrop(height=input_size[1], width=input_size[0],
                                    scale=(0.5, 1.0), ratio=(0.8, 1.2), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=15, p=0.3),

                # 2. colortransformation (alreadyverifythrough)
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1, p=0.5),

                # 3. noise声andblurry
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),

                # ❌ removednotcompatibleof CoarseDropout

            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels', 'occlusion'],
                min_area=16,
                min_visibility=0.1
            ))
        else:
            self.rgb_augmentations = A.Compose([
                A.Resize(height=input_size[1], width=input_size[0]),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels', 'occlusion'],
                min_area=1,
                min_visibility=0.1
            ))

        # thermalimagingenhancetransformation
        self.thermal_augmentations = A.Compose([
            A.RandomResizedCrop(height=input_size[1], width=input_size[0],
                                scale=(0.5, 1.0), ratio=(0.8, 1.2), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ])

        self.to_tensor = ToTensorV2()

        # RGBnormalizeparameter
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

    def __call__(self, data: Dict) -> Dict:
        """application/applytrainingtransformation"""

        # 1. processingRGBimageandboundarybox
        if 'rgb' in data:
            rgb_image = data['rgb']
            bboxes = data.get('bboxes', np.zeros((0, 4)))
            labels = data.get('labels', np.zeros((0,)))
            occlusion = data.get('occlusion', np.zeros((0,)))

            # convertfor/asnumpy
            if torch.is_tensor(rgb_image):
                rgb_image = rgb_image.cpu().numpy()

            if not isinstance(rgb_image, np.ndarray):
                rgb_image = np.array(rgb_image)

            # [criticalkeep] checkandadjustmentdimension顺sequence: (C, H, W) -> (H, W, C)
            if rgb_image.ndim == 3 and rgb_image.shape[0] in [1, 3] and rgb_image.shape[0] < rgb_image.shape[1]:
                rgb_image = np.transpose(rgb_image, (1, 2, 0))

            # make sureimageisuint8classtype [0-255]
            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.astype(np.uint8)

            # make sureboundaryboxisnumpycountgroup
            if not isinstance(bboxes, np.ndarray):
                if torch.is_tensor(bboxes):
                    bboxes = bboxes.cpu().numpy()
                else:
                    bboxes = np.array(bboxes)

            if bboxes.ndim == 1 and len(bboxes) == 4:
                bboxes = bboxes.reshape(1, -1)

            # make sureboundaryboxdatahaveeffect
            valid_bboxes = []
            valid_labels = []
            valid_occlusion = []

            if len(bboxes) > 0:
                h, w = rgb_image.shape[:2]
                for i, box in enumerate(bboxes):
                    # checkcoordinatewhetherhaveeffect
                    if box[2] > box[0] and box[3] > box[1] and \
                            box[0] >= 0 and box[1] >= 0 and \
                            box[2] <= 1.0 and box[3] <= 1.0:  # assumeinputisnormalizeof0-1
                        # Albumentations pascal_voc formatneedabsolutecoordinate
                        abs_box = [
                            box[0] * w,
                            box[1] * h,
                            box[2] * w,
                            box[3] * h
                        ]
                        valid_bboxes.append(abs_box)
                        valid_labels.append(labels[i])
                        if len(occlusion) > i:
                            valid_occlusion.append(occlusion[i])
                        else:
                            valid_occlusion.append(0.0)

            # application/applyenhance
            if len(valid_bboxes) > 0:
                try:
                    augmented = self.rgb_augmentations(
                        image=rgb_image,
                        bboxes=valid_bboxes,
                        labels=valid_labels,
                        occlusion=valid_occlusion
                    )
                    rgb_image = augmented['image']

                    # convertfallbackrelativecoordinate [0-1]
                    aug_h, aug_w = rgb_image.shape[:2]
                    new_bboxes = []
                    if augmented['bboxes']:
                        for box in augmented['bboxes']:
                            new_bboxes.append([
                                box[0] / aug_w,
                                box[1] / aug_h,
                                box[2] / aug_w,
                                box[3] / aug_h
                            ])

                    bboxes = np.array(new_bboxes) if new_bboxes else np.zeros((0, 4))
                    labels = np.array(augmented['labels']) if augmented['labels'] else np.zeros((0,))
                    occlusion = np.array(augmented['occlusion']) if augmented['occlusion'] else np.zeros((0,))
                except Exception as e:
                    print(f"Augmentation failed: {e}")
                    # fallbackexit：onlyadjustmentlargesmall
                    rgb_image = cv2.resize(rgb_image, (self.input_size[0], self.input_size[1]))
                    bboxes = np.zeros((0, 4))
                    labels = np.zeros((0,))
            else:
                rgb_image = cv2.resize(rgb_image, (self.input_size[0], self.input_size[1]))
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))

            # convertfor/astensoramount (ToTensorV2 willautowill HWC convertfor/as CHW)
            rgb_tensor = self.to_tensor(image=rgb_image)['image']

            # normalize
            if self.normalize_rgb:
                rgb_tensor = rgb_tensor.float() / 255.0
                for c in range(3):
                    rgb_tensor[c] = (rgb_tensor[c] - self.rgb_mean[c]) / self.rgb_std[c]

            data['rgb'] = rgb_tensor
            data['bboxes'] = bboxes
            data['labels'] = labels
            data['occlusion'] = occlusion

        # 2. processingthermalimagingimage
        if 'thermal' in data:
            thermal_image = data['thermal']

            # convertfor/asnumpy
            if torch.is_tensor(thermal_image):
                thermal_image = thermal_image.cpu().numpy()

            # processingdimension (C, H, W) -> (H, W) or (H, W, C)
            if thermal_image.ndim == 3:
                if thermal_image.shape[0] == 1:  # (1, H, W)
                    thermal_image = thermal_image[0]
                elif thermal_image.shape[0] == 3:  # (3, H, W)
                    thermal_image = np.transpose(thermal_image, (1, 2, 0))
                    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2GRAY)

            # make sureisuint8
            if thermal_image.dtype != np.uint8:
                if thermal_image.max() <= 1.0:
                    thermal_image = (thermal_image * 255).astype(np.uint8)
                else:
                    thermal_image = thermal_image.astype(np.uint8)

            # enhance
            try:
                augmented = self.thermal_augmentations(image=thermal_image)
                thermal_image = augmented['image']
            except:
                thermal_image = cv2.resize(thermal_image, (self.input_size[0], self.input_size[1]))

            # make sureconvertfor/as3channel (suitableshouldResNetinput)
            if thermal_image.ndim == 2:
                thermal_image = np.stack([thermal_image] * 3, axis=-1)
            elif thermal_image.ndim == 3 and thermal_image.shape[2] == 1:
                thermal_image = np.concatenate([thermal_image] * 3, axis=-1)

            # convertfor/astensoramount [C, H, W]
            thermal_tensor = torch.from_numpy(thermal_image.transpose(2, 0, 1)).float()

            if self.normalize_thermal:
                thermal_tensor = thermal_tensor / 255.0

            data['thermal'] = thermal_tensor

        # 3. LiDAR (keepnotchange)
        if 'lidar' in data and torch.is_tensor(data['lidar']):
            lidar_tensor = data['lidar']
            if self.use_augmentation:
                if random.random() < 0.5:
                    lidar_tensor = torch.flip(lidar_tensor, dims=[-1])
                if random.random() < 0.3:
                    k = random.randint(1, 3)
                    lidar_tensor = torch.rot90(lidar_tensor, k, dims=[-2, -1])
            data['lidar'] = lidar_tensor

        return data


class ValTransform(MultiModalTransform):
    """validationdatatransformation"""

    def __init__(self,
                 input_size: Tuple[int, int] = (640, 640),
                 normalize_rgb: bool = True,
                 normalize_thermal: bool = True):

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        elif not isinstance(input_size, (tuple, list)):
            input_size = (640, 640)
        else:
            input_size = tuple(input_size)

        super().__init__(input_size)

        self.normalize_rgb = normalize_rgb
        self.normalize_thermal = normalize_thermal

        self.rgb_transform = A.Compose([
            A.Resize(height=input_size[1], width=input_size[0]),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels', 'occlusion'],
            min_area=1,
            min_visibility=0.1
        ))

        self.thermal_transform = A.Compose([
            A.Resize(height=input_size[1], width=input_size[0]),
        ])

        self.to_tensor = ToTensorV2()

        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

    def __call__(self, data: Dict) -> Dict:
        # 1. RGB
        if 'rgb' in data:
            rgb_image = data['rgb']
            bboxes = data.get('bboxes', np.zeros((0, 4)))
            labels = data.get('labels', np.zeros((0,)))
            occlusion = data.get('occlusion', np.zeros((0,)))

            if torch.is_tensor(rgb_image):
                rgb_image = rgb_image.cpu().numpy()

            if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
                rgb_image = np.transpose(rgb_image, (1, 2, 0))

            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)

            valid_bboxes = []
            h, w = rgb_image.shape[:2]
            if len(bboxes) > 0:
                for box in bboxes:
                    valid_bboxes.append([
                        box[0] * w, box[1] * h, box[2] * w, box[3] * h
                    ])

            transformed = self.rgb_transform(
                image=rgb_image,
                bboxes=valid_bboxes if len(valid_bboxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
                occlusion=occlusion.tolist() if len(occlusion) > 0 else []
            )

            rgb_image = transformed['image']

            if transformed['bboxes']:
                aug_h, aug_w = rgb_image.shape[:2]
                new_bboxes = []
                for box in transformed['bboxes']:
                    new_bboxes.append([box[0] / aug_w, box[1] / aug_h, box[2] / aug_w, box[3] / aug_h])
                bboxes = np.array(new_bboxes)
                labels = np.array(transformed['labels'])
                occlusion = np.array(transformed['occlusion'])

            rgb_tensor = self.to_tensor(image=rgb_image)['image']

            if self.normalize_rgb:
                rgb_tensor = rgb_tensor.float() / 255.0
                for c in range(3):
                    rgb_tensor[c] = (rgb_tensor[c] - self.rgb_mean[c]) / self.rgb_std[c]

            data['rgb'] = rgb_tensor
            data['bboxes'] = bboxes
            data['labels'] = labels
            data['occlusion'] = occlusion

        # 2. Thermal
        if 'thermal' in data:
            thermal_image = data['thermal']
            if torch.is_tensor(thermal_image):
                thermal_image = thermal_image.cpu().numpy()

            if thermal_image.ndim == 3 and thermal_image.shape[0] in [1, 3]:
                if thermal_image.shape[0] == 3:
                    thermal_image = np.transpose(thermal_image, (1, 2, 0))
                    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2GRAY)
                else:
                    thermal_image = thermal_image[0]

            if thermal_image.dtype != np.uint8:
                thermal_image = (thermal_image * 255).astype(
                    np.uint8) if thermal_image.max() <= 1.0 else thermal_image.astype(np.uint8)

            transformed = self.thermal_transform(image=thermal_image)
            thermal_image = transformed['image']

            if thermal_image.ndim == 2:
                thermal_image = np.stack([thermal_image] * 3, axis=-1)

            thermal_tensor = torch.from_numpy(thermal_image.transpose(2, 0, 1)).float()
            if self.normalize_thermal:
                thermal_tensor = thermal_tensor / 255.0

            data['thermal'] = thermal_tensor

        return data


class TestTransform(ValTransform):
    pass


class DualResNetTransform:
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size
        self.train_transform = TrainTransform(input_size)
        self.val_transform = ValTransform(input_size)
        self.test_transform = TestTransform(input_size)

    def train(self, data: Dict) -> Dict: return self.train_transform(data)

    def val(self, data: Dict) -> Dict: return self.val_transform(data)

    def test(self, data: Dict) -> Dict: return self.test_transform(data)


def collate_multimodal_batch(batch: List[Dict]) -> Dict:
    from data.dataloaders.collate_fn import collate_fn
    return collate_fn(batch)