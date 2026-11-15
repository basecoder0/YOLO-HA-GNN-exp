import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class RLiViTDataset(Dataset):
    def __init__(self, data_cfg, mode='train'):
        self.root = data_cfg['root_dir']
        self.img_size = data_cfg['img_size']
        self.mode = mode

        # category mapping
        self.class_map = {
            'car': 0,
            'person': 1,
            'bicycle': 2,
        }

        txt_file = data_cfg['train_txt'] if mode == 'train' else data_cfg['test_txt']
        self.base_subfolder = 'R-LiViT_RGB-T'
        list_path = os.path.join(self.root, self.base_subfolder, txt_file)

        print(f"[{mode}] positiveat/intryloadingsplit/dividefile: {list_path}")

        if not os.path.exists(list_path):
            list_path_fallback = os.path.join(self.root, txt_file)
            if os.path.exists(list_path_fallback):
                list_path = list_path_fallback
            else:
                raise FileNotFoundError(f"❌ findnottodatasetsplit/dividefile: {list_path}")

        with open(list_path, 'r') as f:
            self.sequenceuenceuences = [x.strip() for x in f.readlines()]

        self.samples = self._load_samples()
        print(f"[{mode}] R-LiViT datasetloadingsuccess: found {len(self.samples)} asample")

    def _load_samples(self):
        samples = []
        base_path = os.path.join(self.root, self.base_subfolder)

        for sequenceuence in self.sequenceuenceuences:
            rgb_dir = os.path.join(base_path, 'rgb', sequenceuence)
            thermal_dir = os.path.join(base_path, 'thermal', sequenceuence)
            anno_dir = os.path.join(base_path, 'annotations', sequenceuence)

            if not os.path.exists(rgb_dir):
                continue

            for fname in os.listdir(rgb_dir):
                if not (fname.endswith('.png') or fname.endswith('.jpg')): continue
                xml_name = os.path.splitext(fname)[0] + '.xml'
                sample = {
                    'rgb': os.path.join(rgb_dir, fname),
                    'thermal': os.path.join(thermal_dir, fname),
                    'xml': os.path.join(anno_dir, xml_name)
                }
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """
        core fix：Letterbox Resize (keepaspect ratio + fillfullgray边)
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # calculationshrinkscalecompareexample (r = new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # calculationfillfull (padding)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w, h
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = dw / 2, dh / 2  # divide padding into 2 sides (居in)

        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add Border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, r, (dw, dh)

    def __getitem__(self, index):
        s = self.samples[index]

        # 1. read
        rgb = cv2.imread(s['rgb'])
        thermal = cv2.imread(s['thermal'], cv2.IMREAD_GRAYSCALE)

        if rgb is None or thermal is None:
            return self.__getitem__((index + 1) % len(self))

        # BGR -> RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 2. Letterbox Resize (generationreplaceoriginaltoof暴ability Resize)
        # note：RGB and Thermal mustmustapplication/applycompletelysameoftransformation
        rgb_resized, ratio, (dw, dh) = self.letterbox(rgb, (self.img_size, self.img_size))

        # Thermal processing: keepsinglechannel resize，thenafter stack generate 3 channel
        # needmanualcall letterbox logicoronecomplexuseparameter，herefor simplicity call directly
        thermal_resized, _, _ = self.letterbox(thermal, (self.img_size, self.img_size))
        thermal_resized = cv2.merge([thermal_resized, thermal_resized, thermal_resized])

        # 3. convert Tensor
        rgb_t = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        thermal_t = torch.from_numpy(thermal_resized).permute(2, 0, 1).float() / 255.0

        # 4. labeltransformation
        labels = []
        if os.path.exists(s['xml']):
            try:
                tree = ET.parse(s['xml'])
                root = tree.getroot()

                for obj in root.findall('object'):
                    cls_name = obj.find('name').text
                    if cls_name not in self.class_map: continue
                    cls_id = self.class_map[cls_name]

                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    # === coordinatetransformation (Apply Letterbox Transform) ===
                    # firstshrinkscale
                    xmin *= ratio
                    xmax *= ratio
                    ymin *= ratio
                    ymax *= ratio

                    # thenaddoffset (Padding)
                    xmin += dw
                    xmax += dw
                    ymin += dh
                    ymax += dh

                    # === convert YOLO (cx, cy, w, h) ===
                    w_box = (xmax - xmin)
                    h_box = (ymax - ymin)
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2

                    # normalize (divide/removeby/with 640)
                    if w_box > 0 and h_box > 0:
                        labels.append([
                            0,
                            cls_id,
                            cx / self.img_size,
                            cy / self.img_size,
                            w_box / self.img_size,
                            h_box / self.img_size
                        ])
            except Exception as e:
                print(f"Error parsing XML {s['xml']}: {e}")

        labels_t = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 6))

        return {
            'rgb': rgb_t,
            'thermal': thermal_t,
            'labels': labels_t,
            'batch_idx': 0
        }