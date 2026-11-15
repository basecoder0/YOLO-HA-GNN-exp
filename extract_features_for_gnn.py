# extract_features_1280_s.py
"""
Stage 2 featureextractscript (1280 highclear + YOLOv8s adaptversion)
"""
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision.ops import roi_align
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.full_pipeline import YOLOGNNFusion
from utils.device_utils import setup_device


# 1280 Letterbox (andtrainingkeeponeconsistent)
def letterbox(img, new_shape=(1280, 1280), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


# Projector (autoadaptchannelcount)
class FeatureProjector(nn.Module):
    def __init__(self, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.fc1 = None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def init_fc1(self, total_dim):
        device = next(self.parameters()).device
        print(f"🔌 [Projector] autodetectioninputfeaturedimension: {total_dim} (adapt yolov8s) -> mappingto {self.hidden_dim}")
        self.fc1 = nn.Linear(total_dim, self.hidden_dim).to(device)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, rois_p3, rois_p4, rois_p5):
        # expandflatandcombineconnect
        cat_feat = torch.cat([rois_p3.flatten(1), rois_p4.flatten(1), rois_p5.flatten(1)], dim=1)
        if self.fc1 is None: self.init_fc1(cat_feat.shape[1])
        return self.fc2(self.relu(self.fc1(cat_feat)))


class Stage1FeatureExtractor:
    def __init__(self):
        self.device = setup_device()
        # pathconfiguration
        self.base_path = r"D:\BaiduNetdiskDownload\R-LiViT\R-LiViT"
        self.rgb_base = os.path.join(self.base_path, "R-LiViT_RGB-T", "rgb")
        self.thermal_base = os.path.join(self.base_path, "R-LiViT_RGB-T", "thermal")
        self.annotations_base = os.path.join(self.base_path, "R-LiViT_RGB-T", "annotations")

        # 🔥 outputdirectory: 1280 specialuse
        self.output_dir = Path("./stage2_gnn_dataset_1280")
        self.output_dir.mkdir(exist_ok=True)

        print("🔧 loading Stage 1 model (1280version + YOLOv8s)...")
        # trypassinput model_path，if YOLOGNNFusion notbranchholdparameterwillerror，
        # butregardingforyouchangepast tense marker yolo_with_fusion ofdefaultvalue，even without passing parameters it defaults toload s。
        # heretocompatibleproperty，firsttrypassparameter，iferrorpleasedelete model_path parameter。
        try:
            self.model = YOLOGNNFusion(num_classes=8, confidence_threshold=0.01, nms_threshold=0.6,
                                       model_path='weights/yolov8s.pt').to(self.device)
        except TypeError:
            print("⚠️ YOLOGNNFusion notconnectbear model_path，willusingdefaultinitialization (pleasemake sure yolo_with_fusion.py defaultvaluefor/as s)")
            self.model = YOLOGNNFusion(num_classes=8, confidence_threshold=0.01, nms_threshold=0.6).to(self.device)

        # loadtraininggoodof best.pt
        ckpt_path = "runs/stage1_rlivit_highres_8cls/best.pt"
        if not os.path.exists(ckpt_path): ckpt_path = "runs/stage1_rlivit_highres_8cls/last.pt"

        print(f"📂 readweight: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ findnottoweightfile {ckpt_path}，please firstcompleted Stage 1 training！")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # filter mismatched keys (keepriskbeginsee)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()

        self.features_buffer = {}
        self._register_hooks()

        # Projector
        self.projector = FeatureProjector().to(self.device)
        self.projector.eval()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output): self.features_buffer[name] = output

            return hook

        # notevolumehookelementto Neck outputlayer
        self.model.yolo_fusion.neck_layers[5].register_forward_hook(get_activation('p3'))
        self.model.yolo_fusion.neck_layers[8].register_forward_hook(get_activation('p4'))
        self.model.yolo_fusion.neck_layers[11].register_forward_hook(get_activation('p5'))

    def process_dataset(self):
        split_dir = os.path.join(self.base_path, "R-LiViT_RGB-T")
        splits = {'train': 'train.txt', 'test': 'test.txt'}
        timestamps = ['02', '06', '10', '14', '18', '22', '26', '30', '34', '38']
        sample_lists = {'train': [], 'test': []}

        print("\n🚀 startfeatureextract (1280 Resolution)...")
        for split_name, split_file in splits.items():
            if not os.path.exists(os.path.join(split_dir, split_file)): continue
            with open(os.path.join(split_dir, split_file), 'r') as f:
                sequenceuenceuences = [x.strip() for x in f.readlines() if x.strip()]

            for sequenceuence in tqdm(sequenceuenceuences, desc=f"Processing {split_name}"):
                for ts in timestamps:
                    sample_id = f"{sequenceuence}_{ts}"
                    rgb_path = os.path.join(self.rgb_base, sequenceuence, f"{ts}.png")
                    thm_path = os.path.join(self.thermal_base, sequenceuence, f"{ts}.png")
                    ann_path = os.path.join(self.annotations_base, sequenceuence, f"{ts}.xml")
                    if not os.path.exists(rgb_path): continue

                    # 🔥 1280 resolution
                    target_size = 1280

                    img_rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                    img_thm = cv2.imread(thm_path, cv2.IMREAD_GRAYSCALE)
                    img_thm = cv2.merge([img_thm, img_thm, img_thm])

                    img_rgb, ratio, (dw, dh) = letterbox(img_rgb, (target_size, target_size))
                    img_thm, _, _ = letterbox(img_thm, (target_size, target_size))

                    t_rgb = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                    t_thm = torch.from_numpy(img_thm).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

                    try:
                        with torch.no_grad():
                            self.features_buffer = {}
                            outputs = self.model(t_rgb, t_thm)
                            pred_boxes = outputs['detections']

                            feats = {k: self.features_buffer.get(k) for k in ['p3', 'p4', 'p5']}
                            if any(v is None for v in feats.values()): continue

                        if pred_boxes is None or len(pred_boxes) == 0: continue

                        boxes_xyxy = pred_boxes[:, :4]
                        scores = pred_boxes[:, 4:6]

                        # extractfeature (thiswhen Projector willautoadapt yolov8s ofchannelcount)
                        node_features = self.extract_roi_features(feats, boxes_xyxy)

                        # normalize Box (divide/removeby/with 1280)
                        norm_boxes = boxes_xyxy.clone()
                        norm_boxes[:, 0] = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2) / target_size
                        norm_boxes[:, 1] = ((boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2) / target_size
                        norm_boxes[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / target_size
                        norm_boxes[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / target_size

                        gt_boxes, gt_cls = self._load_xml_gt(ann_path, ratio, dw, dh)
                        labels = self._match_gt(boxes_xyxy, gt_boxes, gt_cls)

                        save_data = {
                            'sample_id': sample_id,
                            'boxes': norm_boxes.cpu(),
                            'features': node_features.cpu(),
                            'scores': scores.cpu(),
                            'labels': labels.cpu(),
                            'img_size': target_size
                        }

                        save_path = self.output_dir / f"{sample_id}.pt"
                        torch.save(save_data, save_path)
                        sample_lists[split_name].append(
                            {'id': sample_id, 'path': str(save_path), 'num_nodes': len(labels)})

                    except Exception as e:
                        pass

        for k, v in sample_lists.items():
            with open(self.output_dir / f"stage2_{k}_list.json", 'w') as f: json.dump(v, f, indent=2)
        print(f"\n✅ 1280featureextractcompleted！directory: {self.output_dir}")

    def extract_roi_features(self, feat_dict, boxes):
        batch_idx = torch.zeros((boxes.shape[0], 1), device=self.device)
        boxes_with_idx = torch.cat([batch_idx, boxes], dim=1)
        p3 = roi_align(feat_dict['p3'], boxes_with_idx, (7, 7), spatial_scale=1 / 8)
        p4 = roi_align(feat_dict['p4'], boxes_with_idx, (7, 7), spatial_scale=1 / 16)
        p5 = roi_align(feat_dict['p5'], boxes_with_idx, (7, 7), spatial_scale=1 / 32)
        return self.projector(p3, p4, p5)

    def _load_xml_gt(self, xml, ratio, dw, dh):
        if not os.path.exists(xml): return torch.empty(0, 4).to(self.device), torch.empty(0).to(self.device)
        root = ET.parse(xml).getroot()
        boxes, clss = [], []
        # 8 classmapping (needand Dataset oneconsistent)
        cls_map = {'car': 1, 'person': 2, 'bicycle': 3, 'motorcycle': 4, 'bus': 5, 'truck': 6, 'tramway': 7,
                   'escooter': 8, 'e-scooter': 8, 'van': 1, 'suv': 1, 'trailer': 6}
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name not in cls_map: continue
            b = obj.find('bndbox')
            xmin = float(b.find('xmin').text) * ratio[0] + dw
            xmax = float(b.find('xmax').text) * ratio[0] + dw
            ymin = float(b.find('ymin').text) * ratio[0] + dh
            ymax = float(b.find('ymax').text) * ratio[0] + dh
            boxes.append([xmin, ymin, xmax, ymax])
            clss.append(cls_map[name])
        if not boxes: return torch.empty(0, 4).to(self.device), torch.empty(0).to(self.device)
        return torch.tensor(boxes).to(self.device), torch.tensor(clss).to(self.device)

    def _match_gt(self, preds, gts, gt_cls):
        if len(gts) == 0: return torch.zeros(len(preds), dtype=torch.long).to(self.device)
        lt = torch.max(preds[:, None, :2], gts[:, :2])
        rb = torch.min(preds[:, None, 2:], gts[:, 2:])
        area_i = (rb - lt).clamp(min=0).prod(2)
        area_p = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        area_g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        iou = area_i / (area_p[:, None] + area_g - area_i + 1e-6)
        vals, idxs = iou.max(1)
        labels = torch.zeros(len(preds), dtype=torch.long).to(self.device)
        labels[vals > 0.5] = gt_cls[idxs[vals > 0.5]]
        return labels


if __name__ == "__main__":
    Stage1FeatureExtractor().process_dataset()