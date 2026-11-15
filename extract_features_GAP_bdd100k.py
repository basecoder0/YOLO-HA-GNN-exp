import os
import sys
import json
import torch
import cv2
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import roi_align

# auto locate project rootdirectory
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from models.full_pipeline import YOLOGNNFusion
from utils.device_utils import setup_device

# ================= 🔧 configuration area =================
BDD_ROOT = r"D:\BaiduNetdiskDownload\BDD100K"
IMG_DIR = os.path.join(BDD_ROOT, "b_car", "images", "val")  # readverifycollect
OUTPUT_DIR = Path("stage2_gnn_dataset_bdd100k_val")  # outputfeatureoffolder

# Stage 1 weight (forextractfeature)
CHECKPOINT_S1 = "runs/stage1_rlivit_highres_8cls/best.pt"
# if best notexist，try last
if not os.path.exists(CHECKPOINT_S1):
    CHECKPOINT_S1 = "runs/stage1_rlivit_highres_8cls/last.pt"


# ===============================================

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


class BDDFeatureExtractor:
    def __init__(self):
        self.device = setup_device()
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

        print("🔧 loading Stage 1 model (forfeatureextract)...")
        # initializationmodelframeworkconstruct
        try:
            self.model = YOLOGNNFusion(num_classes=8, confidence_threshold=0.01, nms_threshold=0.6,
                                       model_path='weights/yolov8s.pt').to(self.device)
        except TypeError:
            self.model = YOLOGNNFusion(num_classes=8, confidence_threshold=0.01, nms_threshold=0.6).to(self.device)

        # loadtraininggoodofweight
        if not os.path.exists(CHECKPOINT_S1):
            raise FileNotFoundError(f"❌ findnottoweightfile {CHECKPOINT_S1}")

        print(f"📦 loadingweight: {CHECKPOINT_S1}")
        checkpoint = torch.load(CHECKPOINT_S1, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()

        self.features_buffer = {}
        self._find_feature_layers()  # notevolumehookelementgetinspacelayerfeature

    def _find_feature_layers(self):
        print("\n🔍 --- motionstatelayerlevelsearch (P3, P4, P5) ---")
        temp_hooks = []
        neck = self.model.yolo_fusion.neck_layers

        def get_shape_hook(idx):
            def hook(model, input, output): self.features_buffer[idx] = output.shape

            return hook

        for i, layer in enumerate(neck):
            temp_hooks.append(layer.register_forward_hook(get_shape_hook(i)))

        # runoneminortemporarydatatosuredefinelayerindex
        dummy_rgb = torch.zeros(1, 3, 1280, 1280).to(self.device)
        dummy_thm = torch.zeros(1, 3, 1280, 1280).to(self.device)
        with torch.no_grad():
            self.model(dummy_rgb, dummy_thm)

        found_p3, found_p4, found_p5 = None, None, None
        for i, shape in self.features_buffer.items():
            h = shape[2]
            if h == 160:
                found_p3 = i
            elif h == 80:
                found_p4 = i
            elif h == 40:
                found_p5 = i

        for h in temp_hooks: h.remove()
        self.features_buffer = {}

        if found_p3 is None or found_p4 is None or found_p5 is None:
            raise RuntimeError("❌ unable toautolocate P3/P4/P5 featurelayer！")

        print(f"✅ lockdefinefeaturelayer: P3({found_p3}), P4({found_p4}), P5({found_p5})")

        def get_activation(name):
            def hook(model, input, output): self.features_buffer[name] = output

            return hook

        neck[found_p3].register_forward_hook(get_activation('p3'))
        neck[found_p4].register_forward_hook(get_activation('p4'))
        neck[found_p5].register_forward_hook(get_activation('p5'))

    def extract_roi_features_gap(self, feat_dict, boxes):
        batch_idx = torch.zeros((boxes.shape[0], 1), device=self.device)
        boxes_with_idx = torch.cat([batch_idx, boxes], dim=1)

        # RoI Align -> [N, C, 7, 7]
        p3 = roi_align(feat_dict['p3'], boxes_with_idx, (7, 7), spatial_scale=1 / 8)
        p4 = roi_align(feat_dict['p4'], boxes_with_idx, (7, 7), spatial_scale=1 / 16)
        p5 = roi_align(feat_dict['p5'], boxes_with_idx, (7, 7), spatial_scale=1 / 32)

        # GAP -> [N, C]
        p3_gap = p3.mean(dim=[2, 3])
        p4_gap = p4.mean(dim=[2, 3])
        p5_gap = p5.mean(dim=[2, 3])

        return torch.cat([p3_gap, p4_gap, p5_gap], dim=1)

    def process_bdd(self):
        target_size = 1280
        # getallimagepiece
        img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

        if len(img_files) == 0:
            print(f"❌ error: at/in {IMG_DIR} belownothavefindto .jpg imagepiece！")
            return

        print(f"\n🚀 startprocessing BDD100K validationcollect (total {len(img_files)} tensor)...")

        sample_list = []  # forgenerate json index

        for img_path in tqdm(img_files, desc="Extracting Features"):
            file_name = os.path.basename(img_path)
            stem = os.path.splitext(file_name)[0]  # e.g., b1c81faa-3df17267

            # checkwhetheralreadyexist，avoidweightcomplexrun
            save_path = self.output_dir / f"{stem}.pt"
            if save_path.exists():
                sample_list.append({'id': stem, 'path': str(save_path)})
                continue

            # 1. read RGB
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 2. pseudocreate Thermal (convertgraydegree)
            # becausefor/as Stage1 modelneeddual streaminput，but BDD onlyhave RGB
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_thm = cv2.merge([img_gray, img_gray, img_gray])  # 3channelgraydegreeimage

            # 3. Letterbox Resize
            img_rgb, _, _ = letterbox(img_rgb, (target_size, target_size))
            img_thm, _, _ = letterbox(img_thm, (target_size, target_size))

            # 4. convert Tensor
            t_rgb = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            t_thm = torch.from_numpy(img_thm).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

            try:
                with torch.no_grad():
                    self.features_buffer = {}
                    outputs = self.model(t_rgb, t_thm)
                    pred_boxes = outputs['detections']

                    # make sureextracttopast tense markerfeaturelayer
                    if not all(k in self.features_buffer for k in ['p3', 'p4', 'p5']): continue

                # ifnothavedetectiontoobjectbody，skip
                if pred_boxes is None or len(pred_boxes) == 0:
                    continue

                boxes_xyxy = pred_boxes[:, :4]
                scores = pred_boxes[:, 4:6]

                # 5. GAP featureextract
                node_features = self.extract_roi_features_gap(self.features_buffer, boxes_xyxy)

                # 6. normalizecoordinate (Norm XYWH)
                norm_boxes = torch.zeros_like(boxes_xyxy)
                norm_boxes[:, 0] = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2) / target_size  # cx
                norm_boxes[:, 1] = ((boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2) / target_size  # cy
                norm_boxes[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / target_size  # w
                norm_boxes[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / target_size  # h

                save_data = {
                    'sample_id': stem,
                    'boxes': norm_boxes.cpu(),
                    'features': node_features.cpu(),
                    'scores': scores.cpu(),
                    'img_size': target_size
                }
                torch.save(save_data, save_path)
                sample_list.append({'id': stem, 'path': str(save_path)})

            except Exception as e:
                # print(f"Error {stem}: {e}")
                pass

        # generateindexfile，foraftercontinuevisualizationscriptusing
        # hereof key mustmustis 'test'，becausefor/asyouofvisualizationscriptreadofis samples
        # but to not overwrite the original R-LiViT list，ourexistfor/as stage2_bdd_val_list.json
        with open(self.output_dir / "stage2_bdd_val_list.json", 'w') as f:
            json.dump(sample_list, f, indent=2)

        print(f"\n✅ featureextractcompleted！")
        print(f"   - savepath: {self.output_dir}")
        print(f"   - listfile: {self.output_dir / 'stage2_bdd_val_list.json'}")
        if 'node_features' in locals():
            print(f"   - featuredimension: {node_features.shape[1]}")


if __name__ == "__main__":
    BDDFeatureExtractor().process_bdd()