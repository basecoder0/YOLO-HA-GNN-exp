import torch
import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
from utils.device_utils import setup_device

# ================= configuration =================
OURS_S1 = r"D:\YOLO-GNN-Fusion\runs\stage1_rlivit_highres_8cls\best.pt"
BASE_MODEL = r"D:\YOLO-GNN-Fusion\weights\yolov8s.pt"
DATA_CFG = {'root_dir': "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT", 'train_txt': "train.txt", 'test_txt': "test.txt",
            'img_size': 1280}
OUTPUT_DIR = "attention_vis"


# =======================================

class AttentionVisualizer:
    def __init__(self):
        self.device = setup_device()
        self.model = YOLOWithFusion(model_path=BASE_MODEL, num_classes=8).to(self.device).eval()
        self.model.load_state_dict(torch.load(OURS_S1, map_location=self.device))
        self.dataset = RLiViTDataset(DATA_CFG, mode='test')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.att_map = None

    def hook_fn(self, module, input, output):
        # assumeoutputis Tensor，calculationmeanasheatimage
        if isinstance(output, torch.Tensor):
            self.att_map = output.mean(dim=1, keepdim=True)

    def run(self):
        # 🔥 autosearch Fusion layer
        hooked = False
        print("🔍 Scanning model for fusion layers...")
        for name, module in self.model.named_modules():
            # criticalwordmatch：findnameinwith fusion or attention oflayer
            if "fusion" in name.lower() and isinstance(module, torch.nn.Conv2d):
                print(f"✅ Hooking layer: {name}")
                module.register_forward_hook(self.hook_fn)
                hooked = True
                break  # only Hook firstafindtoof

        if not hooked:
            print("⚠️ notfindto Fusion layer，try Hook Backbone offinalonelayer...")
            # Fallback: Hook SPPF
            for name, module in self.model.named_modules():
                if "sppf" in name.lower():
                    module.register_forward_hook(self.hook_fn)
                    hooked = True
                    break

        indices = np.random.choice(len(self.dataset), 5, replace=False)

        for idx in indices:
            sample_info = self.dataset.samples[idx]  # Use samples list
            rgb_path = sample_info.get('rgb') or sample_info.get('image')
            thm_path = sample_info.get('thermal')

            if not rgb_path or not os.path.exists(rgb_path): continue

            # Inference
            sample = self.dataset[idx]  # Get Tensor data
            rgb_t = sample['rgb'].unsqueeze(0).to(self.device)
            thm_t = sample['thermal'].unsqueeze(0).to(self.device)
            self.model({'rgb': rgb_t, 'thermal': thm_t})

            if self.att_map is not None:
                att = self.att_map[0, 0].cpu().detach().numpy()
                # Normalize
                att = (att - att.min()) / (att.max() - att.min() + 1e-6)
                att = cv2.resize(att, (1280, 1280))
                heatmap = cv2.applyColorMap(np.uint8(255 * att), cv2.COLORMAP_JET)

                rgb_img = cv2.imread(rgb_path)
                rgb_img = cv2.resize(rgb_img, (1280, 1280))
                thm_img = cv2.imread(thm_path)
                thm_img = cv2.resize(thm_img, (1280, 1280))

                overlay = cv2.addWeighted(rgb_img, 0.6, heatmap, 0.4, 0)
                combined = np.hstack([rgb_img, thm_img, overlay])
                cv2.imwrite(f"{OUTPUT_DIR}/att_{idx}.jpg", combined)
                print(f"📸 Saved: {OUTPUT_DIR}/att_{idx}.jpg")


if __name__ == "__main__":
    AttentionVisualizer().run()