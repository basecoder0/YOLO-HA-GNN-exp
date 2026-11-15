# verify_gnn_dataset_1280.py
import os
import torch
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEFAULT_DIR = "./stage2_gnn_dataset_1280"

class DatasetVerifier:
    def __init__(self, data_dir=DEFAULT_DIR):
        self.data_dir = Path(data_dir)
        self.train_list = self.data_dir / "stage2_train_list.json"
        self.id2name = {0: 'Bg', 1: 'Car', 2: 'Person', 3: 'Bike', 4: 'Motor', 5: 'Bus', 6: 'Truck', 7: 'Tram', 8: 'Escooter'}

    def compute_iou(self, box1, boxes2):
        b1_x1, b1_y1 = box1[0]-box1[2]/2, box1[1]-box1[3]/2
        b1_x2, b1_y2 = box1[0]+box1[2]/2, box1[1]+box1[3]/2
        b2_x1, b2_y1 = boxes2[:,0]-boxes2[:,2]/2, boxes2[:,1]-boxes2[:,3]/2
        b2_x2, b2_y2 = boxes2[:,0]+boxes2[:,2]/2, boxes2[:,1]+boxes2[:,3]/2
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = (inter_x2-inter_x1).clamp(min=0) * (inter_y2-inter_y1).clamp(min=0)
        b1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        b2_area = (b2_x2-b2_x1)*(b2_y2-b2_y1)
        return inter_area / (b1_area + b2_area - inter_area + 1e-6)

    def run(self):
        print(f"🔍 verify dataset: {self.data_dir}")
        if not self.train_list.exists():
            print("❌ index file missing")
            return
        with open(self.train_list) as f: samples = json.load(f)

        print("\n📈 category distribution (check for new categories)...")
        label_counts = {}
        for info in random.sample(samples, min(1000, len(samples))):
            try:
                d = torch.load(info['path'], map_location='cpu', weights_only=False)
                for l in d['labels'].tolist(): label_counts[l] = label_counts.get(l, 0) + 1
            except: pass
        for k in sorted(label_counts.keys()):
            print(f"   ID {k} ({self.id2name.get(k)}): {label_counts[k]}")

        print("\n📡 clustering analysis...")
        total_pos, total_neig = 0, 0
        for info in tqdm(random.sample(samples, min(200, len(samples)))):
            try:
                d = torch.load(info['path'], map_location='cpu', weights_only=False)
                pos = d['boxes'][d['labels'] > 0]
                neg = d['boxes'][d['labels'] == 0]
                if len(pos) == 0: continue
                for i in range(len(pos)):
                    if len(neg) == 0: break
                    ious = self.compute_iou(pos[i], neg)
                    total_neig += (ious > 0.1).sum().item()
                    total_pos += 1
            except: pass

        avg = total_neig / max(1, total_pos)
        print(f"\n📊 average false positive count: {avg:.2f}")
        if avg < 0.5: print("⚠️ warning: environment too clean")
        else: print("✅ environment moderate")

if __name__ == "__main__":
    DatasetVerifier().run()