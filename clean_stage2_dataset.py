# clean_stage2_dataset_1280.py
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# 🔥 pointing to 1280 dataset
DEFAULT_DIR = "./stage2_gnn_dataset_1280"

def check_logical_health(data, sample_id):
    boxes = data['boxes']
    if boxes.numel() == 0: return True
    centers = boxes[:, :2]
    dists = torch.cdist(centers, centers)
    dists.fill_diagonal_(1.0)
    if (dists < 1e-5).any():
        print(f"❌ {sample_id}: foundcompletely overlapping nodes")
        return False
    w, h = boxes[:, 2], boxes[:, 3]
    if (ar := w / (h + 1e-6)).max() > 20 or ar.min() < 1 / 20:
        print(f"❌ {sample_id}: extreme aspect ratio")
        return False
    if data['features'].abs().mean() > 5.0:
        print(f"❌ {sample_id}: feature value anomaly")
        return False
    return True

def clean_dataset(data_dir=DEFAULT_DIR):
    data_path = Path(data_dir)
    train_path = data_path / "stage2_train_list.json"
    test_path = data_path / "stage2_test_list.json"
    trash_dir = data_path / "corrupted_files"
    trash_dir.mkdir(exist_ok=True)

    if not train_path.exists():
        print(f"❌ index file not found: {train_path}")
        return

    with open(train_path, 'r') as f: train_list = json.load(f)
    with open(test_path, 'r') as f: test_list = json.load(f)

    all_samples = train_list + test_list
    valid_train, valid_test = [], []
    removed = 0

    print(f"🔍 start cleaning {data_dir} ({len(all_samples)} file)...")

    for info in tqdm(all_samples):
        pt_path = info['path']
        is_valid = True
        try:
            if not os.path.exists(pt_path): is_valid = False
            else:
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
                if not check_logical_health(data, info['id']): is_valid = False
                if torch.isnan(data['features']).any(): is_valid = False
        except: is_valid = False

        if is_valid:
            if info in train_list: valid_train.append(info)
            else: valid_test.append(info)
        else:
            if os.path.exists(pt_path):
                try: shutil.move(pt_path, trash_dir / Path(pt_path).name)
                except: pass
            removed += 1

    with open(train_path, 'w') as f: json.dump(valid_train, f, indent=2)
    with open(test_path, 'w') as f: json.dump(valid_test, f, indent=2)
    print(f"✅ cleaning completed! removed: {removed}")

if __name__ == "__main__":
    clean_dataset()