import torch
import os
import sys
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

# addprojectrootdirectorytopath
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# importcustommodule
from models.yolo.yolo_with_fusion_ablation import YOLOWithFusionAblation
from data.datasets.r_livit_dataset import RLiViTDataset
from utils.device_utils import setup_device
from utils.metrics.evaluation_metrics import ap_per_class, box_iou

# ================= 🔧 debugconfiguration =================
# onlyrun Baseline performdiagnosebreak
EXPERIMENT = {
    "name": "Baseline (YOLOv8)",
    "path": r"D:\YOLO-GNN-Fusion\runs\ablation_rgb\best.pt",
    "mode": "rgb",
    "base_model": "weights/yolov8s.pt"
}

DATA_CFG = {
    'root_dir': "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT",
    'train_txt': "train.txt",
    'test_txt': "test.txt",
    'img_size': 1280,
    'batch_size': 2,  # smallbatchminormethodconvenientdebug
    'num_workers': 0
}


# ================= 🛠️ insideplaceutility functions =================
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def my_collate_fn(batch):
    rgb_list = []
    thermal_list = []
    labels_list = []
    for i, item in enumerate(batch):
        rgb_list.append(item['rgb'])
        thermal_list.append(item['thermal'])
        lbl = item['labels']
        if lbl.numel() > 0:
            lbl = lbl.clone()
            lbl[:, 0] = i
            labels_list.append(lbl)
    rgb_batch = torch.stack(rgb_list, 0)
    thermal_batch = torch.stack(thermal_list, 0)
    if len(labels_list) > 0:
        labels_batch = torch.cat(labels_list, 0)
    else:
        labels_batch = torch.zeros((0, 6))
    return {'rgb': rgb_batch, 'thermal': thermal_batch, 'labels': labels_batch}


def inspect_weights(model):
    """checkdetectionheadofweightstatus"""
    print("\n🔍 --- weight健康degreecheck ---")
    # check Head ofscore/divideclassbranch
    cv3_weight = model.head.cv3[0][0].conv.weight
    print(f"   Head.cv3 (Cls) Weight Mean: {cv3_weight.mean().item():.6f} | Std: {cv3_weight.std().item():.6f}")

    # check Head of DFL branch (negative责boxdecoding)
    if hasattr(model.head, 'dfl'):
        dfl_weight = model.head.dfl.conv.weight
        print(f"   Head.dfl (Box) Weight Mean: {dfl_weight.mean().item():.6f} | Std: {dfl_weight.std().item():.6f}")
    else:
        print("   ⚠️ Head nothave DFL module？thiswillcauseunable todecoding！")


def smart_load_weights_debug(model, ckpt_path, device):
    print(f"\n📥 positiveat/inloadingweight: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model'].state_dict() if hasattr(ckpt.get('model'), 'state_dict') else ckpt.get('model', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # statisticsmatchsituationsituation
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # trystandard YOLO 迁移
        if any(k.startswith('model.0') for k in ckpt_keys) and not any(k.startswith('rgb_backbone') for k in ckpt_keys):
            print("   ⚠️ detectiontostandard YOLO format，executekeynamemapping...")
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    parts = k.split('.')
                    layer_idx = int(parts[1])
                    rest = '.'.join(parts[2:])
                    if 0 <= layer_idx <= 9:
                        new_k = f'rgb_backbone.{layer_idx}.{rest}'
                    elif 10 <= layer_idx <= 21:
                        new_k = f'neck_layers.{layer_idx - 10}.{rest}'
                    elif layer_idx == 22:
                        new_k = f'head.{rest}'
                    else:
                        new_k = k
                    new_sd[new_k] = v
                else:
                    new_sd[k] = v
            state_dict = new_sd
            ckpt_keys = set(state_dict.keys())

        # check Head weightwhethermatch
        head_keys_model = {k for k in model_keys if 'head' in k}
        head_keys_ckpt = {k for k in ckpt_keys if 'head' in k}
        missing_head = head_keys_model - head_keys_ckpt

        print(f"   📊 keynamestatistics: modeltotalkeycount={len(model_keys)}, weightfilekeycount={len(ckpt_keys)}")
        print(f"   🎯 Headlayermatch: modelHeadkey={len(head_keys_model)}, weightHeadkey={len(head_keys_ckpt)}")

        if len(missing_head) > 0:
            print(f"   ❌ warning！detectionheadhave {len(missing_head)} aparameternotloading！thiswillcausepredictionfullerror！")
            print(f"   examplelacklosekey: {list(missing_head)[:3]}")
        else:
            print("   ✅ detectionheadparametercompletelymatch！")

        model.load_state_dict(state_dict, strict=False)
        return True
    except Exception as e:
        print(f"   ❌ loadingerror: {e}")
        return False


def debug_inference():
    device = setup_device()
    print(f"🚀 startdiagnosebreakmode (Device: {device})")

    # 1. modelinitialization
    config = EXPERIMENT
    model = YOLOWithFusionAblation(
        model_path=config['base_model'],
        num_classes=8,
        fusion_mode=config['mode']
    )
    model.to(device)

    # 2. weightload & check
    if not smart_load_weights_debug(model, config['path'], device):
        return

    inspect_weights(model)
    model.eval()

    # 3. dataload
    dataset = RLiViTDataset(DATA_CFG, mode='test')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=my_collate_fn)

    print("\n🧐 --- inferenceoutputdiagnosebreak (onlyrun 1 a Batch) ---")
    batch = next(iter(dataloader))
    rgb = batch['rgb'].to(device)
    thermal = batch['thermal'].to(device)

    with torch.no_grad():
        inputs = {'rgb': rgb, 'thermal': thermal}
        preds = model(inputs)

        # processing Tuple
        if isinstance(preds, tuple):
            print("   ℹ️ modelreturnpast tense marker Tuple (matchpredict/pre期)")
            preds = preds[0]  # [bs, 8400, 4+nc]
        else:
            print(f"   ℹ️ modelreturnpast tense marker {type(preds)}")

        # checkvalue
        print(f"   📦 output Tensor Shape: {preds.shape}")

        # checkconfidence
        # preds: [bs, 4+nc, anchors] or [bs, anchors, 4+nc]
        # v8 Detect Head output is usually [bs, 4+nc, 8400]
        if preds.shape[1] > preds.shape[2]:  # [bs, 8400, 84] -> [bs, 84, 8400]
            preds = preds.transpose(1, 2)
            print("   ℹ️by/withthisconvertplace Tensor shape")

        # thiswhen preds shouldfor/as [bs, 4+nc, anchors] -> 8+4=12
        # orone ultralytics ofoutputis [bs, 4+nc, 8400]
        # let us print to see

        # Ultralytics of detect head outputusuallyis [bs, 4+nc, 8400] (nc=8, total=12)
        conf_scores = preds[:, 4:, :].amax(dim=1)  # [bs, 8400]
        max_conf = conf_scores.max().item()
        mean_conf = conf_scores.mean().item()

        print(f"   🔥 maximumconfidence (Max Conf): {max_conf:.6f}")
        print(f"   📉 averageconfidence (Mean Conf): {mean_conf:.6f}")

        if max_conf < 0.01:
            print("   ❌ strictweightwarning：maximumconfidencetoolow！modelfew乎nothavedetectiontoanytarget。")
            print("      mayoriginalbecause：1. weightloadingfailed（虽thennoterror） 2. inputdatanormalizeerror 3. trainingfailed")
        else:
            print("   ✅ confidencepositiveoften。")

        # checkcoordinatedecoding
        # getconfidencemosthighofoneabox
        b_idx, a_idx = torch.where(conf_scores == max_conf)
        b_idx, a_idx = b_idx[0], a_idx[0]

        best_box = preds[b_idx, :4, a_idx]  # x, y, w, h
        print(f"   📍 bestboxcoordinate (Raw): {best_box.tolist()}")

        if best_box.max() < 2.0:
            print("   ❌ strictweightwarning：boxcoordinateseems to be normalized (0~1) orone Grid offsetamount！")
            print("      thisnotice味着 Detect headnothaveautodecoding。mAP calculationwillfailed（because we expect pixelscoordinate）。")
        else:
            print("   ✅ boxcoordinatelooks like pixel-level (>1)。")


if __name__ == "__main__":
    debug_inference()