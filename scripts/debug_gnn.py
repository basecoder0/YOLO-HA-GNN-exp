import torch
import torch.nn.functional as F
import os
import sys
import json
import numpy as np

# addprojectrootdirectory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from utils.device_utils import setup_device

# ================= configuration =================
# pointing toyouof Stage 2 datasetandmodel
DATASET_JSON = os.path.join(PROJECT_ROOT, "stage2_gnn_dataset_1280", "stage2_test_list.json")
# youof GNN weightpath
CKPT_PATH = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_1280\epoch_100.pth"


# =======================================

def diagnose():
    device = setup_device()
    print(f"🔬 start GNN depthdiagnosebreak (Device: {device})")
    print(f"   📂 projectrootdirectory: {PROJECT_ROOT}")

    # 1. checkdatasetindexfile
    if not os.path.exists(DATASET_JSON):
        print(f"❌ findnottodatasetindex: {DATASET_JSON}")
        return

    with open(DATASET_JSON, 'r') as f:
        samples = json.load(f)
    print(f"✅ datasetindexloadingsuccess，total {len(samples)} asample")

    # 2. searchfindsample (containspathfixlogic)
    target_sample = None
    first_path_raw = samples[0]['path'] if len(samples) > 0 else "None"
    print(f"   ℹ️ JSON inrecordoffirst条path (original): {first_path_raw}")

    for s in samples:
        pt_path = s['path']

        # 🔥🔥🔥 core fix: autocombineconnectabsolutepath 🔥🔥🔥
        if not os.path.isabs(pt_path):
            pt_path = os.path.join(PROJECT_ROOT, pt_path)

        if os.path.exists(pt_path):
            target_sample = s
            target_sample['path'] = pt_path  # updatefor/aspositivesureofabsolutepath
            break

    if target_sample is None:
        print("❌ strictweighterror：unable tofindtoanyexistof .pt file！")
        print(f"   trysearchfindofpathexample: {os.path.join(PROJECT_ROOT, samples[0]['path'])}")
        print("   pleasecheck: stage2_gnn_dataset_1280 fileclampwhetherat/inprojectrootdirectorybelow？")
        return

    print(f"\n🧐 positiveat/inresolve剖sample: {target_sample['id']}")
    print(f"   📂 actualloadingpath: {target_sample['path']}")

    # 3. load .pt data
    data = torch.load(target_sample['path'], map_location=device, weights_only=False)
    print("\n📦 [.pt datainsidecontainercheck]")
    print(f"   Keys: {list(data.keys())}")

    # checkfeature
    raw_feat = data.get('roi_features', data.get('features'))
    if raw_feat is None:
        print("   ❌ featurelacklose！findnotto 'roi_features' or 'features'")
        return
    print(f"   Feature Shape: {raw_feat.shape}")
    print(f"   Feature Stats: Min={raw_feat.min():.4f}, Max={raw_feat.max():.4f}, Mean={raw_feat.mean():.4f}")
    if torch.isnan(raw_feat).any():
        print("   ❌ warning：featureincontains NaN！")

    # check盒element
    boxes = data['boxes']
    print(f"   Boxes Shape: {boxes.shape} (predict/pre期isnormalizecoordinate)")

    # checkscore
    scores = data['scores']
    print(f"   Scores Shape: {scores.shape}")

    # checklabel (thisiscritical！)
    labels = data.get('labels', data.get('gt_labels'))
    if labels is None:
        print("   ⚠️ warning：.pt fileinnothave 'labels' or 'gt_labels'！")
    else:
        print(f"   Labels Found! Shape: {labels.shape}")
        if len(labels) > 0:
            print(f"   Labels Sample (Raw): {labels[0].tolist()}")
            # checkwhetheris 1-based
            unique_cls = labels[:, 0].unique() if labels.dim() > 1 else labels.unique()
            print(f"   Unique Classes: {unique_cls.tolist()}")
            if unique_cls.min() >= 1:
                print("   ℹ️ foundlabelis 1-based (1-8)，evaluationscriptalreadycontainsautoreduce 1 offix。")
            else:
                print("   ℹ️ labellooksis 0-based (0-7)。")

    # 4. loadmodel
    print("\n🤖 [model loadingandinference]")
    model = HierarchicalGNN(node_feature_dim=256, hidden_dim=256).to(device)
    head_ex = torch.nn.Linear(128, 1).to(device)
    head_cl = torch.nn.Linear(128, 8).to(device)

    if not os.path.exists(CKPT_PATH):
        print(f"❌ findnottoweightfile: {CKPT_PATH}")
        return

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # compatiblepropertyload
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
        head_ex.load_state_dict(ckpt['head_exist'])
        head_cl.load_state_dict(ckpt['head_cls'])
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval();
    head_ex.eval();
    head_cl.eval()
    print("   ✅ modelweightloadingsuccess")

    # checkweightstatistics
    w_ex = head_ex.weight.data
    print(f"   Head_Exist Weight Stats: Mean={w_ex.mean():.6f}, Std={w_ex.std():.6f}")

    # 5. forward propagation
    # simulateevaluationwhenofpredict/preprocessing
    feat = F.normalize(torch.nan_to_num(raw_feat.float(), 0.0).clamp(-100, 100), p=2, dim=1)

    with torch.no_grad():
        out, keep, _ = model([feat], [boxes.float()], [scores[:, 0]])

        # printinspacefeature
        print(f"   GNN Output Shape: {out.shape}")

        # prediction Exist
        logits_ex = head_ex(out)
        prob_ex = torch.sigmoid(logits_ex)

        # prediction Class
        logits_cl = head_cl(out)
        prob_cl = torch.softmax(logits_cl, dim=1)
        pred_cls = torch.argmax(prob_cl, dim=1)

    print("\n🔍 [predictionresultanalysis]")
    print(f"   Exist Logits (Raw): Min={logits_ex.min():.4f}, Max={logits_ex.max():.4f}, Mean={logits_ex.mean():.4f}")
    print(f"   Exist Prob (Sigmoid): Min={prob_ex.min():.4f}, Max={prob_ex.max():.4f}, Mean={prob_ex.mean():.4f}")

    # statisticshow many samples exceed 0.4 ofthreshold
    pass_count = (prob_ex > 0.4).sum().item()
    total_count = prob_ex.numel()
    print(f"   🔥 thresholdtest (Threshold=0.4): {pass_count}/{total_count} anodethrough")

    if pass_count == 0:
        print("   ❌ 潜at/inproblem：allnodeofexistpropertyscorealllowfor 0.4。")
        print("      recommend：at/in evaluate_final.py inwill GNN thresholdfrom 0.4 reducelowto 0.1 or 0.01 testtest。")
    else:
        print("   ✅ existpropertyscoredistributionpositiveoften。")

    print(f"   Class Predictions (Top 10): {pred_cls[:10].tolist()}")


if __name__ == "__main__":
    diagnose()