import torch
import torch.nn.functional as F
import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# addprojectrootdirectory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.datasets.r_livit_dataset import RLiViTDataset


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou(box1, box2):
    def box_area(box): return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def run_debug_probe():
    # configuration
    data_cfg = {
        'root_dir': "D:/BaiduNetdiskDownload/R-LiViT/R-LiViT",
        'train_txt': "train.txt",
        'test_txt': "test.txt",
        'img_size': 1280
    }

    test_list_path = os.path.join(PROJECT_ROOT, "stage2_gnn_dataset_1280/stage2_test_list.json")
    if not os.path.exists(test_list_path):
        print(f"❌ list file not found: {test_list_path}")
        return

    print("📚 initialization GT dataset...")
    gt_dataset = RLiViTDataset(data_cfg, mode='test')

    # establish ID index
    id_to_idx = {}
    print("🗺️ establish ID mapping...")
    for i, s in enumerate(gt_dataset.samples):
        path_parts = s['xml'].replace('\\', '/').split('/')
        if len(path_parts) >= 2:
            key = f"{path_parts[-2]}_{os.path.splitext(path_parts[-1])[0]}"
            id_to_idx[key] = i

    print(f"✅ ID mappingestablishcompleted，total {len(id_to_idx)} asample")

    with open(test_list_path, 'r') as f:
        samples = json.load(f)

    X, y_exist = [], []
    limit = 200  # increase samples slightlyamountby/withmake surestatistics显著
    print(f"🚀 startloadingfirst {limit} asample (fixversion - alreadyadd detach)...")

    for i, info in enumerate(samples):
        if i >= limit: break

        pt_path = info['path']
        if not os.path.isabs(pt_path):
            pt_path = os.path.join(PROJECT_ROOT, pt_path)

        print(f"📄 [{i}] processing: {info['id']}")

        if not os.path.exists(pt_path):
            print(f"    ❌ filenotexist!")
            continue

        data = torch.load(pt_path, map_location='cpu')

        # checkfeature
        raw_feat = data.get('roi_features', data.get('features'))
        if raw_feat is None:
            print(f"    ❌ featurelacklose! Keys: {data.keys()}")
            continue

        # 🔥🔥🔥 core fix：addabove .detach() 🔥🔥🔥
        # even tensor havegradient，detach afteralsoability/cansafety/fullconvert numpy
        feat = F.normalize(torch.nan_to_num(raw_feat.float(), 0.0).clamp(-100, 100), p=2, dim=1).detach().numpy()

        # check Box (make sureconvertformat)
        boxes_xywh = data['boxes'].float()
        boxes = xywh2xyxy(boxes_xywh)  # make sureusing xyxy calculation IoU

        # match GT
        labels_exist = np.zeros(len(feat))

        if info['id'] in id_to_idx:
            idx = id_to_idx[info['id']]
            sample = gt_dataset[idx]

            if len(sample['labels']) > 0:
                # thenminorconfirm Dataset returnofis Normalized XYWH stillis Pixel XYWH?
                # RLiViTDataset return Normalized XYWH
                # pt inof boxes alsois Normalized XYWH
                # allby/withourallconvertgenerate Normalized XYXY tocalculate IoU isnotproblemof

                gt_xywh = sample['labels'][:, 2:]
                gt_box = xywh2xyxy(gt_xywh)

                ious = box_iou(boxes, gt_box)  # [N, M]
                if ious.numel() > 0:
                    max_iou, _ = ious.max(dim=1)
                    # strictstrictdefine义positivesample
                    labels_exist[max_iou > 0.5] = 1
        else:
            print(f"    ⚠️ ID notfindto: {info['id']}")

        X.append(feat)
        y_exist.append(labels_exist)

    if len(X) == 0:
        print("❌ final X for/asempty，notsuccessloadinganydata")
        return

    X = np.concatenate(X, axis=0)
    y_exist = np.concatenate(y_exist, axis=0)

    print(f"\n📊 loadingsuccess! totalRoIcount: {len(y_exist)}")
    print(f"   positivesample (IoU>0.5): {sum(y_exist == 1)}")
    print(f"   negativesample (IoU<=0.5): {sum(y_exist == 0)}")

    if sum(y_exist == 1) < 10:
        print("⚠️ positivesampletooleast，unable torunscore/divideclassdevice。saybright Stage 1 recallextremelow，orone Label Assignment stillhaveproblem。")
        return

    # simpleof Linear Probe
    print("\n🧪 positiveat/inrunlinepropertydetect (Linear Probe)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_exist, test_size=0.3, random_state=42, stratify=y_exist)

    # increase max_iter preventnotcollect敛
    clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    f1_pos = report['1.0']['f1-score']

    print("-" * 30)
    print(f"🎯 positivesample F1-Score: {f1_pos:.4f}")
    print("-" * 30)

    if f1_pos < 0.3:
        print("❌❌❌ conclusion：featureinvalid！")
        print("   simpleoflinepropertyscore/divideclassdevicecompletelyscore/dividenotopenpositivenegativesample。")
        print("   mayoriginalbecause：")
        print("   1. featureextractlayerlevelerrordigit (Shape Trap)")
        print("   2. RoI Align coordinateerrordigit (Format Trap)")
    elif f1_pos < 0.6:
        print("⚠️ conclusion：feature质amountone般")
        print("   containsonedefineinformation，butnoise声verylarge。")
    else:
        print("✅✅✅ conclusion：feature质amountbetter秀")
        print("   featurebook身ishaveareascore/dividedegreeof，GNN shouldability/cantraininggood。")


if __name__ == "__main__":
    run_debug_probe()