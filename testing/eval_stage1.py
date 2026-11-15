import torch
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms

# addprojectpath
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn
from ultralytics.utils.ops import xywh2xyxy


def evaluate():
    # 1. configuration
    config_path = 'configs/train/stage1.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Evaluation on {device}...")

    # 2. loadverifycollect (note：hereuse test_txt or train_txt doverify均optional，rootaccordingyouofsplit/divide)
    # assumeourfirstverify train set see if it is fitted
    eval_dataset = RLiViTDataset(cfg['data'], mode='train')

    # tosave省time，optionalby/withfirstonlyrun 200 tensorseeseemetric
    # eval_dataset.samples = eval_dataset.samples[:200]

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,  # verifyoptionalby/withuselarge Batch
        shuffle=False,
        collate_fn=fusion_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 3. loadmodel
    model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])

    # betterfirstload best.pt
    ckpt_path = f"runs/{cfg['experiment_name']}/best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"runs/{cfg['experiment_name']}/last.pt"

    print(f"📂 Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # metricstatistics
    stats = []  # [tp, conf, pred_cls, target_cls]
    iou_thres = 0.5
    conf_thres = 0.001  # evaluationwhenusuallyuseextremelowthresholdby/withcalculation mAP

    print("running evaluation...")
    for batch in tqdm(eval_loader):
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)

        # normalize
        if rgb.max() > 1.0: rgb = rgb.float() / 255.0
        if thermal.max() > 1.0: thermal = thermal.float() / 255.0

        inputs = {'rgb': rgb, 'thermal': thermal}

        with torch.no_grad():
            # inference
            preds = model(inputs)
            if isinstance(preds, tuple): preds = preds[0]

            # [bs, 67, 8400] -> [bs, 8400, 67]
            preds = preds.permute(0, 2, 1)

            # traverse Batch inofeachonetensorimage
            for i in range(len(preds)):
                # --- parseprediction ---
                pred = preds[i]  # [8400, 4+nc]

                # 1. parse Box (XYWH) and Score
                box_xywh = pred[:, :4]
                cls_scores = pred[:, 4:]
                conf, cls_id = cls_scores.max(1)

                # 2. thresholdfilterchoose
                mask = conf > conf_thres
                box_xywh = box_xywh[mask]
                conf = conf[mask]
                cls_id = cls_id[mask]

                if len(box_xywh) == 0:
                    continue

                # 3. 🔥🔥 criticalfixpositive：XYWH -> XYXY 🔥🔥
                box_xyxy = xywh2xyxy(box_xywh)

                # 4. NMS
                # simpleof Class-agnostic NMS
                keep = nms(box_xyxy, conf, 0.7)
                box_xyxy = box_xyxy[keep]
                conf = conf[keep]
                cls_id = cls_id[keep]

                # --- parsetruevalue ---
                b_idx = batch['batch_idx']
                target_mask = (b_idx == i)
                target_boxes = batch['bboxes'][target_mask].to(device)  # [N, 4] (cx, cy, w, h)
                target_cls = batch['cls'][target_mask].to(device)

                if len(target_boxes) == 0:
                    continue

                # 🔥 truevaluealsoneedconvert XYXY (if Dataset giveofis xywh)
                # rootaccording r_livit_dataset.py，itoutputofis normalized cx, cy, w, h
                # needfirst反normalize (x640)，thenconvert XYXY
                img_size = 640
                target_boxes = target_boxes * img_size
                target_boxes_xyxy = xywh2xyxy(target_boxes)

                # --- calculation Matches (TP/FP) ---
                # calculationpredictionboxandtruevalueboxof IoU
                ious = box_iou(box_xyxy, target_boxes_xyxy)  # [M, N]

                # simpleofgreedymatch
                matches = []  # record matched predictionsboxindex
                detected_targets = set()

                if ious.numel() > 0:
                    # geteachapredictionboxmaximumof IoU
                    max_iou, max_idx = ious.max(1)

                    for p_idx, (iou, t_idx) in enumerate(zip(max_iou, max_idx)):
                        if iou > iou_thres and t_idx.item() not in detected_targets:
                            # also need tocheckcategorywhetheroneconsistent
                            if cls_id[p_idx] == target_cls[t_idx]:
                                matches.append(p_idx)
                                detected_targets.add(t_idx.item())

                # recordresult [tp, conf]
                # heredosimplifycalculation：onlycalculate Recall
                tp_count = len(matches)
                total_targets = len(target_boxes)
                stats.append((tp_count, total_targets))

    # aggregatetotalcalculation Recall
    total_tp = sum(x[0] for x in stats)
    total_objects = sum(x[1] for x in stats)

    recall = total_tp / total_objects if total_objects > 0 else 0
    print(f"\n✅ Evaluation Finished!")
    print(f"🎯 Total Objects: {total_objects}")
    print(f"🎯 Correctly Detected (TP): {total_tp}")
    print(f"🚀 Estimated Recall @ IoU 0.5: {recall:.4f}")

    if recall > 0.3:
        print("\n🎉 PASS! modelalreadypreparegoodperform Stage 2 (GNN) training。")
    else:
        print("\n⚠️ FAIL. Recall stilltoolow，pleasecheck NMS parameterortrainingwhen长。")


if __name__ == '__main__':
    evaluate()