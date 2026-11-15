# verify_stage1_1280.py
import torch
import cv2
import numpy as np
import os
import yaml
import torchvision
from torch.utils.data import DataLoader
import sys

# addprojectrootdirectory
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn
from ultralytics.utils.ops import xywh2xyxy

# 8classmappingtable (forshow)
ID2CLASS = {
    0: 'Car', 1: 'Person', 2: 'Bike', 3: 'Motor',
    4: 'Bus', 5: 'Truck', 6: 'Tram', 7: 'Scooter'
}


def simple_nms_final(prediction, conf_thres=0.1, iou_thres=0.5):
    """
    NMS afterprocessing (adapt 1280 resolution)
    prediction: [bs, 4+nc, N_anchors]
    """
    # dimensionconvert: [bs, N, 4+nc]
    prediction = prediction.permute(0, 2, 1)

    bs = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        # 1. parse Box (XYWH) and Class Score
        box = x[:, :4]
        cls_score = x[:, 4:]

        # 2. gettomaximumconfidence
        conf, j = cls_score.max(1, keepdim=True)

        # 3. filterlowconfidence
        mask = conf.view(-1) > conf_thres
        x_filtered = torch.cat((box, conf, j.float()), 1)[mask]

        if x_filtered.shape[0] == 0:
            continue

        # 4. XYWH -> XYXY
        box_xywh = x_filtered[:, :4]
        box_xyxy = xywh2xyxy(box_xywh)

        # 5. multiplecategory NMS
        # throughoffset trick implementnotsamecategorymutualnotsuppression
        c = x_filtered[:, 5:6] * 7680
        boxes, scores = box_xyxy + c, x_filtered[:, 4]

        i = torchvision.ops.nms(boxes, scores, iou_thres)

        output[xi] = torch.cat((box_xyxy[i], x_filtered[i, 4:]), 1)

    return output


def verify():
    print("🕵️‍♂️ [Verify Stage 1] startvalidation 1280 highcleartrainingeffect...")

    # 1. readconfiguration
    config_path = 'configs/train/stage1.yaml'
    if not os.path.exists(config_path):
        print(f"❌ findnottoconfigurationfile: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. loaddataset (onlygetfirst 20 tensorseeeffect)
    print("📚 loadingdataset (ImgSize: 1280)...")
    dataset = RLiViTDataset(cfg['data'], mode='test')  # recommenduse test collect
    dataset.samples = dataset.samples[:20]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=fusion_collate_fn)

    # 3. initializationmodel
    model_path = cfg['model']['pretrained']
    num_classes = cfg['model']['num_classes']
    print(f"🏗️ initializationmodel: {model_path} (Classes: {num_classes})")

    model = YOLOWithFusion(model_path=model_path, num_classes=num_classes)

    # 4. loadtraininggoodofweight
    ckpt_path = f"runs/{cfg['experiment_name']}/best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"runs/{cfg['experiment_name']}/last.pt"

    if not os.path.exists(ckpt_path):
        print(f"❌ findnottotrainingweight: {ckpt_path}")
        return

    print(f"📂 loading Checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. inferenceanddrawimage
    output_dir = "stage1_1280_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"🎨 resultwillsaveto: {output_dir}/")

    for i, batch in enumerate(dataloader):
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)

        if rgb.max() > 1.0: rgb = rgb.float() / 255.0
        if thermal.max() > 1.0: thermal = thermal.float() / 255.0

        inputs = {'rgb': rgb, 'thermal': thermal}

        with torch.no_grad():
            preds = model(inputs)
            if isinstance(preds, tuple): preds = preds[0]

            # NMS
            final_preds = simple_nms_final(preds, conf_thres=0.25, iou_thres=0.45)

        # drawimageprepare
        img_vis = rgb[0].cpu().permute(1, 2, 0).numpy() * 255
        img_vis = cv2.cvtColor(img_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
        h, w = img_vis.shape[:2]

        # --- A. drawpredictionbox (red) ---
        pred = final_preds[0]
        if len(pred) > 0:
            print(f"   📸 imagepiece {i}: checkout {len(pred)} atarget")
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                cls_id = int(cls)
                cls_name = ID2CLASS.get(cls_id, str(cls_id))

                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            print(f"   ⚠️ imagepiece {i}: notcheckouttarget")

        # --- B. drawtruevaluebox (greencolor) ---
        # 🔥 fixlogic：becausefor/as batch_size=1，allby/with batch['bboxes'] inofallboxallattributeforthistensorimage
        # if batch_size > 1，thenneeduse batch['batch_idx'] == i tofilterchoose

        targets = batch['bboxes']  # [N, 4]
        tcls = batch['cls']  # [N]

        if len(targets) > 0:
            for box, cls in zip(targets, tcls):
                # RLiViTDataset returnofisnormalize [cx, cy, w, h]
                cx, cy, bw, bh = box.cpu().numpy()
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # cls_name = ID2CLASS.get(int(cls), str(int(cls)))
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # cv2.putText(img_vis, "GT", (x1, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imwrite(f"{output_dir}/vis_{i}.jpg", img_vis)

    print("✅ validationcompleted！please check stage1_1280_results fileclamp")


if __name__ == '__main__':
    verify()