import torch
import yaml
import cv2
import os
import sys
import numpy as np
import random

# addprojectrootdirectory，make sureability/canimportmodule
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
# make sureimportis our custom postprocess，avoid ultralytics versioncompatibleproblem
from utils.postprocess import non_max_suppression


def run_demo():
    print("============================================================")
    print("🎬 positiveat/instartvisualization Demo (Method 1: in途checkmode)")
    print("============================================================")

    # 1. loadconfiguration
    config_path = 'configs/train/stage1.yaml'
    if not os.path.exists(config_path):
        print(f"❌ findnottoconfigurationfile: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'runs/demo_results'
    os.makedirs(output_dir, exist_ok=True)

    # 2. suredefineweightfilepath
    # betterfirstsearchfind 'check.pt' (thisisyouat/intrainingin途complexcontrolouttooffile)
    if os.path.exists('check.pt'):
        weights_path = 'check.pt'
        print(f"✅ foundtemporarycheckweight: {weights_path} (betterfirstloading)")
    else:
        # ifnothave check.pt，tryfind best.pt or last.pt
        weights_path = f"runs/{cfg['experiment_name']}/best.pt"
        if not os.path.exists(weights_path):
            weights_path = f"runs/{cfg['experiment_name']}/last.pt"

    if not os.path.exists(weights_path):
        print(f"❌ findnottoanyweightfile！please firstmake sureyoualreadycomplexcontrolpast tense markerweighttoprojectrootdirectory (check.pt)")
        print(f"   commandexample: copy runs\\stage1_rlivit_fusion\\last.pt check.pt")
        return

    # 3. loadmodel
    print(f"🔄 positiveat/inloadingmodel: {weights_path} ...")
    try:
        model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ model loadingsuccess！")
    except Exception as e:
        print(f"❌ model loadingfailed: {e}")
        return

    # 4. loadtestcollect (random抽getimagepiece)
    print("🔄 positiveat/inloadingtestcollectsample...")
    dataset = RLiViTDataset(cfg['data'], mode='test')

    # random抽get 8 tensorimagepieceperformshow
    num_samples = 8
    if len(dataset) < num_samples: num_samples = len(dataset)
    indices = random.sample(range(len(dataset)), num_samples)

    print(f"🚀 startinference {num_samples} tensorimagepiece...")

    for idx in indices:
        sample = dataset[idx]

        # prepareinput Tensor (alreadymatch RGB format)
        rgb_tensor = sample['rgb'].unsqueeze(0).to(device)  # [1, 3, 640, 640]
        thermal_tensor = sample['thermal'].unsqueeze(0).to(device)

        # preparevisualizationimage
        # Dataset returnofis Normalized RGB Tensor (0-1)，ourneedconvertfallback BGR (0-255) give OpenCV show
        vis_img = sample['rgb'].permute(1, 2, 0).cpu().numpy()  # [H, W, 3], RGB
        vis_img = (vis_img * 255).astype(np.uint8).copy()  # 0-255
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)  # RGB -> BGR (criticalfix！)

        # inference
        with torch.no_grad():
            preds = model({'rgb': rgb_tensor, 'thermal': thermal_tensor})
            # NMS afterprocessing
            # conf_thres=0.25: onlyshowconfidence > 0.25 ofbox，avoidtoo乱
            # iftrainingnothing can be seen in the early stage，optionalby/withwill conf_thres adjustlowto 0.05
            preds = non_max_suppression(preds, conf_thres=0.01, iou_thres=0.5)

        # draw bounding box
        det = preds[0]  # firsttensorimageofresult

        if len(det):
            # coordinatealreadyis 640x640 尺degreepast tense marker，directlydraw
            for *xyxy, conf, cls in det:
                c = int(cls)
                cls_name = cfg['model']['names'][c]
                label = f"{cls_name} {conf:.2f}"

                # coordinategetinteger
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                # draw矩shape (greencolor)
                cv2.rectangle(vis_img, p1, p2, (0, 255, 0), 2)

                # drawlabelbackgroundandtext
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
                cv2.rectangle(vis_img, p1, c2, (0, 255, 0), -1, cv2.LINE_AA)  # filled
                cv2.putText(vis_img, label, (p1[0], p1[1] - 2), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        else:
            # ifnotdetectionto，写aextractshow
            cv2.putText(vis_img, "No Detection", (30, 30), 0, 1, (0, 0, 255), 2)

        # saveresult
        save_path = os.path.join(output_dir, f"demo_{idx}.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"   alreadysave: {save_path}")

    print(f"\n✅ allperformshowimagepiecealreadysaveto: {os.path.abspath(output_dir)}")
    print("   pleaseopenfileclampcheckseedetectioneffect！")


if __name__ == '__main__':
    run_demo()