import os
import sys
import torch
import cv2
import json
import numpy as np
import torchvision
from pathlib import Path
from tqdm import tqdm  # 🔥 引入进度条，因为跑全量需要知道进度

# 自动定位项目根目录
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from models.gnn.hierarchical_gnn import HierarchicalGNN
from utils.device_utils import setup_device

# ================= 🔧 配置区域 =================

# 🔥🔥🔥 阈值调整区 (手动修改这里) 🔥🔥🔥
CONF_THRES = 0.40  # GNN 检测置信度阈值 (建议尝试 0.20 - 0.40)

# 数据集目录
DATASET_DIR = "stage2_gnn_dataset_GAP_1280"
# 权重路径
CHECKPOINT = r"D:\YOLO-GNN-Fusion\runs\stage2_gnn_GAP_1280\epoch_50.pth"
# 🔥 输出目录改为 comparison_vis/ours
OUTPUT_DIR = os.path.join(project_root, "comparison_vis", "ours")

# 类别映射 (0-based)
CLASS_MAP = {
    0: 'Car', 1: 'Person', 2: 'Bicycle', 3: 'Motorcycle',
    4: 'Bus', 5: 'Truck', 6: 'Tramway', 7: 'Escooter'
}

# 颜色表 (BGR)
COLORS = {
    0: (0, 255, 0),  # Car: Green
    1: (0, 255, 255),  # Person: Yellow
    2: (0, 165, 255),  # Bicycle: Orange
    3: (255, 144, 30),  # Motorcycle
    4: (0, 0, 255),  # Bus: Red
    5: (255, 0, 255),  # Truck: Magenta
    'default': (200, 200, 200)
}


# ================= 🛠️ 工具函数 =================
def scale_coords(img1_shape, coords, img0_shape):
    # 将坐标从 1280x1280 还原回原图尺寸
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


def apply_nms(boxes, scores, labels, iou_thres=0.3):
    """ 对 GNN 的输出进行 NMS 去重 """
    if boxes.numel() == 0: return [], [], []

    max_wh = 4096
    c = labels.float() * max_wh
    boxes_offset = boxes + c.unsqueeze(1)

    keep_indices = torchvision.ops.nms(boxes_offset, scores, iou_thres)

    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def draw_box(img, box, cls_id, score):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(cls_id, COLORS['default'])
    label = f"{CLASS_MAP.get(cls_id, 'Unk')} {score:.2f}"

    # 1. 画框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 2. 画文字背景 (上方)
    font_scale = 0.6
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    t_y1 = y1
    t_y2 = y1 + th + baseline + 5
    # 如果文字框超出上边界，画在框内
    if t_y1 - th < 0:
        t_y1 = y1
        t_y2 = y1 + th + 10
    else:
        t_y2 = y1
        t_y1 = y1 - th - 10

    cv2.rectangle(img, (x1, t_y1), (x1 + tw + 5, t_y2), color, -1)

    # 3. 画文字
    cv2.putText(img, label, (x1 + 2, t_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    device = setup_device()
    print(f"🚀 全量可视化开始 (阈值={CONF_THRES})...")
    print(f"📂 结果将保存至: {OUTPUT_DIR}")

    # 1. 加载数据列表
    list_path = os.path.join(project_root, DATASET_DIR, "stage2_test_list.json")
    if not os.path.exists(list_path):
        print(f"❌ 找不到列表文件: {list_path}")
        return

    with open(list_path, 'r') as f:
        samples = json.load(f)
    print(f"📄 共发现 {len(samples)} 张测试图片")

    # 2. 自动检测维度
    p0 = samples[0]['path']
    if not os.path.isabs(p0): p0 = os.path.join(project_root, p0)
    feat_dim = torch.load(p0, map_location='cpu')['features'].shape[1]

    # 3. 加载模型
    model = HierarchicalGNN(node_feature_dim=feat_dim, hidden_dim=256).to(device)
    head_ex = torch.nn.Linear(128, 1).to(device)
    head_cl = torch.nn.Linear(128, 8).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        head_ex.load_state_dict(ckpt['head_exist'])
        head_cl.load_state_dict(ckpt['head_cls'])
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval();
    head_ex.eval();
    head_cl.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. 遍历所有样本 (使用 tqdm)
    for info in tqdm(samples, desc="Processing Images"):
        try:
            pt_path = info['path']
            if not os.path.isabs(pt_path): pt_path = os.path.join(project_root, pt_path)

            parts = info['id'].split('_')
            # 根据你的路径结构拼接图片路径
            img_path = f"D:/BaiduNetdiskDownload/R-LiViT/R-LiViT/R-LiViT_RGB-T/rgb/{parts[0]}/{parts[1]}.png"
            if not os.path.exists(img_path): continue

            # 加载特征
            data = torch.load(pt_path, map_location=device)
            feat = torch.nn.functional.normalize(torch.nan_to_num(data['features'].float(), 0.0).clamp(-100, 100), p=2,
                                                 dim=1)

            # 坐标转换: Norm XYWH -> Pixel XYXY
            norm_xywh = data['boxes'].float()
            target_size = 1280

            pixel_xywh = norm_xywh * target_size
            pixel_xyxy = torch.zeros_like(pixel_xywh)
            pixel_xyxy[:, 0] = pixel_xywh[:, 0] - pixel_xywh[:, 2] / 2
            pixel_xyxy[:, 1] = pixel_xywh[:, 1] - pixel_xywh[:, 3] / 2
            pixel_xyxy[:, 2] = pixel_xywh[:, 0] + pixel_xywh[:, 2] / 2
            pixel_xyxy[:, 3] = pixel_xywh[:, 1] + pixel_xywh[:, 3] / 2

            # GNN 推理
            scores = data['scores'][:, 0]
            with torch.no_grad():
                out, keep, _ = model([feat], [norm_xywh], [scores])
                p_ex = torch.sigmoid(head_ex(out))
                p_cl = torch.argmax(head_cl(out), dim=1)

            # 加载原图用于画图
            img0 = cv2.imread(img_path)
            if img0 is None: continue
            h0, w0 = img0.shape[:2]

            # 还原到原图尺寸
            orig_boxes_all = scale_coords((1280, 1280), pixel_xyxy.clone(), (h0, w0))

            # 收集候选框
            candidates_boxes = []
            candidates_scores = []
            candidates_cls = []

            if keep and len(keep) > 0:
                keep_idxs = keep[0]
                for i, idx in enumerate(keep_idxs):
                    conf = p_ex[i].item()
                    # 🔥 使用配置的阈值
                    if conf > CONF_THRES:
                        cls = p_cl[i].item()
                        if cls not in CLASS_MAP: continue

                        raw_idx = int(idx)
                        candidates_boxes.append(orig_boxes_all[raw_idx])
                        candidates_scores.append(conf)
                        candidates_cls.append(cls)

            # NMS 去重
            if len(candidates_boxes) > 0:
                cb = torch.stack(candidates_boxes)
                cs = torch.tensor(candidates_scores).to(device)
                cc = torch.tensor(candidates_cls).to(device)

                final_boxes, final_scores, final_cls = apply_nms(cb, cs, cc, iou_thres=0.3)

                # 画图
                for b, s, c in zip(final_boxes, final_scores, final_cls):
                    draw_box(img0, b, c.item(), s.item())

            # 保存
            save_name = f"{parts[0]}_{parts[1]}.jpg"
            save_p = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_p, img0)

        except Exception as e:
            print(f"Error processing {info['id']}: {e}")
            continue

    print(f"\n✅ 全部处理完成！请查看: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()