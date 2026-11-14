import torch
import yaml
import cv2
import os
import sys
import numpy as np
import random

# 添加项目根目录，确保能导入模块
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
# 确保导入的是我们自定义的 postprocess，避免 ultralytics 版本兼容问题
from utils.postprocess import non_max_suppression


def run_demo():
    print("============================================================")
    print("🎬 正在启动可视化 Demo (Method 1: 中途检查模式)")
    print("============================================================")

    # 1. 加载配置
    config_path = 'configs/train/stage1.yaml'
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'runs/demo_results'
    os.makedirs(output_dir, exist_ok=True)

    # 2. 确定权重文件路径
    # 优先寻找 'check.pt' (这是你在训练中途复制出来的文件)
    if os.path.exists('check.pt'):
        weights_path = 'check.pt'
        print(f"✅ 发现临时检查权重: {weights_path} (优先加载)")
    else:
        # 如果没有 check.pt，尝试找 best.pt 或 last.pt
        weights_path = f"runs/{cfg['experiment_name']}/best.pt"
        if not os.path.exists(weights_path):
            weights_path = f"runs/{cfg['experiment_name']}/last.pt"

    if not os.path.exists(weights_path):
        print(f"❌ 找不到任何权重文件！请先确保你已经复制了权重到项目根目录 (check.pt)")
        print(f"   命令示例: copy runs\\stage1_rlivit_fusion\\last.pt check.pt")
        return

    # 3. 加载模型
    print(f"🔄 正在加载模型: {weights_path} ...")
    try:
        model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 加载测试集 (随机抽取图片)
    print("🔄 正在加载测试集样本...")
    dataset = RLiViTDataset(cfg['data'], mode='test')

    # 随机抽取 8 张图片进行展示
    num_samples = 8
    if len(dataset) < num_samples: num_samples = len(dataset)
    indices = random.sample(range(len(dataset)), num_samples)

    print(f"🚀 开始推理 {num_samples} 张图片...")

    for idx in indices:
        sample = dataset[idx]

        # 准备输入 Tensor (已经符合 RGB 格式)
        rgb_tensor = sample['rgb'].unsqueeze(0).to(device)  # [1, 3, 640, 640]
        thermal_tensor = sample['thermal'].unsqueeze(0).to(device)

        # 准备可视化图像
        # Dataset 返回的是 Normalized RGB Tensor (0-1)，我们需要转回 BGR (0-255) 给 OpenCV 显示
        vis_img = sample['rgb'].permute(1, 2, 0).cpu().numpy()  # [H, W, 3], RGB
        vis_img = (vis_img * 255).astype(np.uint8).copy()  # 0-255
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)  # RGB -> BGR (关键修复！)

        # 推理
        with torch.no_grad():
            preds = model({'rgb': rgb_tensor, 'thermal': thermal_tensor})
            # NMS 后处理
            # conf_thres=0.25: 只显示置信度 > 0.25 的框，避免太乱
            # 如果训练初期什么都看不到，可以将 conf_thres 调低到 0.05
            preds = non_max_suppression(preds, conf_thres=0.01, iou_thres=0.5)

        # 画框
        det = preds[0]  # 第一张图的结果

        if len(det):
            # 坐标已经是 640x640 尺度了，直接画
            for *xyxy, conf, cls in det:
                c = int(cls)
                cls_name = cfg['model']['names'][c]
                label = f"{cls_name} {conf:.2f}"

                # 坐标取整
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                # 画矩形 (绿色)
                cv2.rectangle(vis_img, p1, p2, (0, 255, 0), 2)

                # 画标签背景和文字
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
                cv2.rectangle(vis_img, p1, c2, (0, 255, 0), -1, cv2.LINE_AA)  # filled
                cv2.putText(vis_img, label, (p1[0], p1[1] - 2), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        else:
            # 如果没检测到，写个提示
            cv2.putText(vis_img, "No Detection", (30, 30), 0, 1, (0, 0, 255), 2)

        # 保存结果
        save_path = os.path.join(output_dir, f"demo_{idx}.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"   已保存: {save_path}")

    print(f"\n✅ 所有演示图片已保存至: {os.path.abspath(output_dir)}")
    print("   请打开文件夹查看检测效果！")


if __name__ == '__main__':
    run_demo()