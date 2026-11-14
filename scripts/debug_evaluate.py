import torch
import os
import sys
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 导入自定义模块
from models.yolo.yolo_with_fusion_ablation import YOLOWithFusionAblation
from data.datasets.r_livit_dataset import RLiViTDataset
from utils.device_utils import setup_device
from utils.metrics.evaluation_metrics import ap_per_class, box_iou

# ================= 🔧 调试配置 =================
# 只跑 Baseline 进行诊断
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
    'batch_size': 2,  # 小批次方便调试
    'num_workers': 0
}


# ================= 🛠️ 内置工具函数 =================
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
    """检查检测头的权重状态"""
    print("\n🔍 --- 权重健康度检查 ---")
    # 检查 Head 的分类分支
    cv3_weight = model.head.cv3[0][0].conv.weight
    print(f"   Head.cv3 (Cls) Weight Mean: {cv3_weight.mean().item():.6f} | Std: {cv3_weight.std().item():.6f}")

    # 检查 Head 的 DFL 分支 (负责框解码)
    if hasattr(model.head, 'dfl'):
        dfl_weight = model.head.dfl.conv.weight
        print(f"   Head.dfl (Box) Weight Mean: {dfl_weight.mean().item():.6f} | Std: {dfl_weight.std().item():.6f}")
    else:
        print("   ⚠️ Head 没有 DFL 模块？这会导致无法解码！")


def smart_load_weights_debug(model, ckpt_path, device):
    print(f"\n📥 正在加载权重: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model'].state_dict() if hasattr(ckpt.get('model'), 'state_dict') else ckpt.get('model', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 统计匹配情况
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # 尝试标准 YOLO 迁移
        if any(k.startswith('model.0') for k in ckpt_keys) and not any(k.startswith('rgb_backbone') for k in ckpt_keys):
            print("   ⚠️ 检测到标准 YOLO 格式，执行键名映射...")
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

        # 检查 Head 权重是否匹配
        head_keys_model = {k for k in model_keys if 'head' in k}
        head_keys_ckpt = {k for k in ckpt_keys if 'head' in k}
        missing_head = head_keys_model - head_keys_ckpt

        print(f"   📊 键名统计: 模型总键数={len(model_keys)}, 权重文件键数={len(ckpt_keys)}")
        print(f"   🎯 Head层匹配: 模型Head键={len(head_keys_model)}, 权重Head键={len(head_keys_ckpt)}")

        if len(missing_head) > 0:
            print(f"   ❌ 警告！检测头有 {len(missing_head)} 个参数未加载！这会导致预测全错！")
            print(f"   示例缺失键: {list(missing_head)[:3]}")
        else:
            print("   ✅ 检测头参数完全匹配！")

        model.load_state_dict(state_dict, strict=False)
        return True
    except Exception as e:
        print(f"   ❌ 加载报错: {e}")
        return False


def debug_inference():
    device = setup_device()
    print(f"🚀 启动诊断模式 (Device: {device})")

    # 1. 模型初始化
    config = EXPERIMENT
    model = YOLOWithFusionAblation(
        model_path=config['base_model'],
        num_classes=8,
        fusion_mode=config['mode']
    )
    model.to(device)

    # 2. 权重加载 & 检查
    if not smart_load_weights_debug(model, config['path'], device):
        return

    inspect_weights(model)
    model.eval()

    # 3. 数据加载
    dataset = RLiViTDataset(DATA_CFG, mode='test')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=my_collate_fn)

    print("\n🧐 --- 推理输出诊断 (只跑 1 个 Batch) ---")
    batch = next(iter(dataloader))
    rgb = batch['rgb'].to(device)
    thermal = batch['thermal'].to(device)

    with torch.no_grad():
        inputs = {'rgb': rgb, 'thermal': thermal}
        preds = model(inputs)

        # 处理 Tuple
        if isinstance(preds, tuple):
            print("   ℹ️ 模型返回了 Tuple (符合预期)")
            preds = preds[0]  # [bs, 8400, 4+nc]
        else:
            print(f"   ℹ️ 模型返回了 {type(preds)}")

        # 检查数值
        print(f"   📦 输出 Tensor Shape: {preds.shape}")

        # 检查置信度
        # preds: [bs, 4+nc, anchors] or [bs, anchors, 4+nc]
        # v8 Detect Head output is usually [bs, 4+nc, 8400]
        if preds.shape[1] > preds.shape[2]:  # [bs, 8400, 84] -> [bs, 84, 8400]
            preds = preds.transpose(1, 2)
            print("   ℹ️以此转置 Tensor 形状")

        # 此时 preds 应为 [bs, 4+nc, anchors] -> 8+4=12
        # 或者 ultralytics 的输出是 [bs, 4+nc, 8400]
        # 让我们打印来看看

        # Ultralytics 的 detect head 输出通常是 [bs, 4+nc, 8400] (nc=8, total=12)
        conf_scores = preds[:, 4:, :].amax(dim=1)  # [bs, 8400]
        max_conf = conf_scores.max().item()
        mean_conf = conf_scores.mean().item()

        print(f"   🔥 最大置信度 (Max Conf): {max_conf:.6f}")
        print(f"   📉 平均置信度 (Mean Conf): {mean_conf:.6f}")

        if max_conf < 0.01:
            print("   ❌ 严重警告：最大置信度过低！模型几乎没有检测到任何目标。")
            print("      可能原因：1. 权重加载失败（虽然不报错） 2. 输入数据归一化错误 3. 训练失败")
        else:
            print("   ✅ 置信度正常。")

        # 检查坐标解码
        # 取置信度最高的一个框
        b_idx, a_idx = torch.where(conf_scores == max_conf)
        b_idx, a_idx = b_idx[0], a_idx[0]

        best_box = preds[b_idx, :4, a_idx]  # x, y, w, h
        print(f"   📍 最佳框坐标 (Raw): {best_box.tolist()}")

        if best_box.max() < 2.0:
            print("   ❌ 严重警告：框坐标似乎是归一化的 (0~1) 或者 Grid 偏移量！")
            print("      这意味着 Detect 头没有自动解码。mAP 计算会失败（因为我们期望像素坐标）。")
        else:
            print("   ✅ 框坐标看起来是像素级的 (>1)。")


if __name__ == "__main__":
    debug_inference()