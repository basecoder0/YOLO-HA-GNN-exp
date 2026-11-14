import torch
import sys
import os
import yaml

sys.path.append(os.getcwd())
from models.yolo.yolo_with_fusion import YOLOWithFusion


def check_weights():
    print("🔍 正在检查预训练权重加载情况...")

    # 1. 加载官方 yolov8n.pt (作为参考)
    yolo_pt = torch.load('weights/yolov8n.pt', map_location='cpu', weights_only=False)
    yolo_dict = yolo_pt['model'].float().state_dict() if 'model' in yolo_pt else yolo_pt

    # 获取官方第一层卷积的权重 (model.0.conv.weight)
    # YOLOv8n 的第一个模块通常是 Conv
    ref_key = 'model.0.conv.weight'
    if ref_key not in yolo_dict:
        # 尝试找一个存在的 key
        ref_key = list(yolo_dict.keys())[0]

    ref_weight = yolo_dict[ref_key]
    print(f"✅ 官方权重 {ref_key} 均值: {ref_weight.mean().item():.6f}")

    # 2. 初始化我们的融合模型
    config_path = 'configs/train/stage1.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])

    # 3. 检查 RGB 主干是否继承了权重
    # 假设模型结构里有 backbone_rgb 或 similar
    # 我们遍历 model.named_parameters() 找看起来像 backbone 的层

    print("\n🔍 分析当前模型权重状态...")

    rgb_backbone_loaded = False
    thermal_backbone_loaded = False

    # 这里的名字取决于 YOLOWithFusion 的具体实现
    # 通常是 self.backbone (如果只是单流) 或者 self.rgb_backbone / self.thermal_backbone

    model_state = model.state_dict()

    # 寻找对应官方 ref_key 的层
    # 在 YOLOWithFusion 中，官方的 model.0 可能对应 model.rgb_stream.0 或者 model.backbone.0

    matched_keys = []
    for k, v in model_state.items():
        if '0.conv.weight' in k and 'backbone' in k:  # 粗略匹配第一层
            matched_keys.append(k)

    if not matched_keys:
        print("⚠️ 无法自动定位主干网络的第一层，将打印前 5 个层名供人工核对：")
        for k in list(model_state.keys())[:5]:
            print(f"   {k}")

    for k in matched_keys:
        current_weight = model_state[k]
        diff = torch.abs(current_weight - ref_weight).sum().item()

        # 如果权重完全一样 (Diff = 0)，说明加载成功但还没训练
        # 如果权重完全不同 (Diff 很大且看起来像随机分布)，说明加载失败，是随机初始化的
        # 如果权重接近但不为0，说明加载成功且经过了微调 (但在初始化阶段应该是一样的)

        # 此时我们是在检查 "初始化后" 的状态，理论上应该完全一致

        print(f"   层: {k}")
        print(f"   与官方权重差异: {diff:.6f}")

        if diff < 1e-5:
            print("   ✅ 判定: 权重加载成功！")
        else:
            print("   ❌ 判定: 权重不匹配！可能是随机初始化。")


if __name__ == '__main__':
    check_weights()