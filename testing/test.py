# testing/test.py - 完全修复版

import torch
import yaml
import argparse
from pathlib import Path
import sys
import os
import json
import time

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.full_pipeline import YOLOGNNFusion


# ========== 定义必要的工具函数 ==========

def setup_device():
    """统一的设备设置函数"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


def get_data_root():
    """获取统一的数据根目录"""
    possible_paths = [
        'D:/BaiduNetdiskDownload/16356714/R-LiViT',
        'D:/BaiduNetdiskDownload/R-LiViT/R-LiViT',
        './data/R-LiViT',
        'R-LiViT'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 使用数据路径: {path}")
            return path

    return possible_paths[0]


def safe_load_checkpoint(model, checkpoint_path, device):
    """安全加载检查点，兼容不同格式"""
    print(f"加载检查点: {checkpoint_path}")

    try:
        if os.path.exists(checkpoint_path):
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # 提取模型状态
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 0)
                best_metric = checkpoint.get('best_metric', 0.0)
            else:
                state_dict = checkpoint
                epoch = 0
                best_metric = 0.0

            # 加载模型参数
            model_dict = model.state_dict()

            # 过滤不匹配的键
            filtered_state_dict = {}
            for key, param in state_dict.items():
                if key in model_dict:
                    if param.shape == model_dict[key].shape:
                        filtered_state_dict[key] = param
                    else:
                        print(f"⚠️ 形状不匹配: {key} - {param.shape} vs {model_dict[key].shape}")
                else:
                    # 尝试查找相似的键
                    found = False
                    for model_key in model_dict.keys():
                        if key.replace('module.', '') == model_key.replace('module.', ''):
                            if param.shape == model_dict[model_key].shape:
                                filtered_state_dict[model_key] = param
                                found = True
                                break

                    if not found:
                        print(f"⚠️ 未找到对应键: {key}")

            # 加载参数
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict, strict=False)

            print(f"✅ 成功加载检查点")
            print(f"   加载参数: {len(filtered_state_dict)}/{len(model_dict)}")
            print(f"   训练轮次: {epoch}")

            return model, epoch, best_metric

        else:
            print(f"❌ 检查点不存在: {checkpoint_path}")
            return model, 0, 0.0

    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        import traceback
        traceback.print_exc()
        return model, 0, 0.0


def simple_collate_fn(batch):
    """
    简单的collate函数，处理R-LiViT数据集
    """
    if not batch:
        return {}

    collated = {}

    # 处理RGB图像
    rgb_images = []
    thermal_images = []
    targets = []
    image_paths = []

    for item in batch:
        if isinstance(item, dict):
            # 提取RGB
            if 'rgb' in item:
                rgb_images.append(item['rgb'])
            elif 'image' in item:
                rgb_images.append(item['image'])

            # 提取热成像
            if 'thermal' in item:
                thermal_images.append(item['thermal'])

            # 提取目标
            if 'target' in item:
                targets.append(item['target'])
            elif 'annotations' in item:
                targets.append(item['annotations'])

            # 提取图像路径
            if 'image_path' in item:
                image_paths.append(item['image_path'])

    # 堆叠图像
    if rgb_images:
        collated['rgb'] = torch.stack(rgb_images, dim=0)

        # 处理热成像
        if thermal_images and len(thermal_images) == len(rgb_images):
            collated['thermal'] = torch.stack(thermal_images, dim=0)
        else:
            # 创建占位符
            collated['thermal'] = torch.zeros_like(collated['rgb'])

    # 目标保持列表格式
    if targets:
        collated['targets'] = targets

    # 图像路径
    if image_paths:
        collated['image_paths'] = image_paths

    return collated


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试YOLO-GNN融合模型')

    parser.add_argument('--config', type=str, default='configs/test/evaluation.yaml',
                        help='测试配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录（默认自动检测）')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='要评估的数据集划分')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备类型 (auto/cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小（建议为1以确保兼容性）')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载工作进程数（0确保兼容性）')
    parser.add_argument('--output_dir', type=str, default='results/test_output',
                        help='输出目录')
    parser.add_argument('--no_risk_assessment', action='store_true',
                        help='禁用风险评估（与extract_stage1_features.py保持一致）')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='最大处理批次数量（用于快速测试）')

    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"⚠️ 配置文件不存在: {config_path}")
        return {}


def create_model(args, config, device):
    """创建模型 - 与extract_stage1_features.py保持一致"""
    print("创建模型...")

    # 从配置或使用默认值
    model_config = config.get('model', {})

    model = YOLOGNNFusion(
        num_classes=model_config.get('num_classes', 3),
        node_feature_dim=model_config.get('node_feature_dim', 128),
        gnn_hidden_dim=model_config.get('gnn_hidden_dim', 256),
        use_risk_assessment=not args.no_risk_assessment,
        use_lidar=model_config.get('use_lidar', False),
        use_enhanced_risk=model_config.get('use_enhanced_risk', True),
        confidence_threshold=model_config.get('confidence_threshold', 0.1),
        gnn_max_nodes=model_config.get('gnn_max_nodes', 50)
    )
    model.to(device)
    model.eval()

    return model


def create_simple_dataset(data_root, split):
    """创建简单的数据集（不使用复杂的RLiViTDataset）"""
    print(f"创建简单数据集用于{split}划分...")

    samples = []
    data_path = Path(data_root)

    # 查找RGB图像
    rgb_base = data_path / "R-LiViT_RGB-T" / "rgb"
    if rgb_base.exists():
        # 限制样本数量用于测试
        max_sequences = 2
        max_images_per_sequence = 2

        seq_count = 0
        for seq_dir in sorted(rgb_base.iterdir()):
            if seq_dir.is_dir():
                img_count = 0
                for img_file in sorted(seq_dir.glob("*.png")):
                    if img_count >= max_images_per_sequence:
                        break

                    # 查找对应的热成像
                    thermal_path = data_path / "R-LiViT_RGB-T" / "thermal" / seq_dir.name / img_file.name

                    sample = {
                        'image_path': str(img_file),
                        'thermal_path': str(thermal_path) if thermal_path.exists() else None,
                        'sequence': seq_dir.name,
                        'image_name': img_file.stem
                    }
                    samples.append(sample)
                    img_count += 1

                seq_count += 1
                if seq_count >= max_sequences:
                    break

    print(f"找到 {len(samples)} 个样本")
    return samples


class SimpleDataset(torch.utils.data.Dataset):
    """简单数据集类"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        import numpy as np

        sample = self.samples[idx]

        # 加载RGB图像
        rgb_path = sample['image_path']
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"❌ 无法加载图像: {rgb_path}")
            rgb_img = np.zeros((416, 416, 3), dtype=np.uint8)

        # 调整大小
        rgb_img = cv2.resize(rgb_img, (416, 416))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1) / 255.0

        # 加载热成像
        thermal_path = sample.get('thermal_path')
        if thermal_path and os.path.exists(thermal_path):
            thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            if thermal_img is None:
                thermal_img = np.zeros((416, 416), dtype=np.uint8)

            thermal_img = cv2.resize(thermal_img, (416, 416))
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)
            thermal_tensor = torch.from_numpy(thermal_img).float().permute(2, 0, 1) / 255.0
        else:
            thermal_tensor = torch.zeros_like(rgb_tensor)

        return {
            'rgb': rgb_tensor,
            'thermal': thermal_tensor,
            'image_path': rgb_path,
            'sample_info': sample
        }


def run_evaluation(model, dataloader, device, args):
    """运行评估"""
    print(f"\n开始评估...")

    model.eval()
    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.max_batches:
                print(f"  达到最大批次限制 ({args.max_batches})")
                break

            print(f"  处理批次 {batch_idx + 1}/{len(dataloader)}")

            # 提取图像数据
            if 'rgb' in batch:
                rgb_images = batch['rgb'].to(device)
                thermal_images = batch['thermal'].to(device)
                image_paths = batch.get('image_paths', [])
            else:
                print(f"⚠️ 批次中没有RGB图像数据")
                continue

            # 前向传播
            try:
                outputs = model(rgb_images, thermal_images)

                # 收集结果
                batch_result = {
                    'batch_idx': batch_idx,
                    'num_images': len(rgb_images),
                    'stage1_detections': len(outputs.get('stage1_detections', [])),
                    'enhanced_detections': len(outputs.get('enhanced_detections', [])),
                    'has_risk_assessment': outputs.get('risk_assessment') is not None,
                    'gnn_features_shape': list(outputs.get('gnn_features', torch.Tensor()).shape),
                    'image_paths': image_paths
                }
                all_results.append(batch_result)

                print(f"    图像数量: {batch_result['num_images']}")
                print(f"    第一阶段检测: {batch_result['stage1_detections']}")
                print(f"    增强检测: {batch_result['enhanced_detections']}")

                # 显示一些检测详情
                if outputs.get('stage1_detections'):
                    detections = outputs['stage1_detections']
                    if len(detections) > 0:
                        print(
                            f"    检测置信度范围: {min(d.get('confidence', 0) for d in detections):.3f}-{max(d.get('confidence', 0) for d in detections):.3f}")

            except Exception as e:
                print(f"❌ 批次 {batch_idx} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    return all_results


def main():
    """主测试函数"""
    args = parse_args()

    print("=" * 60)
    print("YOLO-GNN 融合模型测试")
    print("=" * 60)

    # 设置设备
    device = setup_device()

    # 获取数据路径
    data_root = args.data_root if args.data_root else get_data_root()
    print(f"数据根目录: {data_root}")

    # 检查路径是否存在
    if not os.path.exists(data_root):
        print(f"❌ 数据根目录不存在: {data_root}")
        print("请检查数据路径或使用 --data_root 参数指定正确路径")
        return

    # 加载配置
    config = load_config(args.config)

    # 创建模型
    model = create_model(args, config, device)

    # 加载检查点
    model, epoch, best_metric = safe_load_checkpoint(model, args.checkpoint, device)

    # 创建简单的数据集和数据加载器
    samples = create_simple_dataset(data_root, args.split)

    if not samples:
        print("❌ 没有找到样本数据")
        return

    dataset = SimpleDataset(samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=simple_collate_fn,
        pin_memory=False
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"数据加载器批次数量: {len(dataloader)}")

    # 运行评估
    results = run_evaluation(model, dataloader, device, args)

    # 输出总结
    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"模型检查点: {args.checkpoint}")
    print(f"训练轮次: {epoch}")
    print(f"最佳指标: {best_metric:.4f}")
    print(f"处理批次: {len(results)}")

    if results:
        total_images = sum(r['num_images'] for r in results)
        total_stage1 = sum(r['stage1_detections'] for r in results)
        total_enhanced = sum(r['enhanced_detections'] for r in results)

        print(f"总处理图像: {total_images}")
        print(f"第一阶段总检测数: {total_stage1}")
        print(f"增强后总检测数: {total_enhanced}")

        if total_images > 0:
            print(f"平均每图像检测数: {total_stage1 / total_images:.2f}")
            print(f"检测率: {total_stage1 / (total_images * 10):.2%}")

    print("=" * 60)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"test_results_{Path(args.checkpoint).stem}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'epoch': epoch,
            'best_metric': best_metric,
            'device': str(device),
            'args': vars(args),
            'results': results,
            'summary': {
                'total_batches': len(results),
                'total_images': sum(r['num_images'] for r in results) if results else 0,
                'total_stage1_detections': sum(r['stage1_detections'] for r in results) if results else 0,
                'total_enhanced_detections': sum(r['enhanced_detections'] for r in results) if results else 0,
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"结果保存到: {result_file}")

    return results


if __name__ == '__main__':
    main()