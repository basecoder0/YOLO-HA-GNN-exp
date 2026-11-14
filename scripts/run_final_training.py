# scripts/run_final_training_fixed.py
# !/usr/bin/env python3
"""
最终版第二阶段训练脚本 - 修复编码问题
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='最终版第二阶段训练')
    parser.add_argument('--config', type=str, default='configs/train/stage2.yaml')
    parser.add_argument('--features_dir', type=str, default='./stage1_features_diverse')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='D:/YOLO-GNN-Fusion/checkpoints_final/best_model_epoch_20_loss_0.2275.pth')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_simple', action='store_true', help='使用简单配置')

    args = parser.parse_args()

    # 加载配置
    import yaml

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"⚠️ 配置文件不存在: {config_path}")
        print("使用内置配置...")
        config = create_default_config(args)
    else:
        try:
            # 使用UTF-8编码读取
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except UnicodeDecodeError:
            print(f"⚠️ UTF-8解码失败，尝试GBK编码...")
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            config = create_default_config(args)

    if args.use_simple:
        print("使用简化配置...")
        config = create_simple_config(args)

    # 更新命令行参数
    config['data']['stage1_features_dir'] = args.features_dir
    config['data']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['optimizer']['lr'] = args.lr

    # 检查特征目录
    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        print(f"❌ 特征目录不存在: {features_dir}")
        print("请先运行特征提取脚本")
        return

    # 检查第一阶段检查点
    stage1_checkpoint = Path(args.stage1_checkpoint)
    if not stage1_checkpoint.exists():
        print(f"❌ 第一阶段检查点不存在: {stage1_checkpoint}")
        return

    print(f"\n🎯 最终版训练配置:")
    print(f"  配置文件: {args.config}")
    print(f"  特征目录: {args.features_dir}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮次: {args.epochs}")

    # 检查训练器类是否存在
    try:
        from training.train_stage2 import Stage2CompleteTrainer
        TrainerClass = Stage2CompleteTrainer
        print("✅ 使用Stage2CompleteTrainer")
    except ImportError:
        print("❌ 无法导入Stage2CompleteTrainer")
        return

    # 创建训练器
    trainer = TrainerClass(config, args.stage1_checkpoint)

    # 开始训练
    trainer.train(config['training']['epochs'])


def create_default_config(args):
    """创建默认配置"""
    return {
        'data': {
            'stage1_features_dir': args.features_dir,
            'train_list_path': os.path.join(args.features_dir, "stage2_train_list.json"),
            'test_list_path': os.path.join(args.features_dir, "stage2_test_list.json"),
            'batch_size': args.batch_size,
            'num_workers': 2,
            'train_max_samples': -1,
            'test_max_samples': -1
        },
        'model': {
            'stage2': {
                'num_classes': 3,
                'node_feature_dim': 128,
                'gnn_hidden_dim': 256,
                'risk_hidden_dim': 64,
                'confidence_threshold': 0.1,
                'gnn_max_nodes': 50,
                'gnn_k_neighbors': 10,
                'gnn_distance_threshold': 0.2
            }
        },
        'training': {
            'epochs': args.epochs,
            'checkpoint': {
                'save_dir': "./checkpoints/stage2_final"
            },
            'optimizer': {
                'type': "adamw",
                'lr': args.lr,
                'weight_decay': 0.001,
                'beta1': 0.9,
                'beta2': 0.999
            },
            'scheduler': {
                'type': "cosine",
                'T_max': args.epochs,
                'eta_min': 1e-6
            },
            'loss': {
                'risk_loss': "mse",
                'risk_weight': 1.0
            }
        },
        'logging': {
            'tensorboard': False,
            'log_dir': "./logs/stage2_final",
            'print_interval': 10
        }
    }


def create_simple_config(args):
    """创建简化配置"""
    return {
        'data': {
            'stage1_features_dir': args.features_dir,
            'train_list_path': os.path.join(args.features_dir, "stage2_train_list.json"),
            'batch_size': args.batch_size,
            'num_workers': 2
        },
        'model': {
            'stage2': {
                'num_classes': 3,
                'gnn_hidden_dim': 256
            }
        },
        'training': {
            'epochs': args.epochs,
            'checkpoint': {
                'save_dir': "./checkpoints/stage2_simple"
            },
            'optimizer': {
                'type': "adamw",
                'lr': args.lr,
                'weight_decay': 0.0001
            },
            'loss': {
                'risk_loss': "mse",
                'risk_weight': 1.0
            }
        }
    }


if __name__ == '__main__':
    main()