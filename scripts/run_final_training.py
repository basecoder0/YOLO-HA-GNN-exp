# scripts/run_final_training_fixed.py
# !/usr/bin/env python3
"""
finalversionsecondstagetrainingscript - fixencodingproblem
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='finalversionsecondstagetraining')
    parser.add_argument('--config', type=str, default='configs/train/stage2.yaml')
    parser.add_argument('--features_dir', type=str, default='./stage1_features_diverse')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='D:/YOLO-GNN-Fusion/checkpoints_final/best_model_epoch_20_loss_0.2275.pth')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_simple', action='store_true', help='usingsimpleconfiguration')

    args = parser.parse_args()

    # loadconfiguration
    import yaml

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"⚠️ configurationfilenotexist: {config_path}")
        print("usinginsideplaceconfiguration...")
        config = create_default_config(args)
    else:
        try:
            # usingUTF-8encodingread
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except UnicodeDecodeError:
            print(f"⚠️ UTF-8decodingfailed，tryGBKencoding...")
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ loadingconfigurationfilefailed: {e}")
            config = create_default_config(args)

    if args.use_simple:
        print("usingsimplifyconfiguration...")
        config = create_simple_config(args)

    # updatecommandlineparameter
    config['data']['stage1_features_dir'] = args.features_dir
    config['data']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['optimizer']['lr'] = args.lr

    # checkfeaturedirectory
    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        print(f"❌ featuredirectorynotexist: {features_dir}")
        print("please firstrunfeatureextractscript")
        return

    # checkfirststagecheckpoint
    stage1_checkpoint = Path(args.stage1_checkpoint)
    if not stage1_checkpoint.exists():
        print(f"❌ firststagecheckpointnotexist: {stage1_checkpoint}")
        return

    print(f"\n🎯 finalversiontrainingconfiguration:")
    print(f"  configurationfile: {args.config}")
    print(f"  featuredirectory: {args.features_dir}")
    print(f"  batchminorlargesmall: {args.batch_size}")
    print(f"  learning rate: {args.lr}")
    print(f"  trainingepoch: {args.epochs}")

    # checktrainingdeviceclasswhetherexist
    try:
        from training.train_stage2 import Stage2CompleteTrainer
        TrainerClass = Stage2CompleteTrainer
        print("✅ usingStage2CompleteTrainer")
    except ImportError:
        print("❌ unable toimportStage2CompleteTrainer")
        return

    # createtrainingdevice
    trainer = TrainerClass(config, args.stage1_checkpoint)

    # starttraining
    trainer.train(config['training']['epochs'])


def create_default_config(args):
    """createdefaultconfiguration"""
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
    """createsimplifyconfiguration"""
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