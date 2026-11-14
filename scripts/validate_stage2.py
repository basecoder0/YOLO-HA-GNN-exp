# scripts/validate_stage2.py
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """加载检查点"""
    print(f"📂 加载检查点: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"✅ 检查点加载成功")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  验证损失: {checkpoint.get('val_loss', 'unknown'):.4f}")

    if 'val_metrics' in checkpoint:
        print(f"  验证指标:")
        for key, value in checkpoint['val_metrics'].items():
            print(f"    {key}: {value:.4f}")

    return checkpoint


def load_model(config: Dict, checkpoint_path: str, device: str = 'cuda'):
    """加载模型"""
    print("\n🔧 加载模型...")

    try:
        from models.full_pipeline import YOLOGNNFusion
    except ImportError as e:
        print(f"❌ 无法导入模型: {e}")
        return None

    model_config = config.get('model', {}).get('stage2', {})

    # 创建模型
    model = YOLOGNNFusion(
        num_classes=model_config.get('num_classes', 3),
        node_feature_dim=model_config.get('node_feature_dim', 128),
        gnn_hidden_dim=model_config.get('gnn_hidden_dim', 256),
        use_risk_assessment=True,
        use_lidar=False,
        confidence_threshold=model_config.get('confidence_threshold', 0.1),
        gnn_max_nodes=model_config.get('gnn_max_nodes', 50)
    ).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 模型权重加载成功")
    else:
        print(f"⚠️ 检查点中没有model_state_dict")

    model.eval()
    return model


class Stage2Validator:
    """第二阶段验证器"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def validate_sample(self, features_path: str) -> Dict:
        """验证单个样本"""
        try:
            # 加载特征
            features = torch.load(features_path, weights_only=False, map_location='cpu')

            # 准备输入
            gnn_features = features.get('gnn_features', torch.zeros(50, 128))
            risk_value = features.get('risk_value', 0.0)

            # 转换为batch格式
            gnn_input = gnn_features.unsqueeze(0).to(self.device)  # [1, 50, 128]

            # 前向传播
            with torch.no_grad():
                # 池化得到场景特征
                scene_features = gnn_input.mean(dim=1)  # [1, 128]

                # 使用风险头
                risk_head = self.model.multi_task_heads.risk_score_head
                pred_risk = risk_head(scene_features).squeeze().cpu().item()

            return {
                'sample_id': features.get('sample_id', 'unknown'),
                'pred_risk': float(pred_risk),
                'true_risk': float(risk_value),
                'error': abs(pred_risk - risk_value),
                'relative_error': abs(pred_risk - risk_value) / max(risk_value, 0.001)
            }

        except Exception as e:
            print(f"❌ 验证样本失败 {features_path}: {e}")
            return None

    def validate_dataset(self, features_dir: str, max_samples: int = 50) -> Dict:
        """验证整个数据集"""
        print(f"\n🔍 验证数据集: {features_dir}")

        features_dir = Path(features_dir)
        feature_files = list(features_dir.glob("*_features.pt"))

        if not feature_files:
            print("❌ 没有找到特征文件")
            return {}

        print(f"  找到 {len(feature_files)} 个特征文件")

        # 限制验证样本数量
        if max_samples > 0:
            feature_files = feature_files[:max_samples]

        results = []
        failed_count = 0

        for i, feature_file in enumerate(feature_files):
            print(f"  处理 {i + 1}/{len(feature_files)}: {feature_file.name}", end='\r')

            result = self.validate_sample(str(feature_file))
            if result:
                results.append(result)
            else:
                failed_count += 1

        print(f"\n✅ 验证完成: {len(results)} 成功, {failed_count} 失败")

        if not results:
            return {}

        # 计算统计指标
        pred_array = np.array([r['pred_risk'] for r in results])
        true_array = np.array([r['true_risk'] for r in results])

        metrics = {
            'num_samples': len(results),
            'mse': float(mean_squared_error(true_array, pred_array)),
            'rmse': float(np.sqrt(mean_squared_error(true_array, pred_array))),
            'mae': float(mean_absolute_error(true_array, pred_array)),
            'r2': float(r2_score(true_array, pred_array)),
            'mean_pred': float(np.mean(pred_array)),
            'mean_true': float(np.mean(true_array)),
            'std_pred': float(np.std(pred_array)),
            'std_true': float(np.std(true_array)),
            'min_pred': float(np.min(pred_array)),
            'max_pred': float(np.max(pred_array)),
            'min_true': float(np.min(true_array)),
            'max_true': float(np.max(true_array)),
            'mean_error': float(np.mean([r['error'] for r in results])),
            'mean_relative_error': float(np.mean([r['relative_error'] for r in results])),
            'median_error': float(np.median([r['error'] for r in results]))
        }

        return {
            'results': results,
            'metrics': metrics,
            'predictions': pred_array.tolist(),
            'ground_truth': true_array.tolist()
        }


def print_validation_report(validation_results: Dict):
    """打印验证报告"""
    if not validation_results or 'metrics' not in validation_results:
        print("❌ 没有验证结果")
        return

    metrics = validation_results['metrics']
    results = validation_results.get('results', [])

    print("\n" + "=" * 60)
    print("📊 第二阶段训练验证报告")
    print("=" * 60)

    print(f"\n📈 基本统计:")
    print(f"  样本数量: {metrics['num_samples']}")
    print(f"  预测均值: {metrics['mean_pred']:.4f}")
    print(f"  真实均值: {metrics['mean_true']:.4f}")
    print(f"  预测范围: [{metrics['min_pred']:.4f}, {metrics['max_pred']:.4f}]")
    print(f"  真实范围: [{metrics['min_true']:.4f}, {metrics['max_true']:.4f}]")

    print(f"\n📊 误差指标:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  平均绝对误差: {metrics['mean_error']:.4f}")
    print(f"  平均相对误差: {metrics['mean_relative_error']:.4f}")
    print(f"  中位数误差: {metrics['median_error']:.4f}")

    # R²解释
    r2 = metrics['r2']
    if r2 > 0.7:
        print(f"  ✅ R² = {r2:.4f} (优秀，模型解释力强)")
    elif r2 > 0.5:
        print(f"  ✅ R² = {r2:.4f} (良好，模型有一定解释力)")
    elif r2 > 0.3:
        print(f"  ⚠️ R² = {r2:.4f} (一般，模型解释力有限)")
    elif r2 > 0:
        print(f"  ⚠️ R² = {r2:.4f} (较差，但比平均值好)")
    else:
        print(f"  ❌ R² = {r2:.4f} (负值，模型预测不如简单平均值)")

    # 显示前几个样本的预测结果
    if results:
        print(f"\n🔍 前5个样本详细结果:")
        for i, result in enumerate(results[:5]):
            print(f"  样本{i + 1}: {result['sample_id']}")
            print(f"    预测: {result['pred_risk']:.4f}")
            print(f"    真实: {result['true_risk']:.4f}")
            print(f"    误差: {result['error']:.4f}")
            print(f"    相对误差: {result['relative_error']:.4f}")
            print()


def plot_validation_results(validation_results: Dict, save_dir: str = "./validation_plots"):
    """绘制验证结果图表"""
    if not validation_results or 'results' not in validation_results:
        print("❌ 没有验证结果可绘制")
        return

    os.makedirs(save_dir, exist_ok=True)

    results = validation_results['results']
    pred_array = np.array([r['pred_risk'] for r in results])
    true_array = np.array([r['true_risk'] for r in results])

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('第二阶段训练验证结果', fontsize=16)

    # 1. 预测 vs 真实值散点图
    ax = axes[0, 0]
    ax.scatter(true_array, pred_array, alpha=0.6)
    ax.plot([0, 1], [0, 1], 'r--', label='理想线')  # 对角线
    ax.set_xlabel('真实风险值')
    ax.set_ylabel('预测风险值')
    ax.set_title('预测 vs 真实值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 误差分布直方图
    ax = axes[0, 1]
    errors = [r['error'] for r in results]
    ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('绝对误差')
    ax.set_ylabel('频数')
    ax.set_title('误差分布')
    ax.axvline(np.mean(errors), color='r', linestyle='--', label=f'均值: {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='g', linestyle='--', label=f'中位数: {np.median(errors):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 残差图
    ax = axes[0, 2]
    residuals = pred_array - true_array
    ax.scatter(true_array, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('真实风险值')
    ax.set_ylabel('残差 (预测-真实)')
    ax.set_title('残差图')
    ax.grid(True, alpha=0.3)

    # 4. 预测值分布
    ax = axes[1, 0]
    ax.hist(pred_array, bins=20, alpha=0.7, label='预测', edgecolor='black')
    ax.hist(true_array, bins=20, alpha=0.7, label='真实', edgecolor='black')
    ax.set_xlabel('风险值')
    ax.set_ylabel('频数')
    ax.set_title('预测值与真实值分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 累积分布函数
    ax = axes[1, 1]
    sorted_pred = np.sort(pred_array)
    sorted_true = np.sort(true_array)
    y_pred = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred)
    y_true = np.arange(1, len(sorted_true) + 1) / len(sorted_true)
    ax.plot(sorted_pred, y_pred, label='预测CDF')
    ax.plot(sorted_true, y_true, label='真实CDF')
    ax.set_xlabel('风险值')
    ax.set_ylabel('累积概率')
    ax.set_title('累积分布函数(CDF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 误差 vs 风险值
    ax = axes[1, 2]
    ax.scatter(true_array, [r['relative_error'] for r in results], alpha=0.6)
    ax.set_xlabel('真实风险值')
    ax.set_ylabel('相对误差')
    ax.set_title('相对误差 vs 真实风险值')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(save_dir, 'stage2_validation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 验证图表已保存到: {plot_path}")

    # 创建箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [pred_array, true_array]
    ax.boxplot(data_to_plot, labels=['预测值', '真实值'])
    ax.set_ylabel('风险值')
    ax.set_title('预测值与真实值箱线图对比')
    ax.grid(True, alpha=0.3)

    boxplot_path = os.path.join(save_dir, 'stage2_boxplot.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📦 箱线图已保存到: {boxplot_path}")


def compare_checkpoints(checkpoints_dir: str, features_dir: str, device: str = 'cuda'):
    """比较多个检查点"""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_files = list(checkpoints_dir.glob("*.pth"))

    if not checkpoint_files:
        print(f"❌ 没有找到检查点文件: {checkpoints_dir}")
        return

    print(f"\n🔍 比较 {len(checkpoint_files)} 个检查点")

    comparison_results = []

    for checkpoint_file in checkpoint_files:
        print(f"\n📊 评估检查点: {checkpoint_file.name}")

        # 加载检查点
        checkpoint = load_checkpoint(str(checkpoint_file), device)
        if not checkpoint:
            continue

        # 创建配置
        config = checkpoint.get('config', {})
        if not config:
            print(f"⚠️ 检查点中没有配置信息")
            continue

        # 加载模型
        model = load_model(config, str(checkpoint_file), device)
        if not model:
            continue

        # 验证
        validator = Stage2Validator(model, device)
        results = validator.validate_dataset(features_dir, max_samples=30)

        if results:
            metrics = results['metrics']
            comparison_results.append({
                'checkpoint': checkpoint_file.name,
                'epoch': checkpoint.get('epoch', 0),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mean_error': metrics['mean_error']
            })

    if not comparison_results:
        print("❌ 没有有效的比较结果")
        return

    # 创建对比表格
    df = pd.DataFrame(comparison_results)
    df = df.sort_values('val_loss')

    print("\n" + "=" * 60)
    print("📊 检查点对比结果")
    print("=" * 60)
    print(df.to_string(index=False))

    # 保存对比结果
    csv_path = os.path.join(checkpoints_dir.parent, 'checkpoint_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 对比结果保存到: {csv_path}")

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 验证损失对比
    axes[0, 0].plot(df['epoch'], df['val_loss'], 'bo-')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('验证损失')
    axes[0, 0].set_title('验证损失随epoch变化')
    axes[0, 0].grid(True, alpha=0.3)

    # R²对比
    axes[0, 1].plot(df['epoch'], df['r2'], 'go-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('R²随epoch变化')
    axes[0, 1].grid(True, alpha=0.3)

    # MAE对比
    axes[1, 0].plot(df['epoch'], df['mae'], 'ro-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE随epoch变化')
    axes[1, 0].grid(True, alpha=0.3)

    # 综合对比
    axes[1, 1].plot(df['epoch'], df['val_loss'] / df['val_loss'].max(), 'b-', label='损失(归一化)')
    axes[1, 1].plot(df['epoch'], df['r2'], 'g-', label='R²')
    axes[1, 1].plot(df['epoch'], df['mae'] / df['mae'].max(), 'r-', label='MAE(归一化)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('指标值')
    axes[1, 1].set_title('综合指标对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(checkpoints_dir.parent, 'checkpoint_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 对比图表保存到: {plot_path}")


def analyze_training_history(checkpoint_path: str):
    """分析训练历史"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'train_losses' not in checkpoint:
        print("❌ 检查点中没有训练历史数据")
        return

    train_losses = checkpoint['train_losses']

    print(f"\n📊 训练历史分析:")
    print(f"  总训练轮次: {len(train_losses)}")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  最佳训练损失: {min(train_losses):.4f}")

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('训练损失')
    plt.title('训练损失曲线')
    plt.grid(True, alpha=0.3)

    # 添加平滑线
    if len(train_losses) > 5:
        smooth_losses = pd.Series(train_losses).rolling(window=5, min_periods=1).mean()
        plt.plot(smooth_losses, 'r--', linewidth=1.5, label='平滑线')
        plt.legend()

    save_dir = Path(checkpoint_path).parent
    plot_path = os.path.join(save_dir, 'training_history_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 训练历史图保存到: {plot_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='第二阶段训练验证脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--features_dir', type=str, default='./stage1_features_diverse',
                        help='特征目录')
    parser.add_argument('--config', type=str,
                        help='配置文件路径 (可选，检查点中有配置)')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='最大验证样本数')
    parser.add_argument('--compare', action='store_true',
                        help='比较目录中所有检查点')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    parser.add_argument('--analyze_history', action='store_true',
                        help='分析训练历史')

    args = parser.parse_args()

    print("=" * 60)
    print("🤖 第二阶段训练验证系统")
    print("=" * 60)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 使用设备: {device}")

    if args.compare:
        # 比较模式
        compare_checkpoints(args.checkpoint, args.features_dir, device)
        return

    if args.analyze_history:
        # 分析训练历史
        analyze_training_history(args.checkpoint)
        return

    # 单检查点验证
    # 1. 加载检查点
    checkpoint = load_checkpoint(args.checkpoint, device)
    if not checkpoint:
        return

    # 2. 获取配置
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config', {})
        if not config:
            print("⚠️ 警告: 使用默认配置")
            config = {
                'model': {
                    'stage2': {
                        'num_classes': 3,
                        'node_feature_dim': 128,
                        'gnn_hidden_dim': 256
                    }
                }
            }

    # 3. 加载模型
    model = load_model(config, args.checkpoint, device)
    if not model:
        return

    # 4. 验证
    validator = Stage2Validator(model, device)
    validation_results = validator.validate_dataset(args.features_dir, args.max_samples)

    if not validation_results:
        print("❌ 验证失败")
        return

    # 5. 打印报告
    print_validation_report(validation_results)

    # 6. 生成图表
    if args.plot:
        save_dir = Path(args.checkpoint).parent
        plot_validation_results(validation_results, str(save_dir))

    # 7. 保存验证结果
    save_dir = Path(args.checkpoint).parent
    results_path = os.path.join(save_dir, 'validation_results.json')

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'validation_results': validation_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 验证结果保存到: {results_path}")


if __name__ == '__main__':
    main()