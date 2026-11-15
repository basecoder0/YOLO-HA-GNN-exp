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
    """loadingcheckpoint"""
    print(f"📂 loadingcheckpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ checkpointnotexist: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"✅ checkpointloadingsuccess")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  validationloss: {checkpoint.get('val_loss', 'unknown'):.4f}")

    if 'val_metrics' in checkpoint:
        print(f"  validationmetric:")
        for key, value in checkpoint['val_metrics'].items():
            print(f"    {key}: {value:.4f}")

    return checkpoint


def load_model(config: Dict, checkpoint_path: str, device: str = 'cuda'):
    """loadingmodel"""
    print("\n🔧 loadingmodel...")

    try:
        from models.full_pipeline import YOLOGNNFusion
    except ImportError as e:
        print(f"❌ unable toimportmodel: {e}")
        return None

    model_config = config.get('model', {}).get('stage2', {})

    # createmodel
    model = YOLOGNNFusion(
        num_classes=model_config.get('num_classes', 3),
        node_feature_dim=model_config.get('node_feature_dim', 128),
        gnn_hidden_dim=model_config.get('gnn_hidden_dim', 256),
        use_risk_assessment=True,
        use_lidar=False,
        confidence_threshold=model_config.get('confidence_threshold', 0.1),
        gnn_max_nodes=model_config.get('gnn_max_nodes', 50)
    ).to(device)

    # loadweight
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ modelweightloadingsuccess")
    else:
        print(f"⚠️ checkpointinnothavemodel_state_dict")

    model.eval()
    return model


class Stage2Validator:
    """secondstagevalidationdevice"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def validate_sample(self, features_path: str) -> Dict:
        """validationsingleasample"""
        try:
            # loadfeature
            features = torch.load(features_path, weights_only=False, map_location='cpu')

            # prepareinput
            gnn_features = features.get('gnn_features', torch.zeros(50, 128))
            risk_value = features.get('risk_value', 0.0)

            # convertfor/asbatchformat
            gnn_input = gnn_features.unsqueeze(0).to(self.device)  # [1, 50, 128]

            # forward propagation
            with torch.no_grad():
                # pooltransformgettoscenefeature
                scene_features = gnn_input.mean(dim=1)  # [1, 128]

                # usingriskhead
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
            print(f"❌ validationsamplefailed {features_path}: {e}")
            return None

    def validate_dataset(self, features_dir: str, max_samples: int = 50) -> Dict:
        """validationintegeradataset"""
        print(f"\n🔍 verify dataset: {features_dir}")

        features_dir = Path(features_dir)
        feature_files = list(features_dir.glob("*_features.pt"))

        if not feature_files:
            print("❌ nothavefindtofeaturefile")
            return {}

        print(f"  findto {len(feature_files)} afeaturefile")

        # limitverifysamplecountamount
        if max_samples > 0:
            feature_files = feature_files[:max_samples]

        results = []
        failed_count = 0

        for i, feature_file in enumerate(feature_files):
            print(f"  processing {i + 1}/{len(feature_files)}: {feature_file.name}", end='\r')

            result = self.validate_sample(str(feature_file))
            if result:
                results.append(result)
            else:
                failed_count += 1

        print(f"\n✅ validationcompleted: {len(results)} success, {failed_count} failed")

        if not results:
            return {}

        # calculationstatisticsmetric
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
    """printvalidation报notification"""
    if not validation_results or 'metrics' not in validation_results:
        print("❌ nothavevalidationresult")
        return

    metrics = validation_results['metrics']
    results = validation_results.get('results', [])

    print("\n" + "=" * 60)
    print("📊 secondstagetrainingvalidation报notification")
    print("=" * 60)

    print(f"\n📈 basebookstatistics:")
    print(f"  samplecountamount: {metrics['num_samples']}")
    print(f"  predictionmean: {metrics['mean_pred']:.4f}")
    print(f"  truemean: {metrics['mean_true']:.4f}")
    print(f"  predictionrange: [{metrics['min_pred']:.4f}, {metrics['max_pred']:.4f}]")
    print(f"  truerange: [{metrics['min_true']:.4f}, {metrics['max_true']:.4f}]")

    print(f"\n📊 errormetric:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  averageabsoluteerror: {metrics['mean_error']:.4f}")
    print(f"  averagerelativeerror: {metrics['mean_relative_error']:.4f}")
    print(f"  medianerror: {metrics['median_error']:.4f}")

    # R²resolveexplain
    r2 = metrics['r2']
    if r2 > 0.7:
        print(f"  ✅ R² = {r2:.4f} (better秀，modelresolveexplainabilitystrong)")
    elif r2 > 0.5:
        print(f"  ✅ R² = {r2:.4f} (良good，modelhaveonedefineresolveexplainability)")
    elif r2 > 0.3:
        print(f"  ⚠️ R² = {r2:.4f} (one般，modelresolveexplainabilityhavelimit)")
    elif r2 > 0:
        print(f"  ⚠️ R² = {r2:.4f} (comparedifference，butcompareaveragevaluegood)")
    else:
        print(f"  ❌ R² = {r2:.4f} (negativevalue，modelpredictionnotlikesimpleaveragevalue)")

    # show predictions for the first few samplesresult
    if results:
        print(f"\n🔍 first5asampledetailedresult:")
        for i, result in enumerate(results[:5]):
            print(f"  sample{i + 1}: {result['sample_id']}")
            print(f"    prediction: {result['pred_risk']:.4f}")
            print(f"    true: {result['true_risk']:.4f}")
            print(f"    error: {result['error']:.4f}")
            print(f"    relativeerror: {result['relative_error']:.4f}")
            print()


def plot_validation_results(validation_results: Dict, save_dir: str = "./validation_plots"):
    """drawvalidationresultimagetable"""
    if not validation_results or 'results' not in validation_results:
        print("❌ nothavevalidationresultoptionaldraw")
        return

    os.makedirs(save_dir, exist_ok=True)

    results = validation_results['results']
    pred_array = np.array([r['pred_risk'] for r in results])
    true_array = np.array([r['true_risk'] for r in results])

    # createimageshape
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('secondstagetrainingvalidationresult', fontsize=16)

    # 1. prediction vs truevalue散pointimage
    ax = axes[0, 0]
    ax.scatter(true_array, pred_array, alpha=0.6)
    ax.plot([0, 1], [0, 1], 'r--', label='manage想line')  # pairangleline
    ax.set_xlabel('trueriskvalue')
    ax.set_ylabel('predictionriskvalue')
    ax.set_title('prediction vs truevalue')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. errordistribution直methodimage
    ax = axes[0, 1]
    errors = [r['error'] for r in results]
    ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('absoluteerror')
    ax.set_ylabel('frequencycount')
    ax.set_title('errordistribution')
    ax.axvline(np.mean(errors), color='r', linestyle='--', label=f'mean: {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='g', linestyle='--', label=f'median: {np.median(errors):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 残differenceimage
    ax = axes[0, 2]
    residuals = pred_array - true_array
    ax.scatter(true_array, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('trueriskvalue')
    ax.set_ylabel('残difference (prediction-true)')
    ax.set_title('残differenceimage')
    ax.grid(True, alpha=0.3)

    # 4. predictionvaluedistribution
    ax = axes[1, 0]
    ax.hist(pred_array, bins=20, alpha=0.7, label='prediction', edgecolor='black')
    ax.hist(true_array, bins=20, alpha=0.7, label='true', edgecolor='black')
    ax.set_xlabel('riskvalue')
    ax.set_ylabel('frequencycount')
    ax.set_title('predictionvalueandtruevaluedistribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 累积distributionfunction
    ax = axes[1, 1]
    sorted_pred = np.sort(pred_array)
    sorted_true = np.sort(true_array)
    y_pred = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred)
    y_true = np.arange(1, len(sorted_true) + 1) / len(sorted_true)
    ax.plot(sorted_pred, y_pred, label='predictionCDF')
    ax.plot(sorted_true, y_true, label='trueCDF')
    ax.set_xlabel('riskvalue')
    ax.set_ylabel('累积probability')
    ax.set_title('累积distributionfunction(CDF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. error vs riskvalue
    ax = axes[1, 2]
    ax.scatter(true_array, [r['relative_error'] for r in results], alpha=0.6)
    ax.set_xlabel('trueriskvalue')
    ax.set_ylabel('relativeerror')
    ax.set_title('relativeerror vs trueriskvalue')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # saveimage
    plot_path = os.path.join(save_dir, 'stage2_validation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 validationimagetablealreadysaveto: {plot_path}")

    # createbox plotimage
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [pred_array, true_array]
    ax.boxplot(data_to_plot, labels=['predictionvalue', 'truevalue'])
    ax.set_ylabel('riskvalue')
    ax.set_title('box plot of predicted values and ground truthimagecomparison')
    ax.grid(True, alpha=0.3)

    boxplot_path = os.path.join(save_dir, 'stage2_boxplot.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📦 box plotimagealreadysaveto: {boxplot_path}")


def compare_checkpoints(checkpoints_dir: str, features_dir: str, device: str = 'cuda'):
    """comparecomparemultipleacheckpoint"""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_files = list(checkpoints_dir.glob("*.pth"))

    if not checkpoint_files:
        print(f"❌ nothavefindtocheckpointfile: {checkpoints_dir}")
        return

    print(f"\n🔍 comparecompare {len(checkpoint_files)} acheckpoint")

    comparison_results = []

    for checkpoint_file in checkpoint_files:
        print(f"\n📊 evaluationcheckpoint: {checkpoint_file.name}")

        # loadcheckpoint
        checkpoint = load_checkpoint(str(checkpoint_file), device)
        if not checkpoint:
            continue

        # createconfiguration
        config = checkpoint.get('config', {})
        if not config:
            print(f"⚠️ checkpointinnothaveconfigurationinformation")
            continue

        # loadmodel
        model = load_model(config, str(checkpoint_file), device)
        if not model:
            continue

        # verify
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
        print("❌ nothavehaveeffectofcomparecompareresult")
        return

    # createcomparisontablestrict
    df = pd.DataFrame(comparison_results)
    df = df.sort_values('val_loss')

    print("\n" + "=" * 60)
    print("📊 checkpointcomparisonresult")
    print("=" * 60)
    print(df.to_string(index=False))

    # savecomparisonresult
    csv_path = os.path.join(checkpoints_dir.parent, 'checkpoint_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 comparisonresultsaveto: {csv_path}")

    # drawcomparisonimage
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # verifylosscomparison
    axes[0, 0].plot(df['epoch'], df['val_loss'], 'bo-')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('validationloss')
    axes[0, 0].set_title('validationlossfollowepochchangetransform')
    axes[0, 0].grid(True, alpha=0.3)

    # R²comparison
    axes[0, 1].plot(df['epoch'], df['r2'], 'go-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('R²followepochchangetransform')
    axes[0, 1].grid(True, alpha=0.3)

    # MAEcomparison
    axes[1, 0].plot(df['epoch'], df['mae'], 'ro-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAEfollowepochchangetransform')
    axes[1, 0].grid(True, alpha=0.3)

    # comprehensivecomparison
    axes[1, 1].plot(df['epoch'], df['val_loss'] / df['val_loss'].max(), 'b-', label='loss(normalize)')
    axes[1, 1].plot(df['epoch'], df['r2'], 'g-', label='R²')
    axes[1, 1].plot(df['epoch'], df['mae'] / df['mae'].max(), 'r-', label='MAE(normalize)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('metricvalue')
    axes[1, 1].set_title('comprehensivemetriccomparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(checkpoints_dir.parent, 'checkpoint_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 comparisonimagetablesaveto: {plot_path}")


def analyze_training_history(checkpoint_path: str):
    """analysistraininghistory"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'train_losses' not in checkpoint:
        print("❌ checkpointinnothavetraininghistorydata")
        return

    train_losses = checkpoint['train_losses']

    print(f"\n📊 traininghistoryanalysis:")
    print(f"  totaltrainingepoch: {len(train_losses)}")
    print(f"  finaltrainingloss: {train_losses[-1]:.4f}")
    print(f"  besttrainingloss: {min(train_losses):.4f}")

    # drawtraininglosscurveline
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('trainingloss')
    plt.title('traininglosscurveline')
    plt.grid(True, alpha=0.3)

    # addflat滑line
    if len(train_losses) > 5:
        smooth_losses = pd.Series(train_losses).rolling(window=5, min_periods=1).mean()
        plt.plot(smooth_losses, 'r--', linewidth=1.5, label='flat滑line')
        plt.legend()

    save_dir = Path(checkpoint_path).parent
    plot_path = os.path.join(save_dir, 'training_history_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 traininghistoryimagesaveto: {plot_path}")


def main():
    """majorfunction"""
    import argparse

    parser = argparse.ArgumentParser(description='secondstagetrainingvalidationscript')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpointpath')
    parser.add_argument('--features_dir', type=str, default='./stage1_features_diverse',
                        help='featuredirectory')
    parser.add_argument('--config', type=str,
                        help='configurationfilepath (optional，checkpointinhaveconfiguration)')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='maximumvalidationsamplecount')
    parser.add_argument('--compare', action='store_true',
                        help='comparecomparedirectoryinallcheckpoint')
    parser.add_argument('--plot', action='store_true',
                        help='generate visualization image table')
    parser.add_argument('--analyze_history', action='store_true',
                        help='analysistraininghistory')

    args = parser.parse_args()

    print("=" * 60)
    print("🤖 secondstagetrainingvalidationcoefficientcount")
    print("=" * 60)

    # setdevice
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 usingdevice: {device}")

    if args.compare:
        # comparecomparemode
        compare_checkpoints(args.checkpoint, args.features_dir, device)
        return

    if args.analyze_history:
        # analysistraininghistory
        analyze_training_history(args.checkpoint)
        return

    # singlecheckpointverify
    # 1. loadcheckpoint
    checkpoint = load_checkpoint(args.checkpoint, device)
    if not checkpoint:
        return

    # 2. getconfiguration
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config', {})
        if not config:
            print("⚠️ warning: usingdefaultconfiguration")
            config = {
                'model': {
                    'stage2': {
                        'num_classes': 3,
                        'node_feature_dim': 128,
                        'gnn_hidden_dim': 256
                    }
                }
            }

    # 3. loadmodel
    model = load_model(config, args.checkpoint, device)
    if not model:
        return

    # 4. verify
    validator = Stage2Validator(model, device)
    validation_results = validator.validate_dataset(args.features_dir, args.max_samples)

    if not validation_results:
        print("❌ validationfailed")
        return

    # 5. print报notification
    print_validation_report(validation_results)

    # 6. generateimagetable
    if args.plot:
        save_dir = Path(args.checkpoint).parent
        plot_validation_results(validation_results, str(save_dir))

    # 7. saveverifyresult
    save_dir = Path(args.checkpoint).parent
    results_path = os.path.join(save_dir, 'validation_results.json')

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'config': config,
            'validation_results': validation_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 validationresultsaveto: {results_path}")


if __name__ == '__main__':
    main()