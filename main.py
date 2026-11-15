#!/usr/bin/env python3
"""
performshowscript
show the complete three-stage pipeline effect
"""

import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
from PIL import Image

# addprojectrootdirectorytopath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import YOLOGNNFusion
from utils.visualization import Visualizer
from data.transforms import ValTransform


def parse_args():
    """parsecommandlineparameter"""
    parser = argparse.ArgumentParser(description='YOLO-GNN-Fusion Demo')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration file')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    parser.add_argument('--rgb_image', type=str, required=True,
                        help='Path to RGB image')

    parser.add_argument('--thermal_image', type=str, required=True,
                        help='Path to thermal image')

    parser.add_argument('--output_dir', type=str, default='./demo_output',
                        help='Output directory for results')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, device: str):
    """loadingmodel"""
    # loadconfiguration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # createmodel
    model = YOLOGNNFusion(
        num_classes=config['model']['stage1']['yolo']['num_classes'],
        use_risk_assessment=True
    ).to(device)

    # loadweight
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    return model, config


def preprocess_images(rgb_path: str, thermal_path: str, input_size: tuple):
    """predict/preprocessingimage"""
    transform = ValTransform(input_size=input_size)

    # loadRGBimage
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # loadthermalimagingimage
    thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
    thermal_img = np.stack([thermal_img] * 3, axis=-1)

    # application/applytransformation
    data = {
        'rgb': rgb_img,
        'thermal': thermal_img,
        'bboxes': np.zeros((0, 4)),  # emptyannotation
        'labels': np.zeros((0,)),
        'occlusion': np.zeros((0,))
    }

    transformed = transform(data)

    return transformed, rgb_img


def run_demo():
    """runperformshow"""
    args = parse_args()

    # createoutputdirectory
    os.makedirs(args.output_dir, exist_ok=True)

    # setdevice
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # loadmodel
    model, config = load_model(args.config, args.checkpoint, device)

    # predict/preprocessingimage
    input_size = config['data']['input_size']
    transformed_data, original_rgb = preprocess_images(
        args.rgb_image, args.thermal_image, input_size
    )

    # preparemodelinput
    rgb_tensor = transformed_data['rgb'].unsqueeze(0).to(device)
    thermal_tensor = transformed_data['thermal'].unsqueeze(0).to(device)

    # modelinference
    with torch.no_grad():
        print("Running model inference...")
        outputs = model(rgb_tensor, thermal_tensor)

    # visualizationresult
    visualizer = Visualizer()

    # firststageresult
    stage1_fig = visualizer.plot_detections(
        original_rgb,
        outputs['stage1_detections'],
        title="Stage 1: YOLO Detections",
        show_confidence=True,
        show_risk=False
    )
    stage1_fig.savefig(os.path.join(args.output_dir, 'stage1_detections.png'),
                       bbox_inches='tight', dpi=300)

    # secondstageresult
    stage2_fig = visualizer.plot_detections(
        original_rgb,
        outputs['enhanced_detections'],
        title="Stage 2: GNN Enhanced",
        show_confidence=True,
        show_risk=True
    )
    stage2_fig.savefig(os.path.join(args.output_dir, 'stage2_enhanced.png'),
                       bbox_inches='tight', dpi=300)

    # thethreestageresult
    if outputs['risk_assessment']:
        risk_fig = visualizer.plot_risk_assessment(
            original_rgb,
            outputs['risk_assessment'],
            title="Stage 3: Risk Assessment"
        )
        risk_fig.savefig(os.path.join(args.output_dir, 'stage3_risk.png'),
                         bbox_inches='tight', dpi=300)

    # comprehensivecomparisonimage
    comparison_fig = visualizer.create_comparison_plot(
        original_rgb,
        outputs['stage1_detections'],
        outputs['enhanced_detections'],
        outputs['risk_assessment']
    )
    comparison_fig.savefig(os.path.join(args.output_dir, 'comparison.png'),
                           bbox_inches='tight', dpi=300)

    # printresultabstractneed
    print("\n" + "=" * 50)
    print("DEMO RESULTS SUMMARY")
    print("=" * 50)

    print(f"Stage 1 - YOLO Detections: {len(outputs['stage1_detections'])} objects")

    enhanced_detections = outputs['enhanced_detections']
    print(f"Stage 2 - GNN Enhanced: {len(enhanced_detections)} objects")

    if enhanced_detections:
        avg_confidence = np.mean([d.get('confidence', 0) for d in enhanced_detections])
        avg_enhanced_confidence = np.mean([d.get('enhanced_confidence', 0) for d in enhanced_detections])
        avg_risk = np.mean([d.get('preliminary_risk', 0) for d in enhanced_detections])

        print(f"  Average confidence: {avg_confidence:.3f} -> {avg_enhanced_confidence:.3f}")
        print(f"  Average risk score: {avg_risk:.3f}")

    if outputs['risk_assessment']:
        risk_results = outputs['risk_assessment']
        print(f"Stage 3 - Risk Assessment:")
        print(f"  Overall risk: {risk_results.get('overall_risk', 0):.3f}")
        print(
            f"  High-risk targets: {len([t for t in risk_results.get('target_risks', []) if t['combined_risk'] > 0.6])}")
        print(f"  Conflict pairs: {len(risk_results.get('conflict_pairs', []))}")

        warnings = risk_results.get('warnings', [])
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for warning in warnings:
                print(f"    - {warning}")

    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    run_demo()

