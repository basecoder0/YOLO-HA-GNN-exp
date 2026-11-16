# Project Structure Guide

## Directory Structure

```
YOLO-GNN-Fusion/
├── configs/                 # Configuration files
│   ├── model/              # Model configuration
│   │   └── yolo_gnn_fusion.yaml
│   ├── train/              # Training configuration
│   │   ├── stage1.yaml
│   │   └── stage2.yaml
│   └── test/               # Testing configuration
│       └── evaluation.yaml
├── data/                   # Data module
│   ├── datasets/           # Datasets
│   │   ├── __init__.py
│   │   └── r_livit_dataset.py
│   ├── transforms/         # Data transformation
│   │   ├── __init__.py
│   │   └── augmentation.py
│   └── dataloaders/        # Data loaders
│       ├── __init__.py
│       └── collate_fn.py
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── backbone/           # Backbone network
│   │   ├── __init__.py
│   │   └── dual_resnet.py
│   ├── fusion/            # Fusion module
│   │   ├── __init__.py
│   │   └── adaptive_fusion.py
│   ├── yolo/              # YOLO module
│   │   ├── __init__.py
│   │   ├── yolo_head.py
│   │   └── yolo_with_fusion.py
│   ├── gnn/               # GNN module
│   │   ├── __init__.py
│   │   └── hierarchical_gnn.py
│   ├── risk_assessment/   # Risk assessment
│   │   ├── __init__.py
│   │   └── risk_calculator.py
│   └── full_pipeline.py   # Complete pipeline
├── training/              # Training scripts
│   ├── __init__.py
│   ├── trainer.py
│   ├── train_stage1.py
│   └── train_stage2.py
├── testing/               # Testing and evaluation
│   ├── __init__.py
│   ├── evaluator.py
│   └── test.py
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── visualization.py   # Visualization
│   ├── losses/            # Loss functions
│   │   ├── __init__.py
│   │   └── yolo_loss.py
│   └── metrics/           # Evaluation metrics
│       ├── __init__.py
│       └── evaluation_metrics.py
├── scripts/               # Execution scripts
│   ├── __init__.py
│   ├── run_training.py
│   └── demo.py
├── main.py               # Main program entry
├── requirements.txt      # Dependencies
└── README.md            # Project documentation
```

## Three-Stage Architecture

### Stage 1: YOLO Perception Module
- **Input**: Visible light image + Infrared image
- **Output**: Initial detection results + Multi-scale features
- **Key Technologies**: 
  - Dual-branch feature extraction (DualResNetBackbone)
  - Adaptive feature fusion (AdaptiveFusionModule)
  - Multi-scale RoI Align feature extraction
  - Low confidence threshold (0.1) candidate generation

### Stage 2: GNN Inference Module
- **Input**: Stage 1 output + Scene context
- **Output**: Optimized detection + Risk score
- **Key Technologies**:
  - Information condensed node feature construction
  - Hierarchical attention mechanism (appearance/spatial/motion)
  - Occlusion-aware message passing
  - Multi-task output head

### Stage 3: Risk Assessment Module
- **Input**: GNN output + LiDAR motion information
- **Output**: Final risk coefficient
- **Key Technologies**:
  - BEV space cross-modal association
  - Structured risk assessment formula
  - Differentiated conflict pair weighting

## Data Interface

Using R-LiViT dataset structure:
- Input path: `D:/BaiduNetdiskDownload/16356714/R-LiViT`
- Supported formats: .bin (LiDAR), .png (visible light/infrared), .xml (annotation)

## Training Workflow

1. **Stage 1 Training**: Using only RGB+infrared, train YOLO perception module
2. **Stage 2 Training**: Using all modalities, train GNN and risk assessment module

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train stage 1 only (checkpoints saved in folder: runs\stage1_rlivit_highres_8cls, best is best.pt)
python training/train_stage1.py

# Verify stage 1 results (observe if red and green boxes roughly overlap)
python verify_stage1_result.py

# Extract stage 1 ROI features (extract to folder: "D:\YOLO-GNN-Fusion\stage2_gnn_dataset_GAP_1280")
# Input stage 1 best model path: ckpt_path = "D:\YOLO-GNN-Fusion\runs\stage1_rlivit_highres_8cls\best.pt"
python extract_features_GAP.py

# Train stage 2 only (checkpoints saved in folder: runs\stage2_gnn_GAP_1280, best is epoch_50.pth")
python training/train_stage2.py

# Test stage 2 results (results stored in "D:\YOLO-GNN-Fusion\comparison_vis\ours")
python visualize_stage2_result.py

# (Object detection only) Visualize HA-GNN vs Baseline (results stored in "D:\YOLO-GNN-Fusion\comparison_vis\side_by_side")
python visulaize_comparison_side_by_side.py

# (Object detection only) Test HA-GNN vs Baseline with false positive/false negative display (results stored in "D:\YOLO-GNN-Fusion\comparison_vis\side_by_side_success_rate")
python visualize_comparison_side_by_side_with_metrics.py

# (Object detection only) Metrics; comparative experiment
python scripts/evaluate_comparison.py

# (Object detection only) Metrics; ablation study
python scripts/evaluate_ablation.py

# Run demonstration (results stored in "D:\YOLO-GNN-Fusion\comparison_vis\risk_assessment")
python inference_final.py
```

## Technical Innovations

1. ✅ Feature-level adaptive fusion (spatial + channel adaptive)
2. ✅ Multi-scale RoI Align feature extraction
3. ✅ Low confidence candidate target generation
4. ✅ Information condensed node feature construction
5. ✅ Hierarchical attention GNN mechanism
6. ✅ Occlusion-aware message passing
7. ✅ BEV space cross-modal association
8. ✅ Structured risk assessment formula
9. ✅ Differentiated conflict pair weighting strategy

All technical innovations described in the documentation have been fully implemented!

