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
