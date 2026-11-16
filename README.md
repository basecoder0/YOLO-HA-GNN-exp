TODO:  Update code for scripts to run all at once...maybe

# Data Interface

The default Stage 1 configuration uses LLVIP (`configs/train/stage1.yaml`):
- Dataset root: `data/datasets/LLVIP/`
- Expected folders: `visible/{train,test}`, `infrared/{train,test}`, and `Annotations/`
- Paired visible/infrared images use matching filenames; annotations are VOC-format XML.
- LLVIP is a single-class dataset (`person`). Update the Stage 1 class names/count and the Stage 2 feature/checkpoint paths together when changing datasets.

# Training Workflow

1. **Stage 1 training**: uses paired visible and infrared images to train the YOLO perception module.
2. **Feature extraction**: uses the Stage 1 checkpoint to create the RoI feature dataset.
3. **Stage 2 training**: trains the GNN using RoI features and dataset annotations.

# Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train Stage 1 on LLVIP (checkpoint: runs/stage1_llvip/best.pt)
python training/train_stage1.py

# Check Stage 1 results (check whether red and green boxes roughly overlap)
python verify_stage1_result.py

# Extract Stage 1 RoI features using configs/train/stage1.yaml
# Output: stage2_gnn_dataset_llvip_1280/
python extract_features_for_gnn.py

# Train Stage 2 on LLVIP (checkpoint: runs/stage2_gnn_llvip/epoch_50.pth)
python training/train_stage2.py

# Test Stage 2 results (results stored in comparison_vis/ours)
python visualize_stage2_result.py

# The comparison and risk-assessment scripts below still use R-LiViT-specific
# paths/classes; they are not part of the LLVIP workflow above.
# (Object detection only) Visualize HA-GNN vs Baseline test (results stored in "D:\YOLO-GNN-Fusion\comparison_vis\side_by_side")
python visualize_comparison_side_by_side.py

# (Object detection only) Test HA-GNN vs Baseline, with false-positive/missed-detection display (results stored in "D:\YOLO-GNN-Fusion\comparison_vis\side_by_side_success_rate")
python visualize_comparison_side_by_side_with_metrics.py

# Run the LLVIP ablation experiments (checkpoints saved under runs/ablation_llvip_*/)
python scripts/train_ablation.py --task rgb      # Baseline (YOLOv8, visible only) → runs/ablation_llvip_rgb/best.pt
python scripts/train_ablation.py --task thermal  # + Infrared (simple add fusion) → runs/ablation_llvip_thermal/best.pt
python scripts/train_ablation.py --task dpafm    # + DPAFM (attention fusion)     → runs/ablation_llvip_dpafm/best.pt

# (Object detection only) Metrics; comparison experiment
python scripts/evaluate_comparison.py

# (Object detection only) Metrics; ablation experiment
python scripts/evaluate_ablation.py

# Run the demo (results stored in "D:\YOLO-GNN-Fusion\comparison_vis\risk_assessment")
python inference_final.py
```

# Technical Innovations

1. ✅ Feature-level adaptive fusion (spatial + channel adaptive)
2. ✅ Multi-scale RoI Align feature extraction
3. ✅ Low-confidence candidate generation
4. ✅ Condensed node feature construction
5. ✅ Hierarchical attention GNN mechanism
6. ✅ Occlusion-aware message passing
7. ✅ BEV spatial cross-modal association
8. ✅ Structured risk assessment formula
9. ✅ Differentiated conflict-pair weighting strategy

The project fully implements all the technical innovations described in this document!
