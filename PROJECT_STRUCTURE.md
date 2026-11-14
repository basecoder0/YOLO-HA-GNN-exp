# 项目结构说明

## 目录结构

```
YOLO-GNN-Fusion/
├── configs/                 # 配置文件
│   ├── model/              # 模型配置
│   │   └── yolo_gnn_fusion.yaml
│   ├── train/              # 训练配置
│   │   ├── stage1.yaml
│   │   └── stage2极.yaml
│   └── test/               # 测试配置
│       └── evaluation.yaml
├── data/                   # 数据模块
│   ├── datasets/           # 数据集
│   │   ├── __init__.py
│   │   └── r_livit_dataset.py
│   ├── transforms/         # 数据变换
│   │   ├── __init__.py
│   │   └── augmentation.py
│   └── dataloaders/        # 数据加载器
│       ├── __init__.py
│       └── collate_fn.py
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── backbone/           # 骨干网络
│   │   ├── __init__.py
│   │   └── dual_resnet.py
│   ├── fusion/            # 融合模块
│   │   ├── __init__.py
│   │   └── adaptive_fusion.py
│   ├── yolo/              # YOLO模块
│   │   ├── __init__.py
│   │   ├── yolo_head.py
│   │   └── yolo_with_fusion.py
│   ├── gnn/               # GNN模块
│   │   ├── __init__.py
│   │   └── hierarchical_gnn.py
│   ├── risk_assessment/   # 风险评估
│   │   ├── __init__.py
│   │   └── risk_calculator.py
│   └── full_pipeline.py   # 完整流水线
├── training/              # 训练脚本
│   ├── __init__.py
│   ├── trainer.py
│   ├── train_stage1.py
│   └── train_stage2.py
├── testing/               # 测试评估
│   ├── __init__.py
│   ├── evaluator.py
│   └── test.py
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── visualization.py   # 可视化
│   ├── losses/            # 损失函数
│   │   ├── __init__.py
│   │   └── yolo_loss.py
│   └── metrics/           # 评估指标
│       ├── __init__.py
│       └── evaluation_metrics.py
├── scripts/               # 运行脚本
│   ├── __init__.py
│   ├── run_training.py
│   └── demo.py
├── main.py               # 主程序入口
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 三阶段架构

### 第一阶段：YOLO感知模块
- **输入**: 可见光图像 + 红外图像
- **输出**: 初步检测结果 + 多尺度特征
- **关键技术**: 
  - 双分支特征提取 (DualResNetBackbone)
  - 自适应特征融合 (AdaptiveFusionModule)
  - 多尺度RoI Align特征提取
  - 低置信度阈值(0.1)候选生成

### 第二阶段：GNN推理模块
- **输入**: 第一阶段输出 + 场景上下文
- **输出**: 优化检测 + 风险评分
- **关键技术**:
  - 信息浓缩节点特征构建
  - 分层注意力机制 (外观/空间/运动)
  - 遮挡感知消息传递
  - 多任务输出头

### 第三阶段：风险评估模块
- **输入**: GNN输出 + LiDAR运动信息
- **输出**: 最终风险系数
- **关键技术**:
  - BEV空间跨模态关联
  - 结构化风险评估公式
  - 差异化冲突对权重

## 数据接口

使用R-LiViT数据集结构：
- 输入路径: `D:/BaiduNetdiskDownload/16356714/R-LiViT`
- 支持格式: .bin (LiDAR), .png (可见光/红外), .xml (标注)

## 训练流程

1. **第一阶段训练**: 仅使用RGB+红外，训练YOLO感知模块
2. **第二阶段训练**: 使用所有模态，训练GNN和风险评估模块

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 单独训练第一阶段(checkpoints保存在文件夹：runs\stage1_rlivit_highres_8cls，最佳为best.pt)
python training/train_stage1.py

#检查第一阶段成果(观察红绿框是否基本重合）
python verify_stage1_result.py

#提取第一阶段roi特征(提取到文件夹："D:\YOLO-GNN-Fusion\stage2_gnn_dataset_GAP_1280")
#应输入第一阶段最佳模型路径：ckpt_path = "D:\YOLO-GNN-Fusion\runs\stage1_rlivit_highres_8cls\best.pt"
python extract_features_GAP.py

# 单独训练第二阶段(checkpoints保存在文件夹：runs\stage2_gnn_GAP_1280,最佳为epoch_50.pth")
python training/train_stage2.py

# 测试第二阶段成果(结果储存在"D:\YOLO-GNN-Fusion\comparison_vis\ours"）
python visualize_stage2_result.py

#（只目标检测）可视化测试HA-GNN vs Baseline（结果储存在"D:\YOLO-GNN-Fusion\comparison_vis\side_by_side"）
python visulaize_comparison_side_by_side.py

#(只目标检测）测试HA-GNN vs Baseline，带误检漏检显示（结果储存在"D:\YOLO-GNN-Fusion\comparison_vis\side_by_side_success_rate"）
python visualize_comparison_side_by_side_with_metrics.py

#(只目标检测）指标；对比实验
python scripts/evaluate_comparison.py

#(只目标检测）指标；消融实验
python scripts/evaluate_ablation.py

#运行演示（结果储存在"D:\YOLO-GNN-Fusion\comparison_vis\risk_assessment"）
python inference_final.py
```

## 技术创新点

1. ✅ 特征级自适应融合 (空间+通道自适应)
2. ✅ 多尺度RoI Align特征提取
3. ✅ 低置信度候选目标生成
4. ✅ 信息浓缩节点特征构建
5. ✅ 分层注意力GNN机制
6. ✅ 遮挡感知消息传递
7. ✅ BEV空间跨模态关联
8. ✅ 结构化风险评估公式
9. ✅ 差异化冲突对权重策略

项目已完整实现文档中描述的所有技术创新点！
