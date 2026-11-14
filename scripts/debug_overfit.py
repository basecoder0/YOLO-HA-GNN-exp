import torch
import torch.optim as optim
import yaml
import os
import sys
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加路径确保能导入
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn
from utils.losses.yolo_loss import CustomYOLOLoss


def run_overfit():
    print("============================================================")
    print("🚀 开始极速过拟合测试 (Overfit Test)")
    print("   目标: 强迫模型死记硬背 4 张图片，验证学习能力。")
    print("============================================================")

    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")

    # 2. 加载配置
    with open('configs/train/stage1.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 3. 准备极小数据集 (只取前 4 张图)
    print("dataset initializing...")
    full_dataset = RLiViTDataset(cfg['data'], mode='train')

    # 强制截断，只留 4 个样本
    # 注意：我们要确保这 4 个样本里真的有目标，否则 loss 也没法降
    full_dataset.samples = full_dataset.samples[:4]
    print(f"✅ 数据集已截断，仅包含 {len(full_dataset)} 个样本")

    loader = DataLoader(
        full_dataset,
        batch_size=4,  # 一次把所有数据喂进去
        shuffle=False,  # 不打乱，方便它背诵
        collate_fn=fusion_collate_fn
    )

    # 4. 初始化模型
    model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])
    model.to(device)
    model.train()

    # 5. 优化器 (使用稍大的学习率加速收敛)
    # 之前是 0.01，我们这里用 0.02 试试，去掉 weight_decay 防止它“学不会”
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    # 6. 损失函数
    criterion = CustomYOLOLoss(model,
                               hyp=cfg['hyperparameters'],
                               alpha=cfg['hyperparameters']['fl_alpha'],
                               gamma=cfg['hyperparameters']['fl_gamma'])

    # 7. 疯狂循环 100 次 (Overfit Loop)
    losses = []
    box_losses = []
    cls_losses = []

    print("\nStarting Loop (100 Epochs)...")
    start_time = time.time()

    for epoch in range(100):
        total_loss = 0

        for batch in loader:
            # 数据上 GPU
            inputs = {'rgb': batch['rgb'].to(device), 'thermal': batch['thermal'].to(device)}
            targets = {
                'batch_idx': batch['batch_idx'].to(device),
                'cls': batch['cls'].to(device),
                'bboxes': batch['bboxes'].to(device)
            }

            optimizer.zero_grad()
            preds = model(inputs)
            loss, loss_items = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 记录分项损失 (Box, Cls)
            # loss_items is detached tensor
            box_losses.append(loss_items[0].item())
            cls_losses.append(loss_items[1].item())

        losses.append(total_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/100 | Total Loss: {total_loss:.4f} | Box: {loss_items[0]:.4f} | Cls: {loss_items[1]:.4f}")

    print(f"\n✅ 测试结束! 耗时: {time.time() - start_time:.2f}s")

    # 8. 结果分析与可视化
    initial_loss = losses[0]
    final_loss = losses[-1]

    print("------------------------------------------------------------")
    print(f"初始 Loss: {initial_loss:.4f}")
    print(f"最终 Loss: {final_loss:.4f}")

    if final_loss < initial_loss * 0.1:
        print("🎉🎉 结论: 通过测试！模型具备学习能力。")
        print("    -> 之前的问题应该是：训练轮数太少 或 学习率不合适。")
    elif final_loss < initial_loss * 0.5:
        print("⚠️ 结论: 勉强通过。Loss 下降了但不够快。")
        print("    -> 可能需要调整超参数 (lr, momentum)。")
    else:
        print("❌ 结论: 测试失败！模型完全学不进去。")
        print("    -> 极有可能是：梯度断裂 (Fusion模块代码问题) 或 标签分配失败。")

    # 保存 Loss 曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Total Loss')
    plt.title('Overfit Test Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('runs/overfit_curve.png')
    print("📊 Loss 曲线已保存至 runs/overfit_curve.png")

    # 9. 最终置信度检查 (看看是不是还是 0.01)
    print("\n🔍 检查最终置信度...")
    with torch.no_grad():
        model.eval()
        inputs = {'rgb': batch['rgb'].to(device), 'thermal': batch['thermal'].to(device)}
        preds = model(inputs)
        if isinstance(preds, (list, tuple)): preds = preds[0]

        # 查看 Class Score
        # preds: [B, 4+nc, Anchors] -> [:, 4:, :]
        cls_scores = preds[:, 4:, :]
        max_conf = cls_scores.max().item()
        print(f"   最终 Max Conf: {max_conf:.4f}")

        if max_conf > 0.5:
            print("✅ 置信度正常！模型确实记住了目标。")
        else:
            print("❌ 置信度依然很低。Loss 降了可能是因为 Box 回归在起作用，但分类没学会。")


if __name__ == '__main__':
    run_overfit()