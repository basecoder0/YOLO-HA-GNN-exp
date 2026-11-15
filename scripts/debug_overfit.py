import torch
import torch.optim as optim
import yaml
import os
import sys
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# addpathmake sureability/canimport
sys.path.append(os.getcwd())

from models.yolo.yolo_with_fusion import YOLOWithFusion
from data.datasets.r_livit_dataset import RLiViTDataset
from data.dataloaders.collate_fn import fusion_collate_fn
from utils.losses.yolo_loss import CustomYOLOLoss


def run_overfit():
    print("============================================================")
    print("🚀 startextremespeedoverfittingtest (Overfit Test)")
    print("   target: strong迫model死record硬背 4 tensorimagepiece，validationstudylearnability/canability。")
    print("============================================================")

    # 1. setdevice
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ usingdevice: {device}")

    # 2. loadconfiguration
    with open('configs/train/stage1.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 3. prepareextremesmalldataset (onlygetfirst 4 tensorimage)
    print("dataset initializing...")
    full_dataset = RLiViTDataset(cfg['data'], mode='train')

    # strongcontrolcutbreak，onlykeep 4 asample
    # note：ourneedmake surethis 4 samples that actually contain targets，else loss alsonotmethodreduce
    full_dataset.samples = full_dataset.samples[:4]
    print(f"✅ datasetalreadycutbreak，onlycontains {len(full_dataset)} asample")

    loader = DataLoader(
        full_dataset,
        batch_size=4,  # oneminor把alldata喂advancego
        shuffle=False,  # not打乱，methodconvenientit背诵
        collate_fn=fusion_collate_fn
    )

    # 4. initializationmodel
    model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])
    model.to(device)
    model.train()

    # 5. optimizer (using稍largeoflearning rateaddspeedcollect敛)
    # previousis 0.01，ourhereuse 0.02 testtest，godrop weight_decay preventit“studynotwill”
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    # 6. loss function
    criterion = CustomYOLOLoss(model,
                               hyp=cfg['hyperparameters'],
                               alpha=cfg['hyperparameters']['fl_alpha'],
                               gamma=cfg['hyperparameters']['fl_gamma'])

    # 7. 疯狂loop 100 minor (Overfit Loop)
    losses = []
    box_losses = []
    cls_losses = []

    print("\nStarting Loop (100 Epochs)...")
    start_time = time.time()

    for epoch in range(100):
        total_loss = 0

        for batch in loader:
            # dataabove GPU
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

            # recordscore/dividetermloss (Box, Cls)
            # loss_items is detached tensor
            box_losses.append(loss_items[0].item())
            cls_losses.append(loss_items[1].item())

        losses.append(total_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/100 | Total Loss: {total_loss:.4f} | Box: {loss_items[0]:.4f} | Cls: {loss_items[1]:.4f}")

    print(f"\n✅ testend! consumewhen: {time.time() - start_time:.2f}s")

    # 8. resultanalysisandvisualization
    initial_loss = losses[0]
    final_loss = losses[-1]

    print("------------------------------------------------------------")
    print(f"initialstart Loss: {initial_loss:.4f}")
    print(f"final Loss: {final_loss:.4f}")

    if final_loss < initial_loss * 0.1:
        print("🎉🎉 conclusion: throughtest！modelspecificpreparestudylearnability/canability。")
        print("    -> the previous issue should be：trainingwheelcounttooleast or learning ratenotcombinesuitable。")
    elif final_loss < initial_loss * 0.5:
        print("⚠️ conclusion: 勉strongthrough。Loss belowreducepast tense markerbutnot够fast。")
        print("    -> mayneedadjustmentexceedparameter (lr, momentum)。")
    else:
        print("❌ conclusion: testfailed！modelcompletelystudynotadvancego。")
        print("    -> extremehavemayis：gradientbreak裂 (Fusionmodulecodeproblem) or labelscore/divideconfigurefailed。")

    # save Loss curvelineimage
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Total Loss')
    plt.title('Overfit Test Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('runs/overfit_curve.png')
    print("📊 Loss curvelinealreadysaveto runs/overfit_curve.png")

    # 9. finalconfidencecheck (seeseeisnotisstillis 0.01)
    print("\n🔍 checkfinalconfidence...")
    with torch.no_grad():
        model.eval()
        inputs = {'rgb': batch['rgb'].to(device), 'thermal': batch['thermal'].to(device)}
        preds = model(inputs)
        if isinstance(preds, (list, tuple)): preds = preds[0]

        # checksee Class Score
        # preds: [B, 4+nc, Anchors] -> [:, 4:, :]
        cls_scores = preds[:, 4:, :]
        max_conf = cls_scores.max().item()
        print(f"   final Max Conf: {max_conf:.4f}")

        if max_conf > 0.5:
            print("✅ confidencepositiveoften！modelsurerealrecordkeeppast tense markertarget。")
        else:
            print("❌ confidencestillverylow。Loss reducepast tense markermayisbecausefor/as Box fallbackreturnat/inbegindouse，butscore/divideclassnotstudywill。")


if __name__ == '__main__':
    run_overfit()