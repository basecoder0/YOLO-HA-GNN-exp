from ultralytics import YOLO


def train_baseline():
    print("🚀 开始训练官方纯 RGB 基准模型...")
    # 直接加载官方 yolov8n.pt，不使用任何我们的自定义代码
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data='configs/data/r_livit_official.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='runs',
        name='baseline_rgb_only',
        device=0,
        exist_ok=True
    )

    print("✅ 基准训练完成！")
    print(f"最终 mAP50: {results.box.map50}")
    print(f"最终 Recall: {results.box.mr}")


if __name__ == '__main__':
    train_baseline()