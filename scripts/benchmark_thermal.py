from ultralytics import YOLO


def train_baseline():
    print("🚀 开始训练官方纯 RGB 基准模型...")
    # 直接加载官方 yolov8n.pt，不使用任何我们的自定义代码
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data='configs/data/r_livit_thermal.yaml',  # <--- 换成 thermal 的配置
        epochs=50,
        imgsz=640,
        batch=16,
        name='baseline_thermal_only',  # <--- 改个名字方便区分
        project='runs',
        workers=0,  # <--- 保持 0 (Windows 必须)
        exist_ok=True,
        pretrained=True,
        # ch=3                                     # 注意：默认按 3 通道读取，哪怕是灰度图也会复制成 3 通道，这样可以用预训练权重。不用改。
    )

    print("✅ 基准训练完成！")
    print(f"最终 mAP50: {results.box.map50}")
    print(f"最终 Recall: {results.box.mr}")


if __name__ == '__main__':
    train_baseline()