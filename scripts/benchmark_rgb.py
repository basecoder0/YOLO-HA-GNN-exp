from ultralytics import YOLO


def train_baseline():
    print("🚀 starttrainingofficialpure RGB baselinemodel...")
    # directlyloadofficial yolov8n.pt，do not use any of our custom code
    model = YOLO('yolov8n.pt')

    # training
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

    print("✅ baselinetrainingcompleted！")
    print(f"final mAP50: {results.box.map50}")
    print(f"final Recall: {results.box.mr}")


if __name__ == '__main__':
    train_baseline()