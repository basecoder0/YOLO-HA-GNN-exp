from ultralytics import YOLO


def train_baseline():
    print("🚀 starttrainingofficialpure RGB baselinemodel...")
    # directlyloadofficial yolov8n.pt，do not use any of our custom code
    model = YOLO('yolov8n.pt')

    # training
    results = model.train(
        data='configs/data/r_livit_thermal.yaml',  # <--- changegenerate thermal ofconfiguration
        epochs=50,
        imgsz=640,
        batch=16,
        name='baseline_thermal_only',  # <--- changeanamemethodconvenientareascore/divide
        project='runs',
        workers=0,  # <--- keep 0 (Windows mustmust)
        exist_ok=True,
        pretrained=True,
        # ch=3                                     # note：defaultper 3 channelread，which怕isgraydegreeimagealsowillcomplexcontrolgenerate 3 channel，like thisoptionalby/withusepredict/pretrainingweight。notusechange。
    )

    print("✅ baselinetrainingcompleted！")
    print(f"final mAP50: {results.box.map50}")
    print(f"final Recall: {results.box.mr}")


if __name__ == '__main__':
    train_baseline()