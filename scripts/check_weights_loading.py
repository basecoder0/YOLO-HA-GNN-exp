import torch
import sys
import os
import yaml

sys.path.append(os.getcwd())
from models.yolo.yolo_with_fusion import YOLOWithFusion


def check_weights():
    print("🔍 positiveat/incheckpredict/pretrainingweightloadingsituationsituation...")

    # 1. loadofficial yolov8n.pt (asparameterrefer)
    yolo_pt = torch.load('weights/yolov8n.pt', map_location='cpu', weights_only=False)
    yolo_dict = yolo_pt['model'].float().state_dict() if 'model' in yolo_pt else yolo_pt

    # getofficialfirstlayerconvolutionofweight (model.0.conv.weight)
    # YOLOv8n the first module is usually Conv
    ref_key = 'model.0.conv.weight'
    if ref_key not in yolo_dict:
        # tryfindoneaexistof key
        ref_key = list(yolo_dict.keys())[0]

    ref_weight = yolo_dict[ref_key]
    print(f"✅ officialweight {ref_key} mean: {ref_weight.mean().item():.6f}")

    # 2. initializationouroffusionmodel
    config_path = 'configs/train/stage1.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = YOLOWithFusion(model_path='weights/yolov8n.pt', num_classes=cfg['model']['num_classes'])

    # 3. check RGB majordrywhether继承past tense markerweight
    # assumemodelstructureinhave backbone_rgb or similar
    # ourtraverse model.named_parameters() findlookslike backbone oflayer

    print("\n🔍 analysiswhenfirstmodelweightstatus...")

    rgb_backbone_loaded = False
    thermal_backbone_loaded = False

    # hereofnameget决for YOLOWithFusion ofspecificimplement
    # usuallyis self.backbone (ifonlyissingle流) orone self.rgb_backbone / self.thermal_backbone

    model_state = model.state_dict()

    # searchfindcorrespondingofficial ref_key oflayer
    # at/in YOLOWithFusion in，officialof model.0 maycorresponding model.rgb_stream.0 orone model.backbone.0

    matched_keys = []
    for k, v in model_state.items():
        if '0.conv.weight' in k and 'backbone' in k:  # roughly match the first layer
            matched_keys.append(k)

    if not matched_keys:
        print("⚠️ unable toautolocatemajordrynetworknetworkoffirstlayer，willprintfirst 5 layer names for manual verification：")
        for k in list(model_state.keys())[:5]:
            print(f"   {k}")

    for k in matched_keys:
        current_weight = model_state[k]
        diff = torch.abs(current_weight - ref_weight).sum().item()

        # ifweightcompletelyonesample (Diff = 0)，saybrightloadsuccessbutstillnottraining
        # ifweightcompletelynotsame (Diff large and looks like randomdistribution)，saybrightloadfailed，israndominitializationof
        # ifweightconnectclosebutnotfor/as0，saybrightloadsuccessand经toopast tense marker微adjust (butat/ininitializationstage should be the same)

        # thiswhenourisat/incheck "initializationafter" ofstatus，manage论aboveshouldcompletelyoneconsistent

        print(f"   layer: {k}")
        print(f"   andofficialweightdifference异: {diff:.6f}")

        if diff < 1e-5:
            print("   ✅ 判define: weightloadingsuccess！")
        else:
            print("   ❌ 判define: weightnotmatch！mayisrandominitialization。")


if __name__ == '__main__':
    check_weights()