import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple, Dict


class DualResNet(nn.Module):
    """
    双branchResNetskeletondrynetworknetwork
    score/divideotherprocessingvisible light and infrared thermal imagingimage
    """

    def __init__(self,
                 backbone_type: str = 'resnet50',
                 pretrained: bool = False,
                 shared_weights: bool = False,
                 out_channels: List[int] = [512, 1024, 2048]):  # modifyfor/asactualofResNet50channelcount
        super().__init__()

        self.backbone_type = backbone_type
        self.shared_weights = shared_weights
        self.out_channels = out_channels

        # createskeletondrynetworknetwork
        if shared_weights:
            self.rgb_backbone = self._create_backbone(backbone_type, pretrained)
            self.thermal_backbone = self.rgb_backbone
        else:
            self.rgb_backbone = self._create_backbone(backbone_type, pretrained)
            self.thermal_backbone = self._create_backbone(backbone_type, pretrained)

        # featurelayername称mapping - usingactualofResNet50channelcount
        self.feature_layers = {
            'layer1': 256,  # ResNet50 layer1 outputchannel
            'layer2': 512,  # ResNet50 layer2 outputchannel
            'layer3': 1024,  # ResNet50 layer3 outputchannel
            'layer4': 2048  # ResNet50 layer4 outputchannel
        }

    def _create_backbone(self, backbone_type: str, pretrained: bool) -> nn.Module:
        """createskeletondrynetworknetwork"""
        if backbone_type == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif backbone_type == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif backbone_type == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif backbone_type == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # removedfinaloffull连connectlayerandaveragepooltransformlayer
        modules = list(backbone.children())[:-2]
        return nn.Sequential(*modules)

    def forward(self, rgb_input: torch.Tensor, thermal_input: torch.Tensor) -> Dict:
        """firsttowardspropagation - extractmultiple尺degreefeature"""
        #print(f"DualResNet input - RGB: {rgb_input.shape}, Thermal: {thermal_input.shape}")

        rgb_features = {}
        thermal_features = {}

        # RGBbranchfeatureextract
        x = rgb_input

        # extractnotsamelayeroffeature
        x = self.rgb_backbone[0](x)  # conv1
        x = self.rgb_backbone[1](x)  # bn1
        x = self.rgb_backbone[2](x)  # relu
        x = self.rgb_backbone[3](x)  # maxpool

        x = self.rgb_backbone[4](x)  # layer1
        rgb_features['layer1'] = x
        #print(f"RGB layer1: {x.shape}")

        x = self.rgb_backbone[5](x)  # layer2
        rgb_features['layer2'] = x
        #print(f"RGB layer2: {x.shape}")

        x = self.rgb_backbone[6](x)  # layer3
        rgb_features['layer3'] = x
        #print(f"RGB layer3: {x.shape}")

        x = self.rgb_backbone[7](x)  # layer4
        rgb_features['layer4'] = x
        #print(f"RGB layer4: {x.shape}")

        # Thermalbranchfeatureextract
        y = thermal_input
        y = self.thermal_backbone[0](y)  # conv1
        y = self.thermal_backbone[1](y)  # bn1
        y = self.thermal_backbone[2](y)  # relu
        y = self.thermal_backbone[3](y)  # maxpool

        y = self.thermal_backbone[4](y)  # layer1
        thermal_features['layer1'] = y
        #print(f"Thermal layer1: {y.shape}")

        y = self.thermal_backbone[5](y)  # layer2
        thermal_features['layer2'] = y
        #print(f"Thermal layer2: {y.shape}")

        y = self.thermal_backbone[6](y)  # layer3
        thermal_features['layer3'] = y
        #print(f"Thermal layer3: {y.shape}")

        y = self.thermal_backbone[7](y)  # layer4
        thermal_features['layer4'] = y
        #print(f"Thermal layer4: {y.shape}")

        return {
            'rgb': rgb_features,
            'thermal': thermal_features
        }

    def get_output_channels(self) -> List[int]:
        """getoutputchannelcount"""
        return list(self.feature_layers.values())