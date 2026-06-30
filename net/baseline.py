"""baseline.py

Feature extractor backbones (ConvNeXt, EfficientNet-V2, ResNet families),
final 1x1 convolutional projection layer, and adaptive pooling utilities.
"""

import argparse

import torch
import torch.nn as nn
from torchvision import models

def convnext_tiny_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create a ConvNeXt-Tiny feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with avgpool and classifier removed.
    """
    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None,
        **kwargs
    )
    with torch.no_grad():
        model.avgpool = nn.Identity()    # Remove average pooling
        model.classifier = nn.Identity()   # Remove classifier head
    return model

def convnext_small_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create a ConvNeXt-Small feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with avgpool and classifier removed.
    """
    model = models.convnext_small(
        weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None,
        **kwargs
    )
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
    return model

def convnext_base_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create a ConvNeXt-Base feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with avgpool and classifier removed.
    """
    model = models.convnext_base(
        weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None,
        **kwargs
    )
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
    return model

def convnext_large_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create a ConvNeXt-Large feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with avgpool and classifier removed.
    """
    model = models.convnext_large(
        weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None,
        **kwargs
    )
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
    return model

class EfficientNetFeatureExtractor(nn.Module):
    """Wraps the convolutional feature block of an EfficientNet model."""

    def __init__(self, base_model: nn.Module):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.features = base_model.features

    def forward(self, x):
        # Returns the feature map without flattening
        return self.features(x)

def efficientnet_v2_s_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create an EfficientNet_V2_S feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with pooling and classification layers removed.
    """
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None,
        **kwargs
    )
    
    return EfficientNetFeatureExtractor(model)

def efficientnet_v2_m_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create an EfficientNet_V2_M feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with pooling and classification layers removed.
    """
    model = models.efficientnet_v2_m(
        weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None,
        **kwargs
    )

    return EfficientNetFeatureExtractor(model)

def efficientnet_v2_l_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create an EfficientNet_V2_L feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V1 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with pooling and classification layers removed.
    """
    model = models.efficientnet_v2_l(
        weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None,
        **kwargs
    )

    return EfficientNetFeatureExtractor(model)


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet50 feature extractor that returns the output of the last convolutional block
    without applying average pooling or flattening.
    This extractor is compatible with ResNet variants that expose the same attributes
    (conv1, bn1, relu, maxpool, layer1..layer4), such as ResNet101 and ResNet152.
    """
    def __init__(self, original_model: nn.Module):
        super(ResNet50FeatureExtractor, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x has shape [batch, channels, height, width]
        return x

def resnet50_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create a ResNet50 feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V2 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor with average pooling and fully-connected layers removed.
    """
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
        **kwargs
    )

    return ResNet50FeatureExtractor(model)

def resnet101_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create a ResNet101 feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V2 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor (same interface as ResNet50 extractor).
    """
    model = models.resnet101(
        weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None,
        **kwargs
    )
    return ResNet50FeatureExtractor(model)

def resnet152_features(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Create a ResNet152 feature extractor.
    
    Args:
        pretrained (bool): If True, load IMAGENET1K_V2 pretrained weights.
        **kwargs: Additional arguments for model initialization.
    
    Returns:
        nn.Module: Feature extractor (same interface as ResNet50 extractor).
    """
    model = models.resnet152(
        weights=models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None,
        **kwargs
    )
    return ResNet50FeatureExtractor(model)

def get_feature_extractor(args) -> nn.Module:
    """
    Get the feature extractor based on the provided arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments with keys:
            - base_model (str): Model name. One of:
              "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
              "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
              "resnet50", "resnet101", "resnet152".
            - pretrain (bool): If True, use pretrained weights.
    
    Returns:
        nn.Module: The selected feature extractor.
    
    Raises:
        ValueError: If an unsupported model name is provided.
    """
    base_model = args.base_model.lower()
    extractor_mapping = {
        "convnext_tiny": convnext_tiny_features,
        "convnext_small": convnext_small_features,
        "convnext_base": convnext_base_features,
        "convnext_large": convnext_large_features,
        "efficientnet_v2_s": efficientnet_v2_s_features,
        "efficientnet_v2_m": efficientnet_v2_m_features,
        "efficientnet_v2_l": efficientnet_v2_l_features,
        "resnet50": resnet50_features,
        "resnet101": resnet101_features,
        "resnet152": resnet152_features,
    }
    
    if base_model not in extractor_mapping:
        raise ValueError(f"Unsupported base model: {args.base_model}")
    
    # Instantiate and return the selected feature extractor.
    extractor_fn = extractor_mapping[base_model]
    return extractor_fn(pretrained=args.pretrain)

def get_final_conv_layer(args, in_channels: int) -> nn.Module:
    """
    Creates a final convolutional layer based on the provided arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments containing:
            - ch_last_conv (int): Number of filters per class. If 0, no convolution is applied.
            - num_classes (int): Number of output classes.
        in_channels (int): Number of input channels from the feature extractor.
    
    Returns:
        nn.Module: Either a convolutional layer with appropriate output channels
                  or an Identity layer if ch_last_conv is 0.
    """
    # If ch_last_conv is 0, no convolution is applied
    if args.ch_last_conv == 0:
        return nn.Identity()
    
    # Calculate the number of output channels based on ch_last_conv
    # If ch_last_conv is 1, out_channels = num_classes
    # Otherwise, out_channels = num_classes * ch_last_conv
    if args.ch_last_conv == 1:
        out_channels = args.num_classes
    else:
        out_channels = args.num_classes * args.ch_last_conv
    
    # Create and return the convolutional layer
    # Using kernel_size=1 for 1x1 convolution
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0
    )

def get_pooling(args: argparse.Namespace) -> nn.Module:
    """
    Returns a pooling layer based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments containing:
            - pooling (str): Pooling type; expected values: "avg" or "max".

    Returns:
        nn.Module: An adaptive pooling layer that transforms features of shape
                   [batch_size, channels, height, width] to [batch_size, channels, 1, 1].

    Raises:
        ValueError: If args.pooling is not "avg" or "max".
    """
    pooling_type = args.pooling.lower()
    if pooling_type == "avg":
        return nn.AdaptiveAvgPool2d((1, 1))
    elif pooling_type == "max":
        return nn.AdaptiveMaxPool2d((1, 1))
    else:
        raise ValueError(f"Unsupported pooling type: {args.pooling}. Choose 'avg' or 'max'.")