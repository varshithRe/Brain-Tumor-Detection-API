import torch.nn as nn
import torchvision.models as models

def build_model(model_name = 'resnet18', num_classes = 2, pretrained = True):
    """
    Build and return a pre-trained model modified for binary classification.

    Parameters:
    model_name (str): Name of the pre-trained model to use.
    num_classes (int): Number of output classes for classification.
    pretrained (bool): Whether to use a model pre-trained on ImageNet.

    Returns:
    model (nn.Module): Modified pre-trained model.
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported. Choose 'resnet18' or 'resnet50'.")

    return model