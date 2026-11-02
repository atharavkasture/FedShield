import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_classifier(num_classes=10, freeze_backbone=True):
    """
    Loads pre-trained ResNet-18 and replaces the final fully connected layer.
    Freezes all feature extraction layers to isolate DP noise to the classifier.
    """
    # Load ResNet-18 with ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace final linear layer (512 -> 10 classes for CIFAR-10)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.fc.requires_grad_(True)
    
    return model