import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initializes the MobileNetV3 model for classification.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
        """
        super(MobileNetV3Classifier, self).__init__()
        # Load the 'small' version of MobileNetV3 for efficiency
        self.mobilenet_v3 = models.mobilenet_v3_small(pretrained=pretrained)

        # Get the number of features in the last layer
        num_ftrs = self.mobilenet_v3.classifier[-1].in_features

        # Replace the last layer with a new one matching num_classes
        self.mobilenet_v3.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.mobilenet_v3(x)

# Example usage (optional):
# if __name__ == '__main__':
#     # Example with 2 classes
#     model = MobileNetV3Classifier(num_classes=2)
#     print(model)

#     # Example with 10 classes and no pretrained weights
#     # model_10 = MobileNetV3Classifier(num_classes=10, pretrained=False)
#     # print(model_10)

#     # Example forward pass with dummy data
#     dummy_input = torch.randn(4, 3, 224, 224) # Batch of 4 images, 3 channels, 224x224 size
#     output = model(dummy_input)
#     print("Output shape:", output.shape) # Should be [4, num_classes] 