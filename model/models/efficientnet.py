import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initializes the EfficientNetB0 model for classification.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
        """
        super(EfficientNetClassifier, self).__init__()
        # Load the EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)

        # Get the number of features in the classifier layer
        # EfficientNet's classifier is typically: (dropout): Dropout(p=0.2, inplace=True), (fc): Linear(in_features=1280, out_features=1000, bias=True)
        num_ftrs = self.efficientnet.classifier[1].in_features

        # Replace the final layer (the Linear layer) with a new one matching num_classes
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# Example usage (optional):
# if __name__ == '__main__':
#     # Example with 2 classes
#     model = EfficientNetClassifier(num_classes=2)
#     print(model)

#     # Example with 10 classes and no pretrained weights
#     # model_10 = EfficientNetClassifier(num_classes=10, pretrained=False)
#     # print(model_10)

#     # Example forward pass with dummy data
#     # Note: EfficientNet-B0 expects input size 224x224
#     dummy_input = torch.randn(4, 3, 224, 224) # Batch of 4 images, 3 channels, 224x224 size
#     output = model(dummy_input)
#     print("Output shape:", output.shape) # Should be [4, num_classes] 