import torch


class GlassesModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(GlassesModel, self).__init__()

        # Define the model using torch.nn.Sequential
        self.model = torch.nn.Sequential(
            # First convolutional layer (input channels: 3 (RGB), output channels: 6)
            # Input shape: (batch_size, 3, 256, 256) => 3 channels (RGB), 256x256 image size
            torch.nn.Conv2d(3, 6, kernel_size=5),  # Output shape after this layer: (batch_size, 6, 252, 252)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # Max pooling with kernel size 2, output shape: (batch_size, 6, 126, 126)
            # Second convolutional layer
            torch.nn.Conv2d(6, 16, kernel_size=5),  # Output shape after this layer: (batch_size, 16, 122, 122)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # Max pooling with kernel size 2, output shape: (batch_size, 16, 61, 61)
            # Flatten the output to feed into fully connected layers
            torch.nn.Flatten(),  # Output shape after flattening: (batch_size, 16 * 61 * 61)
            # Fully connected layers
            torch.nn.Linear(
                16 * 61 * 61, 120
            ),  # Input shape: (batch_size, 16 * 61 * 61), output shape: (batch_size, 120)
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),  # Input shape: (batch_size, 120), output shape: (batch_size, 84)
            torch.nn.ReLU(),
            # Final output layer (num_classes outputs)
            torch.nn.Linear(84, num_classes),  # Input shape: (batch_size, 84), output shape: (batch_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Example to instantiate the model
    model = GlassesModel(num_classes=2)
    print(model)
