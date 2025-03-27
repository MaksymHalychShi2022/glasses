import torch
from torchvision import transforms, models
from PIL import Image


class GlassesModelInference:
    def __init__(self, model_path: str = "best_model.pth", device: str = "cpu") -> None:
        """
        Initialize the GlassesModelInference instance.

        :param model_path: Path to the trained model's state_dict file.
        :param device: The device to run the model on ("cpu" or "cuda").
        """
        # Validate device input and set the device
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be either 'cpu' or 'cuda'.")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the model (using ResNet18, as per your training code)
        self.model = models.resnet18(pretrained=True)

        # Modify the final layer for binary classification
        num_ftrs = self.model.fc.in_features  # Number of input features for the final layer
        self.model.fc = torch.nn.Linear(num_ftrs, 2)  # Change the output layer to have 2 classes (Glasses/No Glasses)

        # Load the model to the specified device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)  # Move model to the correct device
        self.model.eval()  # Set the model to evaluation mode

        # Define the preprocessing transformations
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # Resize the image
                transforms.ToTensor(),  # Convert PIL image to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on ImageNet stats
            ]
        )

    def infer(self, frame: Image.Image) -> str:
        """
        Given a frame, preprocess it and make a prediction using the model.

        :param frame: The input frame as a PIL Image to process.
        :return: A string representing the predicted class label ("Glasses" or "No Glasses").
        """
        # Preprocess the frame
        input_tensor = self.preprocess(frame)
        input_batch = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Run the model on the frame without calculating gradients
        with torch.no_grad():
            output = self.model(input_batch)

        # Get the predicted class (0 for 'No Glasses', 1 for 'Glasses')
        _, predicted = torch.max(output, 1)
        label = "Glasses" if predicted.item() == 1 else "No Glasses"

        return label


if __name__ == "__main__":
    image_path = "data/faces-spring-2020/faces-spring-2020/face-5000.png"
    image = Image.open(image_path).convert("RGB")

    # Automatically determine the device if not specified
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlassesModelInference(model_path="best_model.pth", device=device)
    label = model.infer(image)
    print(f"Image: {image_path}, Predicted Label: {label}")
