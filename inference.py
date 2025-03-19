import torch
from torchvision import transforms
from model import GlassesModel


class GlassesModelInference:
    def __init__(self, model_path: str = "best_model.pth", num_classes: int = 2, device: str = "cpu") -> None:
        """
        Initialize the GlassesModelInference instance.

        :param model_path: Path to the trained model's state_dict file.
        :param num_classes: Number of output classes for the model (default is 2).
        :param device: The device to run the model on ("cpu" or "cuda").
        """
        # Validate device input and set the device
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be either 'cpu' or 'cuda'.")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = GlassesModel(num_classes=num_classes)

        # Load the model to the specified device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)  # Move model to the correct device
        self.model.eval()  # Set the model to evaluation mode

        # Define the preprocessing transformations
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # Resize directly on the image
                transforms.ToTensor(),  # Convert PIL image to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def infer(self, frame: torch.Tensor) -> str:
        """
        Given a frame, preprocess it and make a prediction using the model.

        :param frame: The input frame as a tensor or numpy array to process.
        :return: A string representing the predicted class label ("Glasses" or "No Glasses").
        """
        # Preprocess the frame
        input_tensor = self.preprocess(frame)
        input_batch = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Run the model on the frame without calculating gradients
        with torch.no_grad():
            output = self.model(input_batch)

        # Get the predicted class (0 for 'no glasses', 1 for 'glasses')
        _, predicted = torch.max(output, 1)
        label = "Glasses" if predicted.item() == 1 else "No Glasses"

        return label


if __name__ == "__main__":
    from PIL import Image

    image_path = "data/faces-spring-2020/faces-spring-2020/face-5000.png"
    image = Image.open(image_path).convert("RGB")

    # Automatically determine the device if not specified
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlassesModelInference(model_path="best_model.pth", device=device)
    label = model.infer(image)
    print(f"image: {image_path}, label: {label}")
