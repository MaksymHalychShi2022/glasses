import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GlassesDataset(Dataset):
    def __init__(self, csv_file, img_dir, mode="train"):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            img_dir (str): Directory where the images are stored.
            mode (str): Mode for dataset - 'train', 'val', or 'test'.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mode = mode

        # Create a list of image paths and labels for faster access
        self.image_paths = [os.path.join(img_dir, f"face-{row[0]}.png") for row in self.data.values]
        self.labels = self.data.iloc[:, 1].values  # Assuming labels are in the second column

        # Define transformations based on the mode
        if self.mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.RandomAdjustSharpness(2),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif self.mode == "val" or self.mode == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes are 'train', 'val', 'test'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Check if the image exists to avoid errors
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return as a tuple (image, label)
        return image, label


if __name__ == "__main__":
    # Set up the paths for your dataset
    csv_file = "data/train_cleaned.csv"
    img_dir = "data/faces-spring-2020/faces-spring-2020/"

    # Create the dataset instance for 'train' mode
    dataset_train = GlassesDataset(csv_file=csv_file, img_dir=img_dir, mode="train")

    # Create the DataLoader instance
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)

    # Example to fetch a batch of data
    for batch_idx, (images, labels) in enumerate(dataloader_train):
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Just print one batch for demonstration
