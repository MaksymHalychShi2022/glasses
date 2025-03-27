import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import GlassesDataset  # Import dataset class


# Constants (hyperparameters)
TRAIN_CSV = "data/train_cleaned.csv"  # Path to the train CSV file
VAL_CSV = "data/val_cleaned.csv"  # Path to the validation CSV file
IMG_DIR = "data/faces-spring-2020/faces-spring-2020/"  # Directory where images are stored
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


def train(model, dataloader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)  # Move data to the device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()

        # Optimize the model
        optimizer.step()

        # Track statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = (correct / total) * 100

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during validation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = (correct / total) * 100

    return avg_loss, accuracy


def main():
    # Set the device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the datasets
    train_dataset = GlassesDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR, mode="train")
    val_dataset = GlassesDataset(csv_file=VAL_CSV, img_dir=IMG_DIR, mode="val")

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(pretrained=True)

    # Modify the final layer for binary classification
    num_ftrs = model.fc.in_features  # Number of input features for the final layer
    model.fc = nn.Linear(num_ftrs, 2)  # Change the output layer to have 2 classes (Glasses/No Glasses)

    # Optionally, freeze the early layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Move the model to the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # Only optimize the final fully connected layer

    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train the model
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validate the model
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved as best_model.pth")

    print("Training complete!")


if __name__ == "__main__":
    main()
