# Glasses Classification with PyTorch

This project uses a Convolutional Neural Network (CNN) to classify images of faces as either having glasses or not. The model is trained using a dataset of images and their corresponding labels, with various preprocessing and augmentation techniques applied.

## Installation

1. Clone this repository:

    ```bash
    https://github.com/MaksymHalychShi2022/glasses.git
    cd glasses
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
    ```

3. Install the necessary dependencies:

    - For GPU (CUDA) support:

        ```bash
        pip install -r requirements.txt
        ```

    - For CPU-only support:

        ```bash
        pip install -r requirements-cpu.txt
        ```

4. (Optional) Install pre-commit hooks:

    ```bash
    pre-commit install
    ```

## Dataset 

The dataset used in this project was taken from [Kaggle - Glasses or No Glasses](https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses). It contains 5000 images of faces, with labels for the first 4500 images indicating whether the person is wearing glasses or not. The last 500 images do not have labels and can be used for inference demonstrations.

Download it and extract to `data/` directory:

```text
data/
  ├── faces-spring-2020/
      ├── faces-spring-2020/
          ├── face-0001.png
          ├── face-0002.png
          └── ...
  ├── train.csv
  ├── test.csv
```

Run this script to split labeled dataset into train and validation sets:

```bash
python split_cleaned.py
```

## Training

To train the model, run:

```bash
python train.py
```

This script will train the model on the dataset and save the best-performing model based on validation accuracy as `best_model.pth`.

## Inference

To use the trained model for inference on a new image, run the following:

```bash
python inference.py
```

This will load the trained model and classify the specified image as either "Glasses" or "No Glasses".