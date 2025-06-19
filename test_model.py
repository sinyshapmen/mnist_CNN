import argparse
import random
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms

from src.model import MNISTModel
from src.inference import Inference


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST Inference Script")
    parser.add_argument("--index", type=int, help="Index of MNIST test image")
    parser.add_argument("--image", type=str, help="Path to custom image")
    return parser.parse_args()


def load_custom_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0), image


def main():
    args = parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = MNISTModel()
    infer = Inference(model=model, device=device)

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found at: {args.image}")
            return

        image_tensor, raw_image = load_custom_image(args.image)
        pred = infer.predict(image_tensor)

        plt.imshow(raw_image, cmap="gray")
        plt.title(f"Predicted Label: {pred} (from --image)")
        plt.axis("off")
        plt.show()

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

        idx = args.index if args.index is not None else random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[idx]
        pred = infer.predict(image.unsqueeze(0))

        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"True Label: {label}, Predicted Label: {pred} (index: {idx})")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
