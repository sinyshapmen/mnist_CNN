import torch
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.model import MNISTModel
from src.inference import Inference

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = MNISTModel()
    infer = Inference(model=model, device=device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    idx = random.randint(0, len(test_dataset) - 1)
    image, label = test_dataset[idx]
    pred = infer.predict(image.unsqueeze(0))

    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True Label: {label}, Predicted Label: {pred}")

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()