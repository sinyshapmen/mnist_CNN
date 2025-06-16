from data.load_data import DataLoaderMNIST
from src.inference import Inference
from src.metrics import Metrics
from src.model import MNISTModel
from src.train import Trainer

import torch

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_loader = DataLoaderMNIST()
    train_loader, val_loader, test_loader = data_loader.get_loaders()

    model = MNISTModel().to(device)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device)
    train_losses, val_losses = trainer.fit()

    torch.save(trainer.best_model_wts, 'weights/best_model.pth')
    print("Best model weights saved to Weights/best_model.pth")

    y_true, y_pred = trainer.evaluate(test_loader)

    metrics = Metrics(y_true, y_pred)
    print(metrics.summary())


if __name__ == "__main__":
    main()


    