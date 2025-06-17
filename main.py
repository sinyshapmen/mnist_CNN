from data.load_data import DataLoaderMNIST
from src.inference import Inference
from src.metrics import Metrics
from src.model import MNISTModel
from src.train import Trainer
from src.logger import Logger

import torch

def main():
    logger = Logger(name='main', log_file='logs/main.log').get_logger()

    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        data_loader = DataLoaderMNIST()
        train_loader, val_loader, test_loader = data_loader.get_loaders()
        logger.info("Data loaders created.")


        model = MNISTModel().to(device)
        logger.info("Model initialized.")

        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device)
        train_losses, val_losses = trainer.fit()
        logger.info("Training completed successfully.")

        torch.save(trainer.best_model_wts, 'weights/best_model.pth')
        logger.info("Best model weights saved to weights/best_model.pth")

        y_true, y_pred = trainer.evaluate(test_loader)
        metrics = Metrics(y_true, y_pred)

        metrics_results = metrics.summary()
        logger.info(f"Accuracy:  {metrics_results['accuracy']:.4f}")
        logger.info(f"Precision: {metrics_results['precision']:.4f}")
        logger.info(f"Recall:    {metrics_results['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics_results['f1']:.4f}")

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")



if __name__ == "__main__":
    main()


    