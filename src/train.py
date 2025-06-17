import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from src.logger import Logger
from torch.utils.tensorboard import SummaryWriter



class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=0.1, patience=5, max_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.patience = patience
        self.max_epochs = max_epochs

        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_wts = None

        self.train_losses = []
        self.val_losses = []

        self.weights_path = "weights/best_model.pth"
        self.logger = Logger(name='trainer', log_file='logs/train.log').get_logger()

        log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    def train_epoch(self):
        self.model.train()
        running_loss = 0
        for images, labels in self.train_loader:
            images = images.view(-1, 28 * 28).to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        return running_loss / len(self.train_loader.dataset)

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.view(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item() * images.size(0)

        return running_loss / len(self.val_loader.dataset)

    def fit(self):
        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.logger.info(f"Epoch [{epoch+1}/{self.max_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            self.writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch + 1)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                self.best_model_wts = self.model.state_dict()
                torch.save(self.best_model_wts, self.weights_path)
                self.logger.info("Best model saved.")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if self.best_model_wts:
            self.model.load_state_dict(torch.load(self.weights_path))
            self.logger.info("Loaded best model weights after training.")

        self.writer.close()

        return self.train_losses, self.val_losses

    def evaluate(self, test_loader):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        return y_true, y_pred
