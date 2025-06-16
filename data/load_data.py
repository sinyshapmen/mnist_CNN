import torch
from torchvision import datasets, transforms

class DataLoaderMNIST:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(self):
        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)

    def get_loaders(self):
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self.prepare_data()

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader
