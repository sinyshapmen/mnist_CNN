import torch

class Inference:
    def __init__(self, model, device='cpu', weights_path="weights/best_model.pth"):
        self.device = device
        self.model = model.to(self.device)
        self.weights_path = weights_path
        self._load_weights()

    def _load_weights(self):
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            _, pred = torch.max(output, 1)
            return pred.item()