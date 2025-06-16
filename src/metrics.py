from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.results = self.calculate_metrics()

    def calculate_metrics(self):
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, average='macro')
        recall = recall_score(self.y_true, self.y_pred, average='macro')
        f1 = f1_score(self.y_true, self.y_pred, average='macro')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def summary(self):
        return self.results