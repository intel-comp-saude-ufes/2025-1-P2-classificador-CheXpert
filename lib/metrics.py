import torch
import torchmetrics

class MetricCollection:
    def __init__(self, device):
        self.device = device

        self.metrics = dict({})

    def register(self, name:str, metric):
        self.metrics[name] = metric.to(self.device)
        
    def update(self, preds, targets):
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self):
        return {name: metric.compute().item() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()