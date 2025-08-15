import torch
import torchmetrics

'''
class MetricCollection:
    def __init__(self, device, num_classes: int, task_type: str, threshold: float = 0.5):
        """
        device      -> 'cpu' ou 'cuda'
        num_classes -> número de classes (binary=2; multilabel=labels)
        task_type   -> 'binary', 'multiclass', 'multilabel'
        threshold   -> limiar para binarizar predições no caso de binary/multilabel
        """
        self.device = device
        self.task_type = task_type
        self.num_classes = num_classes
        self.threshold = threshold

        if task_type == "binary":
            self.metrics = {
                "accuracy": torchmetrics.Accuracy(task="binary").to(device),
                "precision": torchmetrics.Precision(task="binary").to(device),
                "recall": torchmetrics.Recall(task="binary").to(device),
                "f1": torchmetrics.F1Score(task="binary").to(device),
            }
        elif task_type == "multiclass":
            self.metrics = {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device),
                "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
                "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
            }
        elif task_type == "multilabel":
            self.metrics = {
                "accuracy": torchmetrics.Accuracy(task="multilabel", num_labels=num_classes).to(device),
                "precision": torchmetrics.Precision(task="multilabel", num_labels=num_classes, average="macro").to(device),
                "recall": torchmetrics.Recall(task="multilabel", num_labels=num_classes, average="macro").to(device),
                "f1": torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average="macro").to(device),
            }
        else:
            raise ValueError(f"Tarefa inválida: {task_type}. Use 'binary', 'multiclass' ou 'multilabel'.")

    def _logits_to_preds(self, logits):
        """Converte logits em predições no formato esperado pelas métricas"""
        if self.task_type == "binary":
            probs = torch.sigmoid(logits)
            return (probs >= self.threshold).long().squeeze()
        elif self.task_type == "multiclass":
            return torch.argmax(logits, dim=1)
        elif self.task_type == "multilabel":
            probs = torch.sigmoid(logits)
            return (probs >= self.threshold).long()
        else:
            raise ValueError("Tarefa inválida.")

    def update(self, logits, targets):
        preds = self._logits_to_preds(logits)
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self):
        return {name: metric.compute().item() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

'''


'''
class MetricCollection:
    def __init__(self, device, num_classes: int, task_type: str):
        self.device = device
        self.task_type = task_type
        self.num_classes = num_classes

        if task_type == "binary":
            self.metrics = {
                "accuracy": torchmetrics.Accuracy(task="binary").to(device),
                "precision": torchmetrics.Precision(task="binary").to(device),
                "recall": torchmetrics.Recall(task="binary").to(device),
                "f1": torchmetrics.F1Score(task="binary").to(device),
            }
        elif task_type == "multiclass":
            self.metrics = {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device),
                "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
                "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
            }
        elif task_type == "multilabel":
            self.metrics = {
                "accuracy": torchmetrics.Accuracy(task="multilabel", num_labels=num_classes).to(device),
                "precision": torchmetrics.Precision(task="multilabel", num_labels=num_classes, average="macro").to(device),
                "recall": torchmetrics.Recall(task="multilabel", num_labels=num_classes, average="macro").to(device),
                "f1": torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average="macro").to(device),
            }
        else:
            raise ValueError(f"Tarefa inválida: {task_type}. Use 'binary', 'multiclass' ou 'multilabel'.")

    def update(self, preds, targets):
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self):
        return {name: metric.compute().item() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
'''
            
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