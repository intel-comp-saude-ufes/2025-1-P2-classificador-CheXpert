class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min", monitor="val_loss"):
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor

        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.best_flag = False

    def step(self, metrics: dict) -> bool:
        if self.monitor not in metrics:
            raise KeyError(f"Metric '{self.monitor}' was not found in the metrics dictionary.")

        score = metrics[self.monitor]

        improvement = (
            score < self.best_score - self.min_delta if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improvement:
            self.best_score = score
            self.counter = 0
            self.best_flag = True
            return False
        else:
            self.counter += 1
            self.best_flag = False
            return self.counter >= self.patience

    def is_best(self) -> bool:
        return self.best_flag