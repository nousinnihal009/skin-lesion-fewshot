class EarlyStopper:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.delta = delta

    def early_stop(self, current_loss):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
