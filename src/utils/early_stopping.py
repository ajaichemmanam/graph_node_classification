class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_val_acc = 0
        self.best_test_acc = 0

    def __call__(self, val_acc, test_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
