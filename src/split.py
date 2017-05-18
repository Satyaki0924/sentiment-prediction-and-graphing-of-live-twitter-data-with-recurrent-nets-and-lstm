class Split(object):
    def __init__(self, features, labels, split=0.8):
        self.split = split
        self.features = features
        self.labels = labels
        self.val_X, self.val_y = None, None

    def train_split(self):
        split_idx = int(len(self.features) * self.split)
        train_X, self.val_X = self.features[:split_idx], self.features[split_idx:]
        train_y, self.val_y = self.labels[:split_idx], self.labels[split_idx:]
        return train_X, train_y

    def test_split(self):
        self.train_split()
        test_idx = int(len(self.val_X) * 0.5)
        self.val_X, test_X = self.val_X[:test_idx], self.val_X[test_idx:]
        self.val_y, test_y = self.val_y[:test_idx], self.val_y[test_idx:]
        return self.val_X, test_X, self.val_y, test_y
