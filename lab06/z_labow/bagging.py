from sklearn.tree import DecisionTreeClassifier
import numpy as np

class BaggingClassifier:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.clfs = []

    def fit(self, X, y):
        for i in range(self.n_samples):
            indices = np.random.choice(X, size=X.shape[0], replace=True)
            X_sampled = X[indices]
            y_sampled = y[indices]
            clf = DecisionTreeClassifier()
            clf.fit(X=X_sampled, y=y_sampled)
            self.clfs.append(clf)
        return self.clfs
    
    def predict(self, X_test):
        predictions = np.array([clf.predict(X_test) for clf in self.clfs])
        
        # majority voting
        num_0 = 0
        num_1 = 0
        for i in range(self.n_samples):
            pred_i = predictions[i]
            


