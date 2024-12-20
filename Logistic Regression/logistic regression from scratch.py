import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class LogisticRegression:
    
    def __init__(self, lr = 0.001, n_inters = 1000):
        self.lr = lr
        self.n_inters = n_inters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # init parameters
        
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        
        for _ in range(self.n_inters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted  = self._sigmoid(linear_model)
            
            dw = (1 / n_sample) * np.dot(X.T, (y_predicted - y))
            
            db = (1 / n_sample) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predicted(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
        
    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
        
        
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regressor = LogisticRegression(lr = 0.0001, n_inters = 1000)
regressor.fit(X_train, y_train)
predictions = regressor.predicted(X_test)

print("LR classification accuracy:{}".format(accuracy(y_test, predictions)))


