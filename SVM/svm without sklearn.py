import numpy as np
import matplotlib.pyplot as plt

# Define the SVM class
class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                if y_[i] * (np.dot(X[i], self.w) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X[i], y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Load the Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Take the first two features and select only two classes
X = X[y != 2, :2]
y = y[y != 2]

# Create an instance of the SVM
svm = SupportVectorMachine()

# Fit the SVM to the data
svm.fit(X, y)

# Define a function to plot the decision boundary
def plot_decision_boundary(X, y, svm):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

# Plot the decision boundary
plot_decision_boundary(X, y, svm)
plt.title('Custom SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()





