import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# Generate some example data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# Define the number of iterations and learning rate
num_iterations = 1000
learning_rate = 0.01

# Initialize the parameters (slope and intercept)
slope = np.random.randn()
intercept = np.random.randn()

# Initialize a list to store the cost history
cost_history = []

# Perform gradient descent
m = len(X)
for iteration in range(num_iterations):
    # Calculate the predictions
    y_pred = slope * X + intercept

    # Calculate the error
    error = y_pred - y

    # Calculate the cost (mean squared error)
    cost = (1 / (2 * m)) * np.sum(error**2)
    cost_history.append(cost)

    # Calculate the gradients
    gradient_slope = (1/m) * np.sum(error * X)
    gradient_intercept = (1/m) * np.sum(error)

    # Update the parameters
    slope -= learning_rate * gradient_slope
    intercept -= learning_rate * gradient_intercept

# The final values of slope and intercept are the coefficients of the linear regression model
final_slope = slope
final_intercept = intercept
print("Intercept:", final_intercept)
print("Slope:", final_slope)

# Plot the cost over iterations
plt.plot(range(num_iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.show()
