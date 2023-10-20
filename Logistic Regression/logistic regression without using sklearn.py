import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = (4 + 3 * X + np.random.rand(100, 1)) > 5  # Binary classification problem

# Define the number of iterations and learning rate
num_iterations = 1000
learning_rate = 0.01

# Initialize the parameters (coefficients for logistic regression)
slope = np.random.randn()
intercept = np.random.randn()

# Lists to store the cost values for plotting
cost_history = []

# Perform gradient descent
m = len(X)
for iteration in range(num_iterations):
    # Calculate the logistic function (sigmoid)
    z = X.dot(slope) + intercept
    h = 1 / (1 + np.exp(-z))

    # Calculate the log loss (binary cross-entropy)
    loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    cost_history.append(loss)  # Append the cost for this iteration

    # Calculate the gradients
    gradient_slope = (1/m) * X.T.dot(h - y)
    gradient_intercept = (1/m) * np.sum(h - y)

    # Update the parameters
    slope -= learning_rate * gradient_slope
    intercept -= learning_rate * gradient_intercept

# The final values of slope and intercept are the coefficients of the logistic regression model
print("Intercept:", intercept)
print("Slope:", slope)

# Calculate and print the final cost
final_cost = cost_history[-1]
print("Final Cost:", final_cost)

# Plot the cost function graph
plt.plot(range(num_iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Graph")
plt.show()
