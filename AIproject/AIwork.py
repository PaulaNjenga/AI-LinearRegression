import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Load the dataset
file_name = './Nairobi Office Price Ex.csv'
df = pd.read_csv(file_name)


# Assuming the dataset has columns 'office_size' and 'office_price'
x = df['SIZE'].values  # Office sizes
y = df['PRICE'].values  # Office prices


def mean_squared_error(y_true, y_pred):
    """Compute the Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, m, c, learning_rate, epochs):
    """Perform gradient descent to learn m and c."""
    n = len(y)

    for epoch in range(epochs):
        # Predictions
        y_pred = m * x + c

        # Calculate the gradients
        dm = (-2 / n) * np.dot(x, (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)

        # Update the parameters
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Calculate and print the error
        error = mean_squared_error(y, y_pred)
        print(f'Epoch {epoch + 1}/{epochs}, MSE: {error:.2f}')

    return m, c


# Initial values
m = np.random.rand()  # Random initial slope
c = np.random.rand()  # Random initial intercept
learning_rate = 0.0001
epochs = 10

# Train the model
m, c = gradient_descent(x, y, m, c, learning_rate, epochs)

# Plotting the results
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of best fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs Price')
plt.legend()
plt.show()

# Predicting the office price for 100 sq. ft.
office_size = 130
predicted_price = m * office_size + c
print(f'Predicted office price for {office_size} sq. ft.: ${predicted_price:.2f}')