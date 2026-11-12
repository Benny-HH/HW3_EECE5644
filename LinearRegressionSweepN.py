import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from data_generator import postfix
from Lift import lift, liftDataset

# Number of samples
N = 1000

# Noise variance 
sigma = 0.01

# Feature dimension
d = 40

psfx = postfix(N,d,sigma) 
      
X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

X = liftDataset(X)

print("Original dataset has 5 features.")
print("Lifted dataset has %d features." % X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

fractions = np.arange(0.1, 1.1, 0.1)

num_samples = []
train_errors = []
test_errors = []

indices = np.random.permutation(len(X_train))

for fr in fractions:
    n_samples = int(fr * len(X_train))

    selected_indices = indices[:n_samples]

    X_subset = X_train[selected_indices]
    y_subset = y_train[selected_indices]

    model = LinearRegression()
    model.fit(X_subset, y_subset)

    rmse_train = rmse(y_subset, model.predict(X_subset))
    rmse_test = rmse(y_test, model.predict(X_test))

    num_samples.append(n_samples)
    train_errors.append(rmse_train)
    test_errors.append(rmse_test)

    print(
        "Training samples: %d, Train RMSE: %f, Test RMSE: %f"
        % (n_samples, rmse_train, rmse_test)
    )

plt.plot(num_samples, train_errors, marker="o", label="Training RMSE")
plt.plot(num_samples, test_errors, marker="o", label="Test RMSE")

plt.xlabel("Number of Training Samples")
plt.ylabel("RMSE")
plt.title("Training and Test RMSE vs. Number of Training Samples (Lifted, 40 Features)")
plt.legend()
plt.grid()

plt.show()

print("\nFinal model parameters:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)