import numpy as np

class GaussianProcessRegressor:
    def __init__(self, kernel, noise_variance=1e-4):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X = None
        self.y = None
        self.K = None
        self.K_inv = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.K = self.kernel(X, X) + self.noise_variance * np.eye(len(X))
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, X_pred):
        K_pred = self.kernel(X_pred, self.X)
        y_pred_mean = K_pred.dot(self.K_inv).dot(self.y)
        K_pred_pred = self.kernel(X_pred, X_pred)
        y_pred_cov = K_pred_pred - K_pred.dot(self.K_inv).dot(K_pred.T)
        return y_pred_mean, y_pred_cov

# Example usage
# Define the kernel function (Squared Exponential kernel)
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sq_dist)

# Generate toy dataset
X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
y_train = np.sin(X_train)

# Create and fit Gaussian Process Regressor
gp_regressor = GaussianProcessRegressor(kernel)
gp_regressor.fit(X_train, y_train)

# Generate test points
X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)

# Predict using Gaussian Process Regressor
y_pred_mean, y_pred_cov = gp_regressor.predict(X_test)

# Print predictions
print("Predicted Mean:")
print(y_pred_mean)
print("Predicted Covariance:")
print(y_pred_cov)
