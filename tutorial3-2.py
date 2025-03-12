import numpy as np

# Feature matrix
X = np.array([[4, 2],
              [2, 4],
              [2, 3],
              [3, 6]])

# Class labels
y = np.array([0, 0, 1, 1])

# Split into two classes
X0 = X[y == 0]
X1 = X[y == 1]

# Mean vectors
mean0 = np.mean(X0, axis=0)
mean1 = np.mean(X1, axis=0)

# Within-class scatter matrix
S_W = np.cov(X0.T) + np.cov(X1.T)

# Compute LDA direction vector (w)
S_W_inv = np.linalg.inv(S_W)
w = np.dot(S_W_inv, (mean0 - mean1))

print("LDA direction vector (w):", w)

# Project original data on this direction
projected = np.dot(X, w)

print("\nProjected data using LDA (manual method):")
print(projected)
