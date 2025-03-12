import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Sample 2x2 feature matrix (4 samples, 2 features)
X = np.array([[4, 2],
              [2, 4],
              [2, 3],
              [3, 6]])

# Class labels
y = np.array([0, 0, 1, 1])

# Fit LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Transform the features (projected values)
X_lda = lda.transform(X)

print("Projected data using LDA (sklearn):")
print(X_lda)
