import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv("Advertising.csv")

# Drop unnecessary index column if present
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Define predictors and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Add a constant term for intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression
print(model.summary())

# Extract and print R²
r_squared = model.rsquared
print(f"\nR-squared (R²): {r_squared:.3f}")

# Extract and print Residual Standard Error
residuals = model.resid
rss = sum(residuals**2)
rse = np.sqrt(rss / (len(X) - len(X.columns)))  # df = n - k
print(f"Residual Standard Error (RSE): {rse:.3f}")

# F-statistic
f_stat = model.fvalue
print(f"F-statistic: {f_stat:.2f}")

# Plot residuals
plt.figure(figsize=(8,5))
plt.scatter(model.fittedvalues, residuals, alpha=0.7, color='tomato')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
