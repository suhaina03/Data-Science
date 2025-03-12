import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset defined directly in code
data = {
    "Soil_pH": [6.5, 7.0, 6.2, 6.8, 6.3, 6.7, 7.1, 6.0, 6.4, 7.2,
                6.9, 6.6, 6.1, 7.0, 6.5, 6.2, 6.8, 7.3, 6.3, 6.9, 6.7],
    "Rainfall_mm": [200, 250, 180, 220, 190, 240, 260, 170, 210, 275,
                    230, 195, 185, 255, 205, 175, 225, 280, 190, 235, 245],
    "Temperature_C": [25, 26, 24, 27, 23, 28, 29, 22, 26, 30,
                      27, 25, 24, 28, 26, 23, 27, 31, 24, 26, 28],
    "Fertilizer_kg": [120, 150, 110, 140, 115, 155, 160, 105, 130, 170,
                      145, 125, 115, 155, 135, 110, 140, 165, 120, 150, 155],
    "Pesticide_L": [2.0, 2.5, 1.8, 2.3, 1.9, 2.6, 2.7, 1.7, 2.1, 2.8,
                    2.4, 2.0, 1.9, 2.5, 2.2, 1.8, 2.3, 2.9, 1.9, 2.4, 2.6],
    "Yield_High": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
                   1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df.drop("Yield_High", axis=1)
y = df["Yield_High"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
print("📈 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Predict for unknown values
new_data = pd.DataFrame({
    "Soil_pH": [6.4, 7.0, 6.6],
    "Rainfall_mm": [215, 250, 220],
    "Temperature_C": [26, 27, 25],
    "Fertilizer_kg": [145, 155, 140],
    "Pesticide_L": [2.1, 2.6, 2.2]
})

new_pred = model.predict(new_data)
print("\n🆕 Predicted Yield (1=High, 0=Low) for new data:\n", new_pred)
