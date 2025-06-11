import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------
# Step 1: Prepare the Dataset
# ------------------------------
# Creating a sample dataset with two features:
# 'total_response' - Number of responses received
# 'transactions' - Number of customer transactions

data = {
    "total_response": [600, 700, 900],  # Dependent variable (target output)
    "transactions": [1000, 1200, 1500]   # Independent variable (input)
}

# Convert dictionary to a Pandas DataFrame for structured data processing
df = pd.DataFrame(data)

# ------------------------------
# Step 2: Define Features & Target Variable
# ------------------------------
# 'X' represents the independent variables (features) used for prediction.
# 'y' is the target variable, which we want to predict (transactions).

X = df[["transactions"]]  # Feature: Total customer responses
y = df["total_response"]       # Target: Corresponding transactions

# ------------------------------
# Step 3: Split Data into Training & Testing Sets
# ------------------------------
# We divide the dataset into training (80%) and testing (20%) subsets.
# This ensures we can evaluate how well the model performs on unseen data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 4: Initialize & Train the Model
# ------------------------------
# We use Linear Regression to find a relationship between total responses and transactions.

model = LinearRegression()
model.fit(X_train, y_train)  # Train the model using the training data

# ------------------------------
# Step 5: Make Predictions
# ------------------------------
# The model now predicts the number of transactions based on responses in the test set.

predictions = model.predict(X_test)

# ------------------------------
# Step 6: Evaluate Model Accuracy
# ------------------------------
# The Mean Squared Error (MSE) is calculated to measure prediction accuracy.
# Lower MSE indicates better model performance.

mse = mean_squared_error(y_test, predictions)

# ------------------------------
# Step 7: Cross-Validation for Model Validation
# ------------------------------
# We use cross-validation to verify the model’s performance across multiple data splits.
# This ensures the model generalizes well to new data.

# cv_scores = cross_val_score(model, X, y, cv=3)  # Using 3-fold cross-validation
# mean_accuracy = np.mean(cv_scores)  # Compute the average accuracy score

# ------------------------------
# Step 8: Display Final Results
# ------------------------------
# Print predicted responses, model error, and validation scores.

print(f"Predicted Number of Response Next Quarter: {int(predictions[0])}")
print(f"Mean Squared Error: {mse:.2f}")  # Display the model error
# print(f"Cross-validation scores: {cv_scores}")  # Display accuracy from cross-validation
# print(f"Mean Accuracy Score: {mean_accuracy:.2f}")  # Show the model's general accuracy