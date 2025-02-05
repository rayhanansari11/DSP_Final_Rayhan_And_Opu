import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from preprocessing import build_preprocessor

# Load dataset
data = pd.read_csv("data/insurance.csv")

# Split data into features (X) and target (y)
X = data.drop(columns=["charges"])
y = data["charges"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build preprocessing pipeline
preprocessor = build_preprocessor()

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_preprocessed, y_train)

# Evaluate the model
X_test_preprocessed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_preprocessed)

print("Model Performance:")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save the model and preprocessor
joblib.dump(model, "models/final_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
