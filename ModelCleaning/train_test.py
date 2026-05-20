import os
import pickle
import mlflow

from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# MLflow tracking
mlflow.set_tracking_uri("http://localhost:5555")

# Workspace path
workspace = os.getenv("GITHUB_WORKSPACE", os.getcwd())

# Dataset path
model_cleaning_dir = os.path.join(workspace, "ModelCleaning")
csv_file_path = os.path.join(model_cleaning_dir, "cleaned_data.csv")

# Check file
if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found at: {csv_file_path}")

# Load dataset
df = read_csv(csv_file_path)

print(df.head())

# Features and target
X = df[["calories", "rating"]]
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Train model
    mind = LinearRegression()
    mind.fit(X_train, y_train)

    # Predict
    predictions = mind.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Log metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # Log model
    result = mlflow.sklearn.log_model(
        sk_model=mind,
        artifact_path="model"
    )

    # Register model
    mlflow.register_model(
        model_uri=result.model_uri,
        name="food_model"
    )

    print(f"Model logged with R2: {r2}")

# Save model locally
with open("Food_Model.pkl", "wb") as file:
    pickle.dump(mind, file)

print("Model trained and saved successfully!")