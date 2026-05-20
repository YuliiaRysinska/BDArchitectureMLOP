import os
import mlflow
from pandas import read_csv
from joblib import dump

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5555")

# Get GitHub workspace path
workspace = os.getenv('GITHUB_WORKSPACE')

# Define ModelCleaning directory
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')

# Define path to cleaned CSV file
csv_file_path = os.path.join(model_cleaning_dir, 'cleaned_data.csv')

# Check if file exists
if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found at: {csv_file_path}")

# Read cleaned dataset
df = read_csv(csv_file_path)

# Show first rows
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
# Start MLflow Run
with mlflow.start_run():
    # Train model
    mind = LinearRegression()
    mind.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Log Metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # Log Model & Resiter
    result = mlflow.sklearn.log_model(sk_model=mind, artifact_path="model")

    mlflow.register_model(
        model_url=result.model_url,
        name="my_linear_model"
        )
    print(f"Model logged by R2: {r2}")

with open("Food Model.pkl", "wb") as file:
    pickle.dump(mind, file)


print("Model trained and saved successfully!")