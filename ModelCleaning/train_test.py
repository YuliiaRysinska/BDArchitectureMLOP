import os

from pandas import read_csv
from joblib import dump

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
X = df["Age"].values.reshape(-1, 1)
y = df["Salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
mind = LinearRegression()

mind.fit(X_train, y_train)

# Save trained model
dump(mind, "AgeSalaryModel.pkl")

print("Model trained and saved successfully!")