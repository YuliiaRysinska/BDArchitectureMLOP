import os
import pandas as pd
import numpy as np


# Load CSV file
df = pd.read_csv('data.csv')

# Show original data
print("Original Data:")
print(df.head())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Strip + lowercase ONLY string columns safely
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip().str.lower()

# Convert Age and Salary to numeric (forces bad values → NaN)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Replace infinities just in case
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# NOW drop missing values
df.dropna(inplace=True)

# Convert string columns to lowercase
df = df.map(lambda x: x.lower() if isinstance(x, str) else x)


# Drop duplicate rows
df.drop_duplicates(inplace=True)

#workspace = os.getenv('GITHUB_WORKSPACE')

# Define the directory where your Python script is located (ModelCleaning)
#model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')

model_cleaning_dir = ('/opt/mlflow/Week7MLModel/ModelCleaning')


# Define the full path for the output file
output_path = os.path.join(model_cleaning_dir, 'cleaned_data.csv')

# Create the directory if it doesn't exist
os.makedirs(model_cleaning_dir, exist_ok=True)

# Save cleaned data
df.to_csv(output_path, index=False)

print(output_path)
# Show original data
print("Cleaned Data:")
print(df.head())

print("\nCleaned data saved to 'cleaned_data.csv'")
