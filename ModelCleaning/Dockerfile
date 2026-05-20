FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn joblib mlflow

CMD ["python", "ModelCleaning/train_test.py"]
