import xgboost
import joblib

print("XGBoost Version:", xgboost.__version__)

model = joblib.load("insurance_fraud_model.pkl")

print(type(model))
print("Model Loaded Successfully")