from xgboost import XGBClassifier

model = XGBClassifier()

model.load_model("insurance_fraud_model.json")

print("Model Loaded Successfully")