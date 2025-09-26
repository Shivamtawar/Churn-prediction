import joblib
import pandas as pd

def predict_churn(model_package, input_data):
    """
    Function to make predictions - use this in your Flask API
    """
    model = model_package['model']
    feature_columns = model_package['feature_columns']
    
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_columns]
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    return {
        'prediction': int(prediction),
        'churn_probability': float(probability[1]),
        'retention_probability': float(probability[0]),
        'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low'
    }

# Load model
def load_model():
    return joblib.load('churn_prediction_model.pkl')
