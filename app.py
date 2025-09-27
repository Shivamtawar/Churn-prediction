# Flask App with Traditional Structure - app.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variable to store model
model_package = None

def detect_data_type(df):
    """Smart detection of raw vs normalized data"""
    raw_indicators = 0
    total_checks = 0
    
    checks = [
        ('Tenure', 1), ('SatisfactionScore', 1), ('OrderCount', 1),
        ('WarehouseToHome', 1), ('HourSpendOnApp', 1)
    ]
    
    for col, threshold in checks:
        if col in df.columns:
            total_checks += 1
            if df[col].iloc[0] > threshold:
                raw_indicators += 1
    
    if total_checks > 0:
        raw_ratio = raw_indicators / total_checks
        data_type = 'raw' if raw_ratio >= 0.5 else 'normalized'
        logger.info(f"Data type detection: {raw_indicators}/{total_checks} raw indicators = {data_type}")
        return data_type
    
    return 'raw'

def normalize_raw_data(df):
    """Normalize raw data to 0-1 range"""
    df = df.copy()
    
    normalization_ranges = {
        'Tenure': (0, 61), 'WarehouseToHome': (5, 127), 'HourSpendOnApp': (0, 5),
        'SatisfactionScore': (1, 5), 'OrderCount': (1, 16), 'DaySinceLastOrder': (0, 46),
        'NumberOfDeviceRegistered': (1, 6), 'NumberOfAddress': (1, 9),
        'OrderAmountHikeFromlastYear': (11, 26), 'CouponUsed': (0, 16),
        'CashbackAmount': (0, 324)
    }
    
    for col, (min_val, max_val) in normalization_ranges.items():
        if col in df.columns:
            df[col] = np.clip((df[col] - min_val) / (max_val - min_val), 0, 1)
    
    return df

def create_advanced_features(df):
    """Create advanced features with smart data type handling"""
    df = df.copy()
    
    data_type = detect_data_type(df)
    
    if data_type == 'raw':
        df = normalize_raw_data(df)
    
    # Create normalized column names
    df['Tenure_norm'] = df.get('Tenure', 0)
    df['Satisfaction_norm'] = df.get('SatisfactionScore', 0)
    df['Orders_norm'] = df.get('OrderCount', 0)
    df['DaysSince_norm'] = df.get('DaySinceLastOrder', 0)
    df['AppHours_norm'] = df.get('HourSpendOnApp', 0)
    df['Warehouse_norm'] = df.get('WarehouseToHome', 0)
    
    # Risk indicators
    risk_features = [
        ('HighRiskTenure', df['Tenure_norm'] < 0.15),
        ('LowSatisfaction', df['Satisfaction_norm'] < 0.5),
        ('VeryLowSatisfaction', df['Satisfaction_norm'] < 0.25),
        ('LowEngagement', df['AppHours_norm'] < 0.4),
        ('VeryLowEngagement', df['AppHours_norm'] < 0.2),
        ('RecentOrderGap', df['DaysSince_norm'] > 0.5),
        ('VeryLongGap', df['DaysSince_norm'] > 0.7),
        ('LowOrderFreq', df['Orders_norm'] < 0.3),
        ('VeryLowOrders', df['Orders_norm'] < 0.15),
        ('HighWarehouseDist', df['Warehouse_norm'] > 0.6),
        ('ComplainFlag', df.get('Complain', 0) > 0)
    ]
    
    for feature_name, condition in risk_features:
        df[feature_name] = np.where(condition, 1, 0)
    
    # Interaction features
    df['SatisfactionEngagement'] = df['Satisfaction_norm'] * df['AppHours_norm']
    df['TenureOrderRatio'] = np.where(df['Orders_norm'] > 0, 
                                    df['Tenure_norm'] / (df['Orders_norm'] + 0.001), 0)
    df['TenureOrderRatio'] = np.clip(df['TenureOrderRatio'], 0, 10)
    df['SatisfactionTenure'] = df['Satisfaction_norm'] * df['Tenure_norm']
    df['EngagementOrderRatio'] = df['AppHours_norm'] * df['Orders_norm']
    
    # Composite risk scores
    df['BasicRiskScore'] = (df['HighRiskTenure'] + df['LowSatisfaction'] + 
                           df['LowEngagement'] + df['RecentOrderGap'] + 
                           df['LowOrderFreq'] + df['ComplainFlag'])
    
    df['AdvancedRiskScore'] = (df['VeryLowSatisfaction'] * 2 + df['VeryLowEngagement'] * 2 + 
                              df['VeryLongGap'] * 2 + df['VeryLowOrders'] * 2 + 
                              df['HighRiskTenure'] + df['ComplainFlag'] * 3)
    
    # Behavioral patterns
    df['DisengagedPattern'] = np.where((df['LowEngagement'] == 1) & (df['LowSatisfaction'] == 1), 1, 0)
    df['NewCustomerRisk'] = np.where((df['HighRiskTenure'] == 1) & (df['LowOrderFreq'] == 1), 1, 0)
    df['ComplainerRisk'] = np.where((df['ComplainFlag'] == 1) & (df['LowSatisfaction'] == 1), 1, 0)
    
    return df.fillna(0).replace([np.inf, -np.inf], 0)

def load_model():
    """Load the trained model"""
    global model_package
    model_files = ['improved_churn_model_v2.pkl', 'improved_churn_model.pkl']
    
    for model_file in model_files:
        try:
            model_package = joblib.load(model_file)
            logger.info(f"Model loaded successfully: {model_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load {model_file}: {e}")
    
    logger.error("Failed to load any model file")
    return False

def enhanced_predict_churn(input_data):
    """Enhanced prediction with smart data handling"""
    try:
        model = model_package['model']
        
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data.copy()
        
        # Fill missing required columns
        required_defaults = {
            'Tenure': 12, 'WarehouseToHome': 15, 'HourSpendOnApp': 2.0,
            'SatisfactionScore': 3, 'OrderCount': 5, 'DaySinceLastOrder': 10,
            'Complain': 0, 'NumberOfDeviceRegistered': 2, 'NumberOfAddress': 2,
            'OrderAmountHikeFromlastYear': 15, 'CouponUsed': 2, 'CashbackAmount': 100,
            'PreferredLoginDevice': 1, 'CityTier': 2, 'PreferredPaymentMode': 1,
            'Gender': 0, 'MaritalStatus': 0, 'PreferedOrderCat': 1
        }
        
        for col, default in required_defaults.items():
            if col not in input_df.columns:
                input_df[col] = default
        
        # Apply feature engineering
        input_enhanced = create_advanced_features(input_df)
        
        # Ensure all model features are present
        model_features = model_package.get('feature_columns', [])
        for col in model_features:
            if col not in input_enhanced.columns:
                input_enhanced[col] = 0
        
        input_final = input_enhanced[model_features]
        
        # Make prediction
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]
        churn_prob = float(probability[1])
        
        # Risk classification
        risk_thresholds = model_package.get('risk_thresholds', {
            'high_risk_threshold': 0.6,
            'medium_risk_threshold': 0.35
        })
        
        high_threshold = risk_thresholds.get('high_risk_threshold', 0.6)
        medium_threshold = risk_thresholds.get('medium_risk_threshold', 0.35)
        
        if churn_prob > high_threshold:
            risk_level = "High"
            action_priority = "IMMEDIATE"
            recommendation = "Immediate intervention required! Contact customer within 24 hours with retention offers."
        elif churn_prob > medium_threshold:
            risk_level = "Medium"
            action_priority = "WITHIN_WEEK"
            recommendation = "Proactive engagement needed within 3-5 days. Send personalized offers or surveys."
        else:
            risk_level = "Low"
            action_priority = "MONITOR"
            recommendation = "Customer appears stable. Continue standard service and monitor quarterly."
        
        # Generate insights
        insights = []
        basic_risk = input_enhanced.get('BasicRiskScore', pd.Series([0])).iloc[0]
        advanced_risk = input_enhanced.get('AdvancedRiskScore', pd.Series([0])).iloc[0]
        
        if basic_risk >= 3:
            insights.append("Multiple risk factors detected")
        if input_enhanced.get('ComplainerRisk', pd.Series([0])).iloc[0] > 0:
            insights.append("Complainer with low satisfaction")
        if input_enhanced.get('NewCustomerRisk', pd.Series([0])).iloc[0] > 0:
            insights.append("New customer with low engagement")
        if input_enhanced.get('DisengagedPattern', pd.Series([0])).iloc[0] > 0:
            insights.append("Shows disengaged behavior pattern")
        if input_enhanced.get('VeryLowSatisfaction', pd.Series([0])).iloc[0] > 0:
            insights.append("Very low satisfaction score")
        if input_enhanced.get('VeryLongGap', pd.Series([0])).iloc[0] > 0:
            insights.append("Very long gap since last order")
        
        confidence_score = max(probability)
        confidence = 'High' if confidence_score > 0.8 else 'Medium' if confidence_score > 0.6 else 'Low'
        
        return {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Will Churn' if prediction == 1 else 'Will Retain',
            'churn_probability': round(churn_prob, 4),
            'retention_probability': round(float(probability[0]), 4),
            'risk_level': risk_level,
            'action_priority': action_priority,
            'recommendation': recommendation,
            'insights': insights,
            'detailed_analysis': {
                'basic_risk_score': int(basic_risk),
                'advanced_risk_score': int(advanced_risk),
                'satisfaction_normalized': round(input_enhanced.get('Satisfaction_norm', pd.Series([0])).iloc[0], 3),
                'engagement_normalized': round(input_enhanced.get('AppHours_norm', pd.Series([0])).iloc[0], 3),
                'tenure_normalized': round(input_enhanced.get('Tenure_norm', pd.Series([0])).iloc[0], 3),
                'data_type_detected': detect_data_type(input_df)
            },
            'confidence': confidence,
            'confidence_score': round(confidence_score, 4),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Load model on startup
load_model()

# Routes
@app.route('/', methods=['GET'])
def index():
    """Main page with the prediction interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_package else 'unhealthy',
        'model_loaded': model_package is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '3.0',
        'features': ['smart_data_detection', 'batch_processing', 'comprehensive_analysis']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single customer prediction"""
    try:
        if not model_package:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        result = enhanced_predict_churn(data)
        return jsonify(result), 200 if result['success'] else 500
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple customers"""
    try:
        if not model_package:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        if not data or 'customers' not in data:
            return jsonify({'success': False, 'error': 'Expected JSON with "customers" array'}), 400
        
        customers = data['customers']
        results = []
        risk_summary = {'High': 0, 'Medium': 0, 'Low': 0}
        
        for i, customer_data in enumerate(customers):
            prediction = enhanced_predict_churn(customer_data)
            prediction['customer_id'] = i
            results.append(prediction)
            
            if prediction['success']:
                risk_summary[prediction['risk_level']] += 1
        
        return jsonify({
            'success': True,
            'total_customers': len(customers),
            'risk_summary': risk_summary,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/file_predict', methods=['POST'])
def file_predict():
    """Process CSV file for batch predictions"""
    try:
        if not model_package:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 503
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files supported'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        results = []
        risk_summary = {'High': 0, 'Medium': 0, 'Low': 0}
        
        for i, row in df.iterrows():
            customer_data = row.to_dict()
            prediction = enhanced_predict_churn(customer_data)
            prediction['row_id'] = i
            results.append(prediction)
            
            if prediction['success']:
                risk_summary[prediction['risk_level']] += 1
        
        return jsonify({
            'success': True,
            'total_rows': len(df),
            'risk_summary': risk_summary,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information and statistics"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503
    
    return jsonify({
        'success': True,
        'model_info': model_package['model_info'],
        'total_features': len(model_package.get('feature_columns', [])),
        'algorithm': model_package.get('model_name', 'Unknown'),
        'risk_thresholds': model_package.get('risk_thresholds', {}),
        'feature_importance': model_package.get('feature_importance', [])[:15],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test_data', methods=['GET'])
def test_data():
    """Get comprehensive test datasets"""
    return jsonify({
        'success': True,
        'test_datasets': {
            'high_risk_customers': [
                {
                    "name": "New Unsatisfied Customer",
                    "data": {
                        "Tenure": 2, "WarehouseToHome": 25, "HourSpendOnApp": 0.5,
                        "SatisfactionScore": 2, "OrderCount": 1, "DaySinceLastOrder": 30,
                        "Complain": 1, "NumberOfDeviceRegistered": 1, "CityTier": 3,
                        "CashbackAmount": 50, "PreferredPaymentMode": 2
                    }
                },
                {
                    "name": "Disengaged Veteran",
                    "data": {
                        "Tenure": 8, "WarehouseToHome": 20, "HourSpendOnApp": 0.8,
                        "SatisfactionScore": 2, "OrderCount": 2, "DaySinceLastOrder": 25,
                        "Complain": 0, "NumberOfDeviceRegistered": 2, "CityTier": 2,
                        "CashbackAmount": 80, "PreferredPaymentMode": 1
                    }
                }
            ],
            'medium_risk_customers': [
                {
                    "name": "Moderate Engagement",
                    "data": {
                        "Tenure": 12, "WarehouseToHome": 15, "HourSpendOnApp": 2.0,
                        "SatisfactionScore": 3, "OrderCount": 6, "DaySinceLastOrder": 8,
                        "Complain": 0, "NumberOfDeviceRegistered": 3, "CityTier": 2,
                        "CashbackAmount": 120, "PreferredPaymentMode": 1
                    }
                },
                {
                    "name": "Declining Activity",
                    "data": {
                        "Tenure": 18, "WarehouseToHome": 12, "HourSpendOnApp": 1.5,
                        "SatisfactionScore": 3, "OrderCount": 4, "DaySinceLastOrder": 15,
                        "Complain": 0, "NumberOfDeviceRegistered": 2, "CityTier": 1,
                        "CashbackAmount": 150, "PreferredPaymentMode": 0
                    }
                }
            ],
            'low_risk_customers': [
                {
                    "name": "Loyal Customer",
                    "data": {
                        "Tenure": 36, "WarehouseToHome": 8, "HourSpendOnApp": 4.2,
                        "SatisfactionScore": 5, "OrderCount": 14, "DaySinceLastOrder": 2,
                        "Complain": 0, "NumberOfDeviceRegistered": 4, "CityTier": 1,
                        "CashbackAmount": 280, "PreferredPaymentMode": 1
                    }
                },
                {
                    "name": "Active Buyer",
                    "data": {
                        "Tenure": 24, "WarehouseToHome": 10, "HourSpendOnApp": 3.5,
                        "SatisfactionScore": 4, "OrderCount": 10, "DaySinceLastOrder": 3,
                        "Complain": 0, "NumberOfDeviceRegistered": 3, "CityTier": 2,
                        "CashbackAmount": 200, "PreferredPaymentMode": 2
                    }
                }
            ]
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)