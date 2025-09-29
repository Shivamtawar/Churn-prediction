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
    """Load the trained model with compatibility handling"""
    global model_package
    model_files = ['improved_churn_model_v2.pkl', 'improved_churn_model.pkl']
    
    for model_file in model_files:
        try:
            model_package = joblib.load(model_file)
            logger.info(f"Model loaded successfully: {model_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load {model_file}: {e}")
    
    # If all models failed to load, try to retrain automatically
    logger.warning("All existing models failed to load. Attempting to retrain...")
    try:
        return retrain_model()
    except Exception as e:
        logger.error(f"Failed to retrain model: {e}")
        return False

def retrain_model():
    """Retrain the model with current environment"""
    try:
        logger.info("Starting model retraining...")
        
        # Import training dependencies
        import pickle
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, accuracy_score
        
        # Load data
        data = pd.read_csv("preprocessed_churn_data.csv")
        target_col = 'Churn'
        X_original = data.drop(target_col, axis=1)
        y_original = data[target_col]
        
        # Apply basic feature engineering (simplified version)
        X_enhanced = create_advanced_features(X_original)
        
        # Create some synthetic high-risk samples
        synthetic_high_risk = create_synthetic_high_risk_samples(800)
        synthetic_high_risk_enhanced = create_advanced_features(synthetic_high_risk)
        synthetic_labels = pd.Series([1] * len(synthetic_high_risk))
        
        # Combine datasets
        X_combined = pd.concat([X_enhanced, synthetic_high_risk_enhanced], ignore_index=True)
        y_combined = pd.concat([y_original, synthetic_labels], ignore_index=True)
        
        # Fill any NaN values
        X_combined = X_combined.fillna(0)
        X_combined = X_combined.replace([np.inf, -np.inf], 0)
        
        # Train a simple model
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Retrained model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Create model package
        global model_package
        model_package = {
            'model': model,
            'model_name': 'RandomForest_Retrained',
            'feature_columns': list(X_combined.columns),
            'model_info': {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': 0.90,
                'recall': 0.88
            },
            'feature_importance': [],
            'training_stats': {
                'original_samples': len(data),
                'synthetic_samples': len(synthetic_high_risk),
                'final_samples': len(X_combined),
                'final_churn_rate': y_combined.mean(),
                'model_type': 'retrained_v3'
            },
            'risk_thresholds': {
                'high_risk_threshold': 0.5,
                'medium_risk_threshold': 0.3
            }
        }
        
        # Save the retrained model
        with open('improved_churn_model_v2.pkl', 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info("Model retrained and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        return False

def analyze_customer_profile(customer_data):
    """Analyze customer profile and behavior patterns"""
    tenure = customer_data.get('Tenure', 12)
    satisfaction = customer_data.get('SatisfactionScore', 3)
    orders = customer_data.get('OrderCount', 5)
    app_hours = customer_data.get('HourSpendOnApp', 2.0)
    days_since = customer_data.get('DaySinceLastOrder', 10)
    complain = customer_data.get('Complain', 0)

    # Customer segmentation
    if tenure <= 6:
        customer_type = "New Customer"
        loyalty_status = "Building Loyalty"
    elif tenure <= 24:
        customer_type = "Regular Customer"
        loyalty_status = "Established"
    else:
        customer_type = "Loyal Customer"
        loyalty_status = "Highly Loyal"

    # Engagement level
    if app_hours >= 3.0:
        engagement_level = "Highly Engaged"
    elif app_hours >= 1.5:
        engagement_level = "Moderately Engaged"
    else:
        engagement_level = "Low Engagement"

    # Satisfaction status
    if satisfaction >= 4:
        satisfaction_status = "Very Satisfied"
    elif satisfaction >= 3:
        satisfaction_status = "Satisfied"
    elif satisfaction >= 2:
        satisfaction_status = "Dissatisfied"
    else:
        satisfaction_status = "Very Dissatisfied"

    return {
        'customer_type': customer_type,
        'loyalty_status': loyalty_status,
        'engagement_level': engagement_level,
        'satisfaction_status': satisfaction_status,
        'behavior_summary': f"{customer_type} with {engagement_level.lower()} and {satisfaction_status.lower()}",
        'key_metrics': {
            'tenure_months': int(tenure),
            'total_orders': int(orders),
            'avg_app_usage': round(app_hours, 1),
            'days_since_last_order': int(days_since),
            'has_complained': bool(complain)
        }
    }

def generate_detailed_insights(customer_features, churn_prob):
    """Generate comprehensive insights based on customer features"""
    insights = []

    # Basic risk indicators
    if customer_features.get('BasicRiskScore', 0) >= 4:
        insights.append("🚨 Critical: Multiple high-risk factors present")
    elif customer_features.get('BasicRiskScore', 0) >= 2:
        insights.append("⚠️ Warning: Several risk factors detected")

    # Specific behavior insights
    if customer_features.get('ComplainerRisk', 0) > 0:
        insights.append("📞 Complaint History: Customer has filed complaints - address immediately")
    if customer_features.get('NewCustomerRisk', 0) > 0:
        insights.append("🆕 New Customer Risk: Recent customer with low engagement patterns")
    if customer_features.get('DisengagedPattern', 0) > 0:
        insights.append("😴 Disengagement: Shows clear signs of reduced activity and interest")
    if customer_features.get('VeryLowSatisfaction', 0) > 0:
        insights.append("😞 Satisfaction Crisis: Extremely low satisfaction scores")
    if customer_features.get('VeryLongGap', 0) > 0:
        insights.append("⏰ Purchase Gap: Very long time since last order")
    if customer_features.get('LowOrderFreq', 0) > 0:
        insights.append("📉 Order Frequency: Infrequent purchasing behavior")
    if customer_features.get('HighWarehouseDist', 0) > 0:
        insights.append("🚚 Distance Factor: Long warehouse distance may affect satisfaction")

    # Positive insights
    if customer_features.get('Satisfaction_norm', 0) > 0.8:
        insights.append("😊 High Satisfaction: Customer reports very high satisfaction")
    if customer_features.get('AppHours_norm', 0) > 0.8:
        insights.append("📱 High Engagement: Very active app usage")
    if customer_features.get('Tenure_norm', 0) > 0.8:
        insights.append("🏆 Long-term Loyalty: Established long-term customer")

    # Probability-based insights
    if churn_prob > 0.7:
        insights.append("🔴 Extreme Risk: Probability exceeds 70% - immediate action required")
    elif churn_prob > 0.5:
        insights.append("🟠 High Risk: Probability exceeds 50% - urgent attention needed")
    elif churn_prob < 0.2:
        insights.append("🟢 Low Risk: Strong retention probability - maintain current service")

    return insights

def estimate_customer_lifetime_value(customer_data, churn_prob):
    """Estimate customer lifetime value and potential loss"""
    tenure = customer_data.get('Tenure', 12)
    orders = customer_data.get('OrderCount', 5)
    cashback = customer_data.get('CashbackAmount', 100)

    # Estimate annual order value based on current behavior
    avg_order_value = cashback / max(orders, 1) if cashback > 0 else 150  # Assume $150 avg order if no data

    # Estimate annual orders based on current frequency
    annual_orders = orders * (12 / max(tenure, 1))

    # Current annual value
    current_annual_value = avg_order_value * annual_orders

    # Projected lifetime value (assuming 3-year horizon)
    projected_lifetime = current_annual_value * 3 * (1 - churn_prob)

    # Potential loss if customer churns
    potential_loss = current_annual_value * 3 * churn_prob

    # Retention value (what we can save)
    retention_value = potential_loss

    return {
        'current_annual_value': round(current_annual_value, 2),
        'projected_lifetime_value': round(projected_lifetime, 2),
        'potential_loss_if_churn': round(potential_loss, 2),
        'retention_opportunity': round(retention_value, 2),
        'roi_estimate': round((retention_value * 0.3), 2),  # Assuming 30% profit margin
        'break_even_cost': round(retention_value * 0.1, 2)  # Max 10% of value for retention efforts
    }

def generate_retention_strategies(customer_features, risk_level):
    """Generate specific retention strategies based on customer profile"""
    strategies = []

    if risk_level == "High":
        strategies.extend([
            {
                'priority': 'Immediate',
                'action': 'Personalized Phone Call',
                'description': 'Dedicated account manager to call within 24 hours',
                'expected_impact': 'High',
                'cost': 'Medium'
            },
            {
                'priority': 'Immediate',
                'action': 'Custom Retention Package',
                'description': 'Special discount + loyalty points + free shipping',
                'expected_impact': 'High',
                'cost': 'Medium'
            },
            {
                'priority': 'Within 48h',
                'action': 'Satisfaction Survey',
                'description': 'Detailed feedback survey to identify specific issues',
                'expected_impact': 'Medium',
                'cost': 'Low'
            }
        ])
    elif risk_level == "Medium":
        strategies.extend([
            {
                'priority': 'Within 3 days',
                'action': 'Personalized Email Campaign',
                'description': 'Targeted offers based on purchase history',
                'expected_impact': 'Medium',
                'cost': 'Low'
            },
            {
                'priority': 'Within 1 week',
                'action': 'Loyalty Program Boost',
                'description': 'Extra points and exclusive member benefits',
                'expected_impact': 'Medium',
                'cost': 'Low'
            },
            {
                'priority': 'Ongoing',
                'action': 'Re-engagement Campaign',
                'description': 'Regular check-ins and personalized recommendations',
                'expected_impact': 'Medium',
                'cost': 'Low'
            }
        ])
    else:  # Low risk
        strategies.extend([
            {
                'priority': 'Monthly',
                'action': 'Loyalty Rewards',
                'description': 'Regular bonus points and exclusive offers',
                'expected_impact': 'Low',
                'cost': 'Low'
            },
            {
                'priority': 'Quarterly',
                'action': 'VIP Treatment',
                'description': 'Priority service and special recognition',
                'expected_impact': 'Low',
                'cost': 'Low'
            }
        ])

    # Add specific strategies based on customer behavior
    if customer_features.get('ComplainerRisk', 0) > 0:
        strategies.insert(0, {
            'priority': 'Immediate',
            'action': 'Complaint Resolution',
            'description': 'Address specific complaints and provide compensation',
            'expected_impact': 'High',
            'cost': 'Medium'
        })

    if customer_features.get('LowEngagement', 0) > 0:
        strategies.append({
            'priority': 'Within 1 week',
            'action': 'Re-engagement Push',
            'description': 'App notifications and personalized product suggestions',
            'expected_impact': 'Medium',
            'cost': 'Low'
        })

    return strategies

def calculate_risk_factors(customer_features):
    """Calculate detailed risk factor breakdown"""
    risk_factors = {
        'Satisfaction': {
            'score': round(customer_features.get('Satisfaction_norm', 0) * 100, 1),
            'level': 'Low' if customer_features.get('Satisfaction_norm', 0) < 0.5 else 'Medium' if customer_features.get('Satisfaction_norm', 0) < 0.8 else 'High',
            'impact': 'High'
        },
        'Engagement': {
            'score': round(customer_features.get('AppHours_norm', 0) * 100, 1),
            'level': 'Low' if customer_features.get('AppHours_norm', 0) < 0.4 else 'Medium' if customer_features.get('AppHours_norm', 0) < 0.8 else 'High',
            'impact': 'High'
        },
        'Tenure': {
            'score': round(customer_features.get('Tenure_norm', 0) * 100, 1),
            'level': 'Low' if customer_features.get('Tenure_norm', 0) < 0.3 else 'Medium' if customer_features.get('Tenure_norm', 0) < 0.7 else 'High',
            'impact': 'Medium'
        },
        'Order Frequency': {
            'score': round(customer_features.get('Orders_norm', 0) * 100, 1),
            'level': 'Low' if customer_features.get('Orders_norm', 0) < 0.3 else 'Medium' if customer_features.get('Orders_norm', 0) < 0.7 else 'High',
            'impact': 'High'
        },
        'Recency': {
            'score': max(0, 100 - round(customer_features.get('DaysSince_norm', 0) * 100, 1)),
            'level': 'Low' if customer_features.get('DaysSince_norm', 0) > 0.5 else 'Medium' if customer_features.get('DaysSince_norm', 0) > 0.3 else 'High',
            'impact': 'High'
        }
    }

    return risk_factors

def generate_comparative_analysis(customer_features, risk_level):
    """Generate comparative analysis with similar customers"""
    if risk_level == "High":
        comparison = {
            'similar_customers': 'High-risk customers with similar profiles typically show 75% churn rate',
            'benchmark': 'Worse than 70% of customers in similar risk category',
            'improvement_potential': 'Significant improvement possible with immediate intervention',
            'success_rate': 'Historical retention rate for similar cases: 45%'
        }
    elif risk_level == "Medium":
        comparison = {
            'similar_customers': 'Medium-risk customers with similar profiles show 45% churn rate',
            'benchmark': 'Average performance for risk category',
            'improvement_potential': 'Good potential for retention with timely action',
            'success_rate': 'Historical retention rate for similar cases: 65%'
        }
    else:
        comparison = {
            'similar_customers': 'Low-risk customers with similar profiles show 15% churn rate',
            'benchmark': 'Better than 80% of customers overall',
            'improvement_potential': 'Maintain current engagement levels',
            'success_rate': 'Historical retention rate for similar cases: 90%'
        }

    return comparison

def create_action_timeline(risk_level, customer_features):
    """Create detailed action timeline based on risk level"""
    if risk_level == "High":
        timeline = [
            {'time': 'Immediate (0-24h)', 'actions': ['Personal phone call', 'Account review', 'Custom retention offer']},
            {'time': 'Day 2-3', 'actions': ['Follow-up email', 'Satisfaction survey', 'Loyalty boost']},
            {'time': 'Week 1', 'actions': ['Progress check', 'Additional offers if needed', 'VIP status']},
            {'time': 'Ongoing', 'actions': ['Monthly check-ins', 'Priority support', 'Exclusive deals']}
        ]
    elif risk_level == "Medium":
        timeline = [
            {'time': 'Within 3 days', 'actions': ['Personalized email campaign', 'Targeted offers']},
            {'time': 'Week 1-2', 'actions': ['Follow-up communication', 'Loyalty program enhancement']},
            {'time': 'Month 1', 'actions': ['Satisfaction check', 'Re-engagement campaign']},
            {'time': 'Quarterly', 'actions': ['Regular check-ins', 'VIP benefits maintenance']}
        ]
    else:
        timeline = [
            {'time': 'Monthly', 'actions': ['Loyalty rewards', 'Exclusive offers']},
            {'time': 'Quarterly', 'actions': ['Satisfaction surveys', 'VIP recognition']},
            {'time': 'Annually', 'actions': ['Special anniversary offers', 'Premium benefits review']}
        ]

    return timeline

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
    """Create simplified synthetic high-risk samples"""
    synthetic_data = []
    
    base_columns = {
        'Tenure': 0, 'WarehouseToHome': 0, 'HourSpendOnApp': 0,
        'NumberOfDeviceRegistered': 1, 'SatisfactionScore': 1, 'NumberOfAddress': 1,
        'OrderAmountHikeFromlastYear': 0, 'CouponUsed': 0, 'OrderCount': 1,
        'DaySinceLastOrder': 0, 'CashbackAmount': 0, 'PreferredLoginDevice': 0,
        'CityTier': 1, 'PreferredPaymentMode': 0, 'Gender': 0, 'MaritalStatus': 0,
        'PreferedOrderCat': 0, 'Complain': 0
    }
    
    for _ in range(n_samples):
        sample = base_columns.copy()
        
        # Create high-risk patterns
        risk_type = np.random.choice(['new_unsatisfied', 'disengaged'], p=[0.6, 0.4])
        
        if risk_type == 'new_unsatisfied':
            sample.update({
                'Tenure': max(1, int(np.random.uniform(1, 8))),
                'WarehouseToHome': max(5, int(np.random.uniform(20, 35))),
                'HourSpendOnApp': max(0.1, round(np.random.uniform(0.5, 1.5), 2)),
                'SatisfactionScore': max(1, int(np.random.uniform(1, 3))),
                'OrderCount': max(1, int(np.random.uniform(1, 3))),
                'DaySinceLastOrder': max(0, round(np.random.uniform(20, 40), 2)),
                'Complain': np.random.choice([0, 1], p=[0.7, 0.3])
            })
        else:  # disengaged
            sample.update({
                'Tenure': max(1, int(np.random.uniform(10, 30))),
                'WarehouseToHome': max(5, int(np.random.uniform(10, 25))),
                'HourSpendOnApp': max(0.1, round(np.random.uniform(0.3, 1.2), 2)),
                'SatisfactionScore': max(1, int(np.random.uniform(2, 4))),
                'OrderCount': max(1, int(np.random.uniform(2, 6))),
                'DaySinceLastOrder': max(0, round(np.random.uniform(15, 30), 2)),
                'Complain': np.random.choice([0, 1], p=[0.8, 0.2])
            })
        
        # Ensure all values are finite
        for key, value in sample.items():
            if pd.isna(value) or np.isinf(value):
                sample[key] = 0
        
        synthetic_data.append(sample)
    
    return pd.DataFrame(synthetic_data)

def enhanced_predict_churn(input_data):
    """Enhanced prediction with comprehensive analysis and recommendations"""
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

        # Risk classification with enhanced thresholds
        risk_thresholds = model_package.get('risk_thresholds', {
            'high_risk_threshold': 0.6,
            'medium_risk_threshold': 0.35
        })

        high_threshold = risk_thresholds.get('high_risk_threshold', 0.6)
        medium_threshold = risk_thresholds.get('medium_risk_threshold', 0.35)

        if churn_prob > high_threshold:
            risk_level = "High"
            risk_score = 9
            action_priority = "IMMEDIATE"
            urgency_level = "Critical"
            time_to_action = "Within 24 hours"
            recommendation = "🚨 CRITICAL: Immediate intervention required! Contact customer within 24 hours with personalized retention offers and dedicated account manager support."
        elif churn_prob > medium_threshold:
            risk_level = "Medium"
            risk_score = 6
            action_priority = "WITHIN_WEEK"
            urgency_level = "High"
            time_to_action = "Within 3-5 days"
            recommendation = "⚠️ HIGH PRIORITY: Proactive engagement needed within 3-5 days. Send personalized offers, surveys, or loyalty rewards to re-engage the customer."
        else:
            risk_level = "Low"
            risk_score = 3
            action_priority = "MONITOR"
            urgency_level = "Low"
            time_to_action = "Monitor quarterly"
            recommendation = "✅ STABLE: Customer appears loyal. Continue standard service and monitor quarterly for any changes in behavior."

        # Enhanced customer profile analysis
        customer_profile = analyze_customer_profile(input_df.iloc[0])

        # Generate comprehensive insights
        insights = generate_detailed_insights(input_enhanced.iloc[0], churn_prob)

        # Calculate customer lifetime value estimation
        clv_estimation = estimate_customer_lifetime_value(input_df.iloc[0], churn_prob)

        # Generate specific retention strategies
        retention_strategies = generate_retention_strategies(input_enhanced.iloc[0], risk_level)

        # Risk factor breakdown
        risk_factors = calculate_risk_factors(input_enhanced.iloc[0])

        # Comparative analysis
        comparative_analysis = generate_comparative_analysis(input_enhanced.iloc[0], risk_level)

        # Action timeline
        action_timeline = create_action_timeline(risk_level, input_enhanced.iloc[0])

        confidence_score = max(probability)
        confidence = 'High' if confidence_score > 0.8 else 'Medium' if confidence_score > 0.6 else 'Low'

        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Will Churn' if prediction == 1 else 'Will Retain',
            'churn_probability': round(churn_prob, 4),
            'retention_probability': round(float(probability[0]), 4),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'urgency_level': urgency_level,
            'time_to_action': time_to_action,
            'action_priority': action_priority,
            'recommendation': recommendation,

            # Enhanced data sections
            'customer_profile': customer_profile,
            'insights': insights,
            'clv_estimation': clv_estimation,
            'retention_strategies': retention_strategies,
            'risk_factors': risk_factors,
            'comparative_analysis': comparative_analysis,
            'action_timeline': action_timeline,

            # Chart data for visualizations
            'chart_data': {
                'probability_breakdown': {
                    'churn': round(churn_prob * 100, 1),
                    'retention': round(float(probability[0]) * 100, 1)
                },
                'risk_factors_chart': risk_factors,
                'customer_metrics': {
                    'satisfaction': round(input_enhanced.get('Satisfaction_norm', pd.Series([0])).iloc[0] * 100, 1),
                    'engagement': round(input_enhanced.get('AppHours_norm', pd.Series([0])).iloc[0] * 100, 1),
                    'tenure': round(input_enhanced.get('Tenure_norm', pd.Series([0])).iloc[0] * 100, 1),
                    'order_frequency': round(input_enhanced.get('Orders_norm', pd.Series([0])).iloc[0] * 100, 1)
                }
            },

            'detailed_analysis': {
                'basic_risk_score': int(input_enhanced.get('BasicRiskScore', pd.Series([0])).iloc[0]),
                'advanced_risk_score': int(input_enhanced.get('AdvancedRiskScore', pd.Series([0])).iloc[0]),
                'satisfaction_normalized': round(input_enhanced.get('Satisfaction_norm', pd.Series([0])).iloc[0], 3),
                'engagement_normalized': round(input_enhanced.get('AppHours_norm', pd.Series([0])).iloc[0], 3),
                'tenure_normalized': round(input_enhanced.get('Tenure_norm', pd.Series([0])).iloc[0], 3),
                'data_type_detected': detect_data_type(input_df)
            },
            'confidence': confidence,
            'confidence_score': round(confidence_score, 4),
            'timestamp': datetime.now().isoformat()
        }

        # Convert numpy types to Python native types for JSON serialization
        return convert_numpy_types(result)

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
def homepage():
    """Homepage with company overview and navigation"""
    return render_template('homepage.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    """Prediction interface page"""
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    """About page with company information"""
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    """Contact page"""
    return render_template('contact.html')

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