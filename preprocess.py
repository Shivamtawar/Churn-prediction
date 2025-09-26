# STEP-BY-STEP CUSTOMER CHURN PREPROCESSING
# Copy and paste each step one by one

# ========================================
# STEP 1: INSTALL REQUIRED LIBRARIES
# ========================================
# Run this in your terminal/command prompt:
# pip install pandas numpy scikit-learn openpyxl matplotlib seaborn

# ========================================
# STEP 2: IMPORT LIBRARIES
# ========================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# ========================================
# STEP 3: LOAD YOUR DATASET
# ========================================
# CHANGE THIS PATH to your Excel file location
file_path = "D:\dataverse\E_Commerce_Dataset.xlsx"
sheet_name = "E Comm"

# Load data
try:
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"✅ Data loaded successfully!")
    print(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Columns: {list(data.columns)}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    print("Please check your file path and sheet name")

# ========================================
# STEP 4: EXPLORE YOUR DATA
# ========================================
print("\n=== DATA EXPLORATION ===")
print("First 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nData types:")
print(data.dtypes)

# ========================================
# STEP 5: DEFINE COLUMN TYPES
# ========================================
# UPDATE these lists to match YOUR dataset column names
numerical_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                  'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear', 
                  'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']

categorical_cols = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode', 'Gender', 
                    'MaritalStatus', 'PreferedOrderCat']

binary_cols = ['Complain']  # Yes/No columns

target_col = 'Churn'  # The column you want to predict

print(f"\n=== COLUMN CLASSIFICATION ===")
print(f"Numerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Binary columns: {binary_cols}")
print(f"Target column: {target_col}")

# ========================================
# STEP 6: HANDLE MISSING VALUES
# ========================================
print("\n=== HANDLING MISSING VALUES ===")

# Fill numerical columns with median (middle value)
for col in numerical_cols:
    if col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            print(f"✅ {col}: {missing_count} missing values filled with {median_val}")

# Fill categorical columns with most common value
for col in categorical_cols:
    if col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
            print(f"✅ {col}: {missing_count} missing values filled with '{mode_val}'")

print("✅ Missing values handled!")

# ========================================
# STEP 7: CONVERT TARGET VARIABLE (CHURN)
# ========================================
print("\n=== PROCESSING TARGET VARIABLE ===")

if target_col in data.columns:
    print(f"Original {target_col} values: {data[target_col].unique()}")
    
    # Convert to binary: 1 = churn, 0 = no churn
    data[target_col] = data[target_col].map({1: 1, 0: 0}).fillna(0).astype(int)
    
    churn_counts = data[target_col].value_counts()
    print(f"After conversion: {churn_counts.to_dict()}")
    print(f"Churn rate: {churn_counts[1] / len(data) * 100:.2f}%")

# ========================================
# STEP 8: CONVERT YES/NO TO 1/0
# ========================================
print("\n=== CONVERTING YES/NO TO 1/0 ===")

for col in binary_cols:
    if col in data.columns:
        print(f"Original {col} values: {data[col].unique()}")
        
        # Handle different possible values
        data[col] = data[col].map({
            'Yes': 1, 'No': 0, 'Y': 1, 'N': 0,
            'yes': 1, 'no': 0, 'y': 1, 'n': 0,
            1: 1, 0: 0, '1': 1, '0': 0
        }).fillna(0).astype(int)
        
        print(f"✅ {col} converted: {data[col].value_counts().to_dict()}")

# ========================================
# STEP 9: ENCODE CATEGORICAL VARIABLES
# ========================================
print("\n=== ENCODING CATEGORICAL VARIABLES ===")

label_encoders = {}
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = data[col].astype(str)  # Convert to string first
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        
        print(f"✅ {col} encoded: {len(le.classes_)} categories")

print("✅ All categorical variables encoded!")

# ========================================
# STEP 10: REMOVE OUTLIERS
# ========================================
print("\n=== REMOVING OUTLIERS ===")

original_rows = len(data)
print(f"Original dataset: {original_rows} rows")

for col in numerical_cols:
    if col in data.columns:
        # Calculate quartiles
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers_before = len(data)
        
        # Remove outliers
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        outliers_removed = outliers_before - len(data)
        if outliers_removed > 0:
            print(f"✅ {col}: {outliers_removed} outliers removed")

final_rows = len(data)
print(f"Final dataset: {final_rows} rows ({original_rows - final_rows} rows removed)")

# ========================================
# STEP 11: NORMALIZE NUMERICAL FEATURES
# ========================================
print("\n=== NORMALIZING FEATURES ===")

# Create scaler
scaler = MinMaxScaler()

# Get numerical columns that exist in data
numerical_cols_present = [col for col in numerical_cols if col in data.columns]

if numerical_cols_present:
    # Apply scaling (converts values to 0-1 range)
    data[numerical_cols_present] = scaler.fit_transform(data[numerical_cols_present])
    print(f"✅ Normalized {len(numerical_cols_present)} numerical features")

# ========================================
# STEP 12: FINAL DATA CHECK
# ========================================
print("\n=== FINAL DATA SUMMARY ===")
print(f"Final shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Check target distribution
if target_col in data.columns:
    target_dist = data[target_col].value_counts(normalize=True)
    print(f"\nTarget distribution:")
    print(f"No Churn (0): {target_dist[0]:.1%}")
    print(f"Churn (1): {target_dist[1]:.1%}")

print(f"\nFirst 3 rows of processed data:")
print(data.head(3))

# ========================================
# STEP 13: SAVE PROCESSED DATA
# ========================================
print("\n=== SAVING PROCESSED DATA ===")

output_file = "preprocessed_churn_data.csv"
data.to_csv(output_file, index=False)

print(f"✅ Processed data saved as '{output_file}'")
print(f"✅ Dataset ready for machine learning!")

# ========================================
# STEP 14: SUMMARY STATISTICS
# ========================================
print("\n" + "="*50)
print("🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"📁 Clean data saved as: {output_file}")
print(f"📊 Final dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"🎯 Ready for Random Forest training!")
print("="*50)

# Show what to do next
print("\n🚀 NEXT STEPS:")
print("1. Load the processed data: pd.read_csv('preprocessed_churn_data.csv')")
print("2. Split into train/test sets")
print("3. Train Random Forest classifier")
print("4. Evaluate model performance")
print("5. Deploy using Flask API")