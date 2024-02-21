# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:15:43 2024

@author: Vijay
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from category_encoders import OneHotEncoder, MEstimateEncoder
# Read the data
X_full = pd.read_csv(r"D:\ASSIGNMENTS\kaggle\obease_data\train.csv")
X_test = pd.read_csv(r"D:\ASSIGNMENTS\kaggle\obease_test_data\test.csv")

# Remove rows with missing target, separate target from predictors
y = X_full.NObeyesdad
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_full.drop(['id', 'NObeyesdad'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, y_encoded, train_size=0.8, random_state=0
)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                  X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

best_params = {'grow_policy': 'depthwise', 'n_estimators': 982, 
               'learning_rate': 0.050053726931263504, 'gamma': 0.5354391952653927, 
               'subsample': 0.7060590452456204, 'colsample_bytree': 0.37939433412123275, 
               'max_depth': 23, 'min_child_weight': 21, 'reg_lambda': 9.150224029846654e-08,
               'reg_alpha': 5.671063656994295e-08}
best_params['booster'] = 'gbtree'
best_params['objective'] = 'multi:softmax'
best_params["device"] = "cuda"
best_params["verbosity"] = 0

# Build the Neural Network model
xgb_classifier =  XGBClassifier(**best_params)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                     
                      ('model', xgb_classifier)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

#print('MAE:', mean_absolute_error(y_valid, preds))

print(accuracy_score(y_valid,preds))

preds_test = clf.predict(X_test)  # Your code here

# Assuming label_encoder is the LabelEncoder you used
inverse_transformed_predictions = label_encoder.inverse_transform(preds_test)

inverse_transformed_predictions
# Save test predictions to file
output = pd.DataFrame({'id': X_test['id'],
                       'NObeyesdad': inverse_transformed_predictions})
output.to_csv('submission.csv', index=False)

