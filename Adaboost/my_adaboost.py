import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics

file_path = "D:/大三下/人工智能和深度学习/ababoost/data/train_done.csv"
test_file_path = "D:/大三下/人工智能和深度学习/ababoost/data/test_done.csv"

# Load training data
try:
    data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
except UnicodeDecodeError:
    try:
        data = pd.read_csv(file_path, encoding='cp1252', low_memory=False)
    except UnicodeDecodeError:
        print("Failed to decode using standard encodings. Please check file encoding.")

# Load test data
try:
    test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1', low_memory=False)
except UnicodeDecodeError:
    try:
        test_data = pd.read_csv(test_file_path, encoding='cp1252', low_memory=False)
    except UnicodeDecodeError:
        print("Failed to decode using standard encodings. Please check file encoding for test data.")

# Feature selection
feature_cols = ['total_loan', 'year_of_loan', 'interest', 'monthly_payment', 'work_year', 'house_exist', 'issue_date',
                'debt_loan_ratio', 'del_in_18month', 'scoring_low', 'scoring_high', 'pub_dero_bankrup', 'early_return',
                'early_return_amount', 'early_return_amount_3mon', 'recircle_b', 'recircle_u', 'earlies_credit_mon'] + \
               data.columns[data.columns.str.startswith('class_')].tolist() + \
               data.columns[data.columns.str.startswith('employer_type_')].tolist() + \
               data.columns[data.columns.str.startswith('censor_status_')].tolist() + \
               data.columns[data.columns.str.startswith('use_')].tolist() + \
               data.columns[data.columns.str.startswith('initial_list_status_')].tolist() + \
               data.columns[data.columns.str.startswith('policy_code_')].tolist()

X = data[feature_cols]
y = data['is_default']

# Encoding target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=3407)

# Model training with cross-validation
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
scores = cross_val_score(abc, X_train, y_train, cv=100)
print("Mean accuracy with 100-fold cross-validation:", scores.mean())

# Predict test data
abc.fit(X_train, y_train)
test_X = test_data[feature_cols]
predicted_y = abc.predict(test_X)

# Save predicted results
result_df = pd.DataFrame({
    'user_id': test_data['user_id'],   # Assuming 'user_id' is a column in the test data
    'loan_id': test_data['loan_id'],   # Assuming 'loan_id' is a column in the test data
    'is_default_predicted': predicted_y
})

result_df.to_csv("D:/大三下/人工智能和深度学习/ababoost/data/predicted_results.csv", index=False)
print("Results saved to predicted_results.csv")
