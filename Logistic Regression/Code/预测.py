import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import load

model_filename = 'D:\AI&DL飞浆\\logistic_regression_model.joblib'
logistic_classifier = load(model_filename)

input_data_path = "D:\AI&DL飞浆\\test_done.csv"
input_data = pd.read_csv(input_data_path)

X_input = input_data.drop(columns=['loan_id', 'user_id','issue_date1','earlies_credit_mon1'])
predictions = logistic_classifier.predict(X_input)

output = pd.DataFrame({'id': range(0, len(predictions)),
                       'isDefault': predictions})

output.to_csv('D:\AI&DL飞浆\\submission.csv', index=False)