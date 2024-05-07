import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, csv_dir, standard_csv_dir='./GridSearchCV/data/train_public.csv', mode='train'):
        super(MyDataset, self).__init__()
        self.df = pd.read_csv(csv_dir)
        st_df = pd.read_csv(standard_csv_dir)
        st_df = st_df.apply(pd.to_numeric, errors='coerce')
        self.mean_df = st_df.mean()
        self.std_df = st_df.std()

        self.num_item = ['total_loan', 'year_of_loan', 'interest', 'monthly_payment',
                         'debt_loan_ratio', 'del_in_18month', 'scoring_low', 'scoring_high',
                         'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup', 'recircle_b',
                         'recircle_u', 'f0', 'f1', 'f2', 'f3', 'f4', 'early_return', 'early_return_amount',
                         'early_return_amount_3mon']
        self.un_num_item = ['class', 'employer_type', 'industry', 'work_year', 'house_exist', 
                            'censor_status', 'use', 'initial_list_status', 'app_type', 'policy_code']
        self.un_use_item = ['loan_id', 'user_id', 'issue_date', 'post_code', 'region', 'earlies_credit_mon', 'title']
        self.un_num_item_list = {item: list(set(st_df[item].dropna().unique())) for item in self.un_num_item}
        self.mode = mode

    def __getitem__(self, index):
        data = [(0 if pd.isnull(self.df[item][index]) else (self.df[item][index] - self.mean_df[item]) / self.std_df[item]) 
                for item in self.num_item]
        emb_data = [self.un_num_item_list[item].index(self.df[item][index]) 
                    if self.df[item][index] in self.un_num_item_list[item] else -1 
                    for item in self.un_num_item]
        data.extend(emb_data)
        label = self.df['isDefault'][index] if self.mode == 'train' else self.df['loan_id'][index]
        return np.array(data, dtype=np.float32), label

    def __len__(self):
        return len(self.df)

# DataLoader设置
train_dataset = MyDataset('./GridSearchCV/data/train_public.csv')
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, drop_last=True)

# 准备数据
X_train = []
y_train = []
for data, label in train_loader:
    X_train.append(data)
    y_train.append(label)

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True))
])

# Define the parameter grid
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__class_weight': [{1: 5}, {1: 10}]
}

# Create the GridSearchCV object with high verbosity
grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=3, verbose=3)

# Fit the model
grid_search.fit(X_train, y_train)

# After fitting, print detailed results
if grid_search.verbose > 0:
    print("Grid search has completed. Displaying results:")
    for i, params in enumerate(grid_search.cv_results_['params']):
        print(f"Evaluation {i+1}/{len(grid_search.cv_results_['params'])}:")
        print(f"Params: {params}")
        print(f"Mean Test Score (AUC): {grid_search.cv_results_['mean_test_score'][i]:.4f}")
        print(f"Std Test Score: {grid_search.cv_results_['std_test_score'][i]:.4f}")

# Save the best model
joblib.dump(grid_search.best_estimator_, './GridSearchCV/model/GridSearchCV_model.joblib')
