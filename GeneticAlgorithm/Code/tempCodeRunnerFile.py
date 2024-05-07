import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from joblib import dump
import time
import numpy as np

def split_csv_file(filepath, output1, output2):
    df = pd.read_csv(filepath)
    half_point = len(df) // 2
    df_first_half = df.iloc[:half_point]
    df_second_half = df.iloc[half_point:]
    
    df_first_half.to_csv(output1, index=False)
    df_second_half.to_csv(output2, index=False)
    print(f"Files saved: {output1} and {output2}")

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['loan_id', 'user_id', 'is_default'])
    y = df['is_default']
    return X, y

def load_and_combine_data(filepath1, filepath2):
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    X = combined_df.drop(columns=['loan_id', 'user_id', 'is_default'])
    y = combined_df['is_default']
    return X, y

def create_model():
    return SVC(probability=True, kernel='rbf', C=1.0, random_state=3407)

def cross_validate_model(model, X, y, n_splits=100):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3407)
    auc_scores = []
    cross_entropy_losses = []
    train_scores = []
    test_scores = []

    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, y_pred))
        y_pred_proba = model.predict_proba(X_test)
        cross_entropy_losses.append(log_loss(y_test, y_pred_proba))
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    return auc_scores, cross_entropy_losses, train_scores, test_scores, start_time

def save_results(filename, results):
    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines([f"{name}: {value}\n" for name, value in results.items()])

def fit_and_save_final_model(model, X, y, filename):
    model.fit(X, y)
    dump(model, filename)
    return filename

def main():
    original_file = './SVM/Data/train_done.csv'
    output_file1 = './SVM/Data/train_done_part1.csv'
    output_file2 = './SVM/Data/train_done_part2.csv'
    split_csv_file(original_file, output_file1, output_file2)
    
if __name__ == '__main__':
    main()