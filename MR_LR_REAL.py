import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from hyperimpute.plugins.imputers import Imputers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time


def omp_select_features(X, y, threshold, max_iter=100):
    numNeedingRepair = 0
    X_impute = X.copy()
    X_impute.fillna(X_impute.mean(), inplace=True)

    assert not X_impute.isna().any().any(), "There are still NaN values in X_impute."
    assert not y.isna().any(), "There are NaN values in the target vector y."

    complete_features = X.columns[X.notna().all()].tolist()
    S = [X.columns.get_loc(feature) for feature in complete_features] 
    print("Number of complete features:", len(S))

    incomplete_features = X.columns[X.isna().any()].tolist()
    incomplete_features_indices = [X.columns.get_loc(feature) for feature in incomplete_features]
    print("Number of incomplete features:", len(incomplete_features_indices))

    if complete_features:
        model = LinearRegression()
        model.fit(X_impute[complete_features], y)
        y_pred = model.predict(X_impute[complete_features])

        r = y - y_pred


    remaining_features = incomplete_features_indices
    print(f"Features with missing values: {remaining_features}")

    for _ in range(max_iter):
        if not remaining_features:
            break
        
        dot_products = X_impute.iloc[:, remaining_features].T @ r

        norms = np.linalg.norm(X_impute.iloc[:, remaining_features], axis=0) * np.linalg.norm(r)

        norms = np.where(norms == 0, 1e-10, norms)

        cosine_similarities = np.abs(dot_products / norms)
        
        nan_dot_products_count = np.sum(np.isnan(dot_products))
        nan_cosine_similarities_count = np.sum(np.isnan(cosine_similarities))

        max_cosine_similarity = np.max(cosine_similarities)

        if threshold > 0 and max_cosine_similarity < threshold:
            print(f"Max Cosine Similarity is below threshold, stopping.")
            break

        j = remaining_features[np.argmax(cosine_similarities)]
        S.append(j)
        remaining_features.remove(j)
        numNeedingRepair += 1

        if np.isnan(max_cosine_similarity):
            print("Max Cosine Similarity is NaN, stopping.")
            break
        model = LinearRegression()
        model.fit(X_impute.iloc[:, S], y)

        y_pred = model.predict(X_impute.iloc[:, S])

        r = y - y_pred

    return (S, numNeedingRepair)

def evaluate_model(X_train, X_test, y_train, y_test, impute_strategy, must_impute=None):
    if impute_strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif impute_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif impute_strategy == 'min':
        imputer = None  
    elif impute_strategy == 'di':
        imputer = Imputers().get('miwae')

    X_train_impute = X_train.copy()
    X_test_impute = X_test.copy()


    start_time = time.time()  

    if must_impute is None:
        print("Imputing all features...")
        if impute_strategy == 'min':
            for feature in range(X_train.shape[1]):
                min_value = X_train.iloc[:, feature].min()
                X_train_impute.iloc[:, feature].fillna(min_value, inplace=True)
                X_test_impute.iloc[:, feature].fillna(min_value, inplace=True)
        else:
            X_train_impute = imputer.fit_transform(X_train)
            X_test_impute = imputer.transform(X_test)
    
    elif len(must_impute) == 0:
        print("No features were selected to be imputed, removing columns with missing values.")
        missing_value_columns = X_train.columns[X_train.isna().any()].tolist()

        X_train_impute = X_train_impute.drop(columns=missing_value_columns)
        X_test_impute = X_test_impute.drop(columns=missing_value_columns)

    else:
        if impute_strategy == 'min':
            for feature in must_impute:
                min_value = X_train.iloc[:, feature].min()
                X_train_impute.iloc[:, feature].fillna(min_value, inplace=True)
                X_test_impute.iloc[:, feature].fillna(min_value, inplace=True)
        else:
            if len(must_impute) == 1000:
                must_impute_sorted = sorted(must_impute)
                X_train_min = X_train.iloc[:, must_impute_sorted]
                X_test_min = X_test.iloc[:, must_impute_sorted]
                
                train_min_with_target = pd.concat([X_train_min, y_train], axis=1)
                train_min_with_target.to_csv("./Linear Regression/datasets/df_train_cancer_MI.csv", index=False)

                test_min_with_target = pd.concat([X_test_min, y_test], axis=1)
                test_min_with_target.to_csv("./Linear Regression/datasets/df_test_cancer_MI.csv", index=False)

                test_min_cleaned_with_target = test_min_with_target.dropna(axis=0)
                test_min_cleaned_with_target.to_csv("./Linear Regression/datasets/df_test_cancer_dropped_MI.csv", index=False)


            X_train_impute.iloc[:, must_impute] = imputer.fit_transform(X_train.iloc[:, must_impute]) 
            X_test_impute.iloc[:, must_impute] = imputer.transform(X_test.iloc[:, must_impute]) 

        missing_value_columns = X_train_impute.columns[X_train_impute.isna().any()].tolist()

        X_train_impute = X_train_impute.drop(columns=missing_value_columns)
        X_test_impute = X_test_impute.drop(columns=missing_value_columns)

    X_train_impute = np.asarray(X_train_impute)
    X_test_impute = np.asarray(X_test_impute)
    assert not np.isnan(X_train_impute).any(), "There are still NaN values in X_train_impute."
    assert not np.isnan(X_test_impute).any(), "There are still NaN values in X_test_impute."

    elapsed_time = time.time() - start_time 

    model = LinearRegression()
    model.fit(X_train_impute, y_train)

   
    y_pred = model.predict(X_test_impute)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse, elapsed_time

# Load data
print("Loading data...")

data = pd.read_csv('./Minimal-Repair/Linear Regression/datasets/CommunitiesAndCrime.csv')


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

X = data.drop(columns=['population'])
y = data['population']


X_train = train_data.drop(columns=['population'])
y_train = train_data['population']

X_test = test_data.drop(columns=['population'])
y_test = test_data['population']


print("Calculating baseline performance...")

mse_mean_baseline, time_mean_baseline = evaluate_model(X_train, X_test, y_train, y_test, 'mean')
mse_knn_baseline, time_knn_baseline = evaluate_model(X_train, X_test, y_train, y_test, 'knn')
mse_min_baseline, time_min_baseline = evaluate_model(X_train, X_test, y_train, y_test, 'min')

print(f"Baseline MSE (MI): {mse_mean_baseline:.4f}, Time: {time_mean_baseline:.4f} seconds")
print(f"Baseline MSE (KI): {mse_knn_baseline:.4f}, Time: {time_knn_baseline:.4f} seconds")
print(f"Baseline MSE (MinI): {mse_min_baseline:.4f}, Time: {time_min_baseline:.4f} seconds")

print("Starting threshold tuning...")
thresholds = np.logspace(-1, -7, num=10)
results = []
print("\n")

for threshold in thresholds:
    start = time.time()
    print("Threshold:", threshold)
    must_impute_features, numNeedingRepair = omp_select_features(X, y, threshold)

    end= time.time()
    proces = end-start
    print(f"Time Finding Minimal: {proces:.4f}")
    print(f"Must-Impute Missing Features: {numNeedingRepair}")
    print(f"Number of Must-Impute Features: {len(must_impute_features)}")
    mse_mean, time_mean = evaluate_model(X_train, X_test, y_train, y_test, 'mean', must_impute=must_impute_features)
    mse_knn, time_knn = evaluate_model(X_train, X_test, y_train, y_test, 'knn', must_impute=must_impute_features)
    mse_min, time_min = evaluate_model(X_train, X_test, y_train, y_test, 'min', must_impute=must_impute_features)
    
    print(f"MSE (MI): {mse_mean:.4f}, Time: {time_mean:.4f} seconds")
    print(f"MSE (KI): {mse_knn:.4f}, Time: {time_knn:.4f} seconds")
    print(f"MSE (MinI): {mse_min:.4f}, Time: {time_min:.4f} seconds")
    print("")

    
    results.append({
        'Threshold': f"{threshold:.4f}",
        '# Features Imputed': numNeedingRepair,
        'MSE (MI)': f"{mse_mean:.4f}",
        'Time (MI)': f"{time_mean:.4f}",
        'MSE (KI)': f"{mse_knn:.4f}",
        'Time (KI)': f"{time_knn:.4f}",
        'MSE (MinI)': f"{mse_min:.4f}",
        'Time (MinI)': f"{time_min:.4f}"
    })


# print results to csv
print(f"Baseline MSE (MI): {mse_mean_baseline:.4f}, Time: {time_mean_baseline:.4f} seconds")
print(f"Baseline MSE (KI): {mse_knn_baseline:.4f}, Time: {time_knn_baseline:.4f} seconds")
print(f"Baseline MSE (MinI): {mse_min_baseline:.4f}, Time: {time_min_baseline:.4f} seconds")

# print baseline results to a seperate csv
baseline_results = {
    '# Features Imputed': len(X.columns[X.isna().any()].tolist()),
    'Baseline MSE (MI)': f"{mse_mean_baseline:.4f}",
    'Time (MI)': f"{time_mean_baseline:.4f}",
    'Baseline MSE (KI)': f"{mse_knn_baseline:.4f}",
    'Time (KI)': f"{time_knn_baseline:.4f}",
    'Baseline MSE (MinI)': f"{mse_min_baseline:.4f}",
    'Time (MinI)': f"{time_min_baseline:.4f}"
}
baseline_results_df = pd.DataFrame([baseline_results])
baseline_results_df.to_csv('./Minimal-Repair/Linear Regression/results/cancerResultsBaseline.csv', index=False)
