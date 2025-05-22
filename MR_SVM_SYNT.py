import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
import pandas as pd
import time
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from hyperimpute.plugins.imputers import Imputers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle as pickle
from re import X 
import sklearn as skl
from sklearn import svm 
from datetime import datetime
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")


#----------------------------------------------------MI Checker---------------------------------------------------

def generate_random_repair_with_edge(dataset, index_and_edge_repair, col_min_global, col_max_global, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    new_dataset = dataset.copy() 
    for key, edge_repair_val in index_and_edge_repair.items():
        new_dataset[key] = edge_repair_val

    nan_mask = np.isnan(new_dataset)
    nan_rows, nan_cols = np.where(nan_mask)
    if nan_rows.size == 0:
        return new_dataset
    col_mins_at_nans = col_min_global[nan_cols]
    col_maxs_at_nans = col_max_global[nan_cols]

    random_values = np.empty_like(col_mins_at_nans, dtype=float)

    normal_case_mask = (~np.isnan(col_mins_at_nans)) & \
                       (~np.isnan(col_maxs_at_nans)) & \
                       (col_mins_at_nans < col_maxs_at_nans)
    
    if np.any(normal_case_mask):
        random_values[normal_case_mask] = np.random.uniform(
            col_mins_at_nans[normal_case_mask],
            col_maxs_at_nans[normal_case_mask],
            size=normal_case_mask.sum()
        )

    special_case_mask = ~normal_case_mask
    if np.any(special_case_mask):
        fill_values_for_special_cases = np.where(
            np.isnan(col_mins_at_nans[special_case_mask]), 
            0, 
            col_mins_at_nans[special_case_mask]
        )
        random_values[special_case_mask] = fill_values_for_special_cases

    new_dataset[nan_rows, nan_cols] = random_values
    
    return new_dataset


def findEdgeRepair(incomplete_example, repaired_other_examples, model, y_incompleteExample):
    col_min = np.nanmin(repaired_other_examples, axis=0)
    col_max = np.nanmax(repaired_other_examples, axis=0)
    repaired_example = incomplete_example.copy()

    for i, val in enumerate(incomplete_example):
        if np.isnan(val):

            if not hasattr(model, 'coef_') or model.coef_ is None:
                 raise RuntimeError("Model coefficients are not available in findEdgeRepair. Ensure the model is fitted.")
            coef_value = model.coef_[0][i]


            current_col_min_i = col_min[i]
            current_col_max_i = col_max[i]
            if np.isnan(current_col_min_i) and np.isnan(current_col_max_i):
                repaired_example[i] = 0
                continue
            if np.isnan(current_col_min_i): current_col_min_i = current_col_max_i if not np.isnan(current_col_max_i) else 0
            if np.isnan(current_col_max_i): current_col_max_i = current_col_min_i if not np.isnan(current_col_min_i) else 0
            
            if np.isnan(current_col_min_i): current_col_min_i = 0
            if np.isnan(current_col_max_i): current_col_max_i = 0

            if y_incompleteExample > 0:
                repaired_example[i] = current_col_min_i if coef_value > 0 else current_col_max_i
            else:
                repaired_example[i] = current_col_max_i if coef_value > 0 else current_col_min_i
    return repaired_example

def checkSV(incompleteExample, repairedOtherExamples, labels, y_incompleteExample, seed=None,
            model_instance=None,
            fitted_flag=None):

    new_fitted_flag = fitted_flag

    if not fitted_flag:
        model_instance.fit(repairedOtherExamples, labels)
        new_fitted_flag = True
    else:
        model_instance.partial_fit(repairedOtherExamples, labels)


    repaired_example = findEdgeRepair(incompleteExample, repairedOtherExamples, model_instance, y_incompleteExample)

    decision_value = np.dot(model_instance.coef_, repaired_example) + model_instance.intercept_
    product = y_incompleteExample * decision_value

    if product < 1:
        return True, repaired_example, new_fitted_flag

    return False, repaired_example, new_fitted_flag

def checkRepairNecessity(possible_repaired_dataset, original_dataset, labels, example_index, seed=None,
                             model_instance=None,
                             fitted_flag=None):
    possible_repaired_data_subset = np.delete(possible_repaired_dataset, example_index, axis=0)
    labels_subset = np.delete(labels, example_index, axis=0)
    if possible_repaired_data_subset.shape[0] == 0 or len(np.unique(labels_subset)) < 2:
        raise ValueError("Insufficient data or classes")

    incomplete_example = original_dataset[example_index]
    is_sv_result, repaired_example_result, updated_fitted_flag = \
        checkSV(incomplete_example, possible_repaired_data_subset, labels_subset, labels[example_index], seed,
                model_instance=model_instance,
                fitted_flag=fitted_flag)
    return is_sv_result, repaired_example_result, updated_fitted_flag


def findminimalRepair(original_dataset, labels, seed=None):
    minimal_Repair = []
    minimal_Repair_examples = []
    index_and_edge_repair = {}

    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed, warm_start=True)
    model_has_been_fully_fitted = False

    global_col_min = np.nanmin(original_dataset, axis=0)
    global_col_max = np.nanmax(original_dataset, axis=0)

    for index, example in enumerate(original_dataset):
        
        if np.isnan(example).any():
            repaired_dataset = generate_random_repair_with_edge(
                original_dataset, 
                index_and_edge_repair, 
                global_col_min, 
                global_col_max, 
                seed
            )

            try:
                is_support_vector, repair_for_this_example, model_has_been_fully_fitted = \
                    checkRepairNecessity(repaired_dataset, original_dataset, labels, index, seed,
                                             model_instance=model,
                                             fitted_flag=model_has_been_fully_fitted)
            except ValueError as e:
                raise e

            index_and_edge_repair[index] = repair_for_this_example

            if is_support_vector:
                minimal_Repair.append([list(repair_for_this_example), index])
                minimal_Repair_examples.append(index)

    return minimal_Repair, minimal_Repair_examples, index_and_edge_repair

def get_Xy(data, label=None):
    if label== None:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return np.array(X), np.array(y)
    else:
        X = data.drop(label, axis=1)
        y = data[label]
    return np.array(X), np.array(y)


#----------------------------------------------------Repair Method---------------------------------------------------

def mice_impute(X_train, max_iter=2, random_state=51, condition="Fit", imputer=None):
    if condition == "Fit":
        complete_X_train = X_train.copy()

        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, verbose=2, skip_complete=False)
        start_fit = time.time()
        imputer.fit(complete_X_train) 
        end_fit = time.time()

        train_fit_time = end_fit-start_fit
        return imputer, train_fit_time

    else:
        if imputer is None:
            raise ValueError("Imputer must be provided for transform step.")
        missing_X_train = X_train.copy()

        start_fit = time.time()
        imputed_values=imputer.transform(missing_X_train)
        end_fit = time.time()
        missing_X_train = imputed_values 

        inference_time = end_fit-start_fit
        return missing_X_train, inference_time 

def mean_Repair(X_train, condition="Fit", imputer=None):
    if condition == "Fit":
        complete_X_train = X_train.copy()
        imputer = SimpleImputer()
        start_fit = time.time()
        imputer.fit(complete_X_train)
        end_fit = time.time()

        train_fit_time = end_fit-start_fit
        return imputer, train_fit_time
    else:
        if imputer is None:
            raise ValueError("Imputer must be provided for transform step.")
        missing_X_train = X_train.copy()

        start_fit = time.time()
        imputed_values=imputer.transform(missing_X_train)
        end_fit = time.time()
        missing_X_train = imputed_values

        inference_time = end_fit-start_fit
        return missing_X_train, inference_time 


def knn_Repair(X_train, condition="Fit", imputer=None, neighbors=5):
    if condition == "Fit":
        complete_X_train = X_train.copy()

        imputer = KNNImputer(n_neighbors=neighbors)
        start_fit = time.time()
        imputer.fit(complete_X_train)
        end_fit = time.time()

        train_fit_time = end_fit-start_fit
        return imputer, train_fit_time
    else:
        if imputer is None:
            raise ValueError("Imputer must be provided for transform step.")
        missing_X_train = X_train.copy()

        start_fit = time.time()
        imputed_values=imputer.transform(missing_X_train)
        end_fit = time.time()
        missing_X_train = imputed_values 
        inference_time = end_fit-start_fit

        return missing_X_train, inference_time 


def SGD_class(X_train, Y_train, X_test, Y_test, iter=1000000, tolerance=1e-7, seed=None):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SGDClassifier(loss='hinge', max_iter=iter, tol=tolerance, fit_intercept=True, random_state=seed)
    start = time.time()
    model.fit(X_train_scaled, Y_train)
    duration = time.time() - start

    train_accuracy = model.score(X_train_scaled, Y_train)
    test_accuracy = model.score(X_test_scaled, Y_test)

    return train_accuracy, test_accuracy, duration

#----------------------------------------------------Active Clean-------------------------------------------------------
def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s,t in enumerate(imap) if t in lset]

def error_classifier(total_labels, full_data):
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
        clf.fit(full_data[indices,:],labels)
        return clf
    else:
        return None

def ec_filter(dirtyex, full_data, clf, t=0.90):
    if clf != None:
        pred = clf.predict_proba(full_data[dirtyex,:])
        return [j for i,j in enumerate(dirtyex) if pred[i][0] < t]
    return dirtyex


def activeclean(dirty_data, clean_data, test_data, full_data, indextuple, task='classification', batchsize=50, total=10000):
    X = dirty_data[0][translate_indices(indextuple[0], indextuple[1]), :]
    y = dirty_data[1][translate_indices(indextuple[0], indextuple[1])]

    X_clean = clean_data[0]
    y_clean = clean_data[1]

    X_test = test_data[0].values
    y_test = test_data[1].values

    lset = set(indextuple[2])
    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    total_labels = []
    total_cleaning = 0  

    topbatch = np.random.choice(range(0, len(dirtyex)), batchsize)
    examples_real = [dirtyex[j] for j in topbatch]
    examples_map = translate_indices(examples_real, indextuple[2])

    cleanex.extend(examples_map)
    for j in set(examples_real):
        dirtyex.remove(j)

    if task =='classification':
        clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    else:
        clf = SGDRegressor(penalty=None,max_iter=200)
    clf.fit(X_clean[cleanex, :], y_clean[cleanex])

    for i in range(50, total, batchsize):
        ypred = clf.predict(X_test)

        examples_real = np.random.choice(dirtyex, batchsize)

        missing_count = sum(1 for r in examples_real if r in indextuple[1])
        total_cleaning += missing_count 

        examples_map = translate_indices(examples_real, indextuple[2])

        total_labels.extend([(r, (r in lset)) for r in examples_real])

        ec = error_classifier(total_labels, full_data)

        for j in examples_real:
            try:
                dirtyex.remove(j)
            except ValueError:
                pass

        dirtyex = ec_filter(dirtyex, full_data, ec)

        cleanex.extend(examples_map)

        clf.partial_fit(X_clean[cleanex, :], y_clean[cleanex])


        if len(dirtyex) < 50:
            print("[ActiveClean Real] No More Dirty Data Detected")
            print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
            if task == 'classification':
                ypred = clf.predict(X_test)
                return total_cleaning, 0 if accuracy_score(y_test, ypred) is None else accuracy_score(y_test, ypred)

            else:
                ypred = clf.predict(X_test)
                return total_cleaning, 0 if mean_squared_error(y_test, ypred) is None else mean_squared_error(y_test, ypred)

    print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
    if task == 'classification':
        ypred = clf.predict(X_test)
        return total_cleaning, 0 if accuracy_score(y_test, ypred) is None else accuracy_score(y_test, ypred)

    else:
        ypred = clf.predict(X_test)
        return total_cleaning, 0 if mean_squared_error(y_test, ypred) is None else mean_squared_error(y_test, ypred)


def generate_AC_data(df_train, df_test):
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    features, target = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    features_test, target_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    ind = list(features[features.isna().any(axis=1)].index)
    not_ind = list(set(range(features.shape[0])) - set(ind))

    feat = np.where(df_train.isnull().any())[0]

    e_feat = np.copy(features)
    for i in ind:
        for j in feat:
            e_feat[i, j] = 0.01 * np.random.rand()

    return (
        features_test,
        target_test,
        csr_matrix(e_feat[not_ind, :]),
        np.ravel(target[not_ind]),
        csr_matrix(e_feat[ind, :]),
        np.ravel(target[ind]),
        csr_matrix(e_feat),
        np.arange(len(e_feat)).tolist(),
        ind,
        not_ind,
    )


def active_clean_driver(df_train, df_test):
    (
        features_test,
        target_test,
        X_clean,
        y_clean,
        X_dirty,
        y_dirty,
        X_full,
        train_indices,
        indices_dirty,
        indices_clean,
    ) = generate_AC_data(df_train, df_test)

    start_time = time.time()

    AC_records_1, AC_score_1 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_2, AC_score_2 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_3, AC_score_3 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_4, AC_score_4 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_5, AC_score_5 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )

    end_time = time.time()

    elapsed_time = end_time - start_time
    AC_time = elapsed_time / 5
    ac_records_list = [AC_records_1, AC_records_2, AC_records_3, AC_records_4, AC_records_5]
    ac_scores_list  = [AC_score_1,  AC_score_2,  AC_score_3,  AC_score_4,  AC_score_5]
    for i in range(1, 6):
        print(f"AC_records_{i} =", eval(f"AC_records_{i}"))
        print(f"AC_score_{i} =", eval(f"AC_score_{i}"))
    AC_records_mean = np.mean(ac_records_list)
    AC_records_std  = np.std(ac_records_list)

    AC_score_mean   = np.mean(ac_scores_list)
    AC_score_std    = np.std(ac_scores_list)

    # Print results
    print(f"\nAverage AC_records: {AC_records_mean:.4f} ± {AC_records_std:.4f}")
    print(f"Average AC_score:   {AC_score_mean:.4f} ± {AC_score_std:.4f}")
    AC_records = (
                         AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5
                 ) / 5
    AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5

    return AC_records, AC_score, AC_time



#----------------------------------------------------Utility Function---------------------------------------------------
def sanity_check(X_check, minimal_Repair_examples_check): 
    missing_rows = np.where(np.isnan(X_check).any(axis=1))[0]
    examples_drop = [example for example in missing_rows if example not in minimal_Repair_examples_check]
    return len(examples_drop)

def make_dirty(df_md, random_seed_md, missing_factor_md, dirty_cols=1): 
    np.random.seed(random_seed_md)
    num_dirty_cols_md = dirty_cols 

    num_rows_md = df_md.shape[0] 
    num_dirty_rows_md = int(missing_factor_md * num_rows_md) 

    feature_columns_md = df_md.columns[:-1] if len(df_md.columns) > 1 else df_md.columns
    if not feature_columns_md.empty and num_dirty_cols_md > 0:
        if num_dirty_cols_md > len(feature_columns_md): 
            num_dirty_cols_md = len(feature_columns_md)
        selected_dirty_cols_md = np.random.choice(feature_columns_md, num_dirty_cols_md, replace=False) 
    else: 
        return df_md.copy()

    if num_rows_md > 0 and num_dirty_rows_md > 0:
        actual_dirty_rows_to_select = min(num_dirty_rows_md, num_rows_md)
        dirty_rows_indices_md = np.random.choice(df_md.index, actual_dirty_rows_to_select, replace=False) 
    else: 
        return df_md.copy()

    df_dirty_output = df_md.copy() 

    if len(dirty_rows_indices_md) > 0 and len(selected_dirty_cols_md) > 0 :
        for col_name in selected_dirty_cols_md: # Changed from direct .loc to loop for safety
            df_dirty_output.loc[dirty_rows_indices_md, col_name] = np.nan
    
    return df_dirty_output


#############################################################################################
#                                                                                           #
#                                   MAIN STARTS HERE                                        #
#                                                                                           #
#############################################################################################

# Main execution block exactly as it was in the Canvas artifact "optimized_Repair_code_minimal_changes"
# WITH CORRECTION FOR 'name' vs 'name_main'
if __name__ == '__main__':
    print("Run started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    seeds_to_try = [42]
    missing_level = [0.2, 0.4, 0.6] 

    DATASET_CHOICE = "default" 
    file_path = None
    name = None 
    col_num = None
    df = None   

    if DATASET_CHOICE == "tuadromd":
        file_path = './MI/SVM/synthetic/data/original/tuadromd.csv' 
        name = "tuadromd" 
        col_num = 48       
        df = pd.read_csv(file_path) 
        last_column_index = df.columns[-1] 
        df[last_column_index] = df[last_column_index].replace({'malware': 1, 'goodware': -1}).astype(int) 
    elif DATASET_CHOICE == "skin":
        file_path = './Minimal-Repair/Synthetic-Datasets/skin.txt' 
        name = "skin" 
        col_num = 1   
        df = pd.read_csv(file_path, sep='\t', header=None) 
        df.iloc[:, -1] = df.iloc[:, -1].replace({1: 1, 2: -1}).astype(int) 
    elif DATASET_CHOICE == "malware":
        data_file = './Minimal-Repair/Synthetic-Datasets/REJAFADA.data' 
        name = "malware"  
        df = pd.read_csv(data_file) 
        df = df.drop(df.columns[0], axis=1)
        first_column = df.pop(df.columns[0])
        df[df.columns[-1]] = first_column 
        df[df.columns[-1]] = df[df.columns[-1]].replace({'M': 1, 'B': -1})
        col_num = 48  
    elif DATASET_CHOICE == "default":
        data_file = './MI/SVM/synthetic/data/original/default.csv'
        name = "default"
        df = pd.read_csv(data_file, header=None).iloc[2:, 1:]
        df = df.astype(float)
        col_num = 10
    else:
        raise ValueError(f"Invalid DATASET_CHOICE: {DATASET_CHOICE}")

    target_col_name = df.columns[-1] 
    stratify_on = None
    if len(df[target_col_name].unique()) > 1: 
        stratify_on = df[target_col_name]   
    
    OG_train_data_df, OG_test_data_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    OG_train_data_df = OG_train_data_df.reset_index(drop=True)
    OG_test_data_df = OG_test_data_df.reset_index(drop=True)

    for missingness in missing_level: 
        print(f"--------------------------------------------- Started Missing Factor: {missingness} for {name} ------------------------------------------")
        train_data = make_dirty(OG_train_data_df, 42, missingness, dirty_cols=col_num)
        has_nan = train_data.isnull().values.any()
        print("Contains NaN:", has_nan)
        test_data = OG_test_data_df.copy()

        label_col = train_data.columns[-1]

        X_train = train_data.drop(columns=[label_col]) 
        y_train = train_data[label_col] 

        X_test = test_data.drop(columns=[label_col]) 
        y_test = test_data[label_col] 

        scaler = MinMaxScaler() 
        X_train_scaled = scaler.fit_transform(X_train.values) 
        X_test_scaled = scaler.transform(X_test.values)   

        train_data_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=train_data.index)
        train_data_scaled[label_col] = y_train

        test_data_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=test_data.index)
        test_data_scaled[label_col] = y_test

        X_train_np, Y_train_np = get_Xy(train_data) 
        X_test_np, Y_test_np = get_Xy(test_data)     

        total_examples = len(X_train_np)
        if DATASET_CHOICE == "default":
            X_train_np = X_train_np.astype(float) 
        rows_with_missing_values = np.isnan(X_train_np).any(axis=1).sum() 
        missing_factor = rows_with_missing_values / total_examples if total_examples > 0 else 0

        print("Number of rows with missing values:", rows_with_missing_values)
        print(f"Total example {X_train_np.shape}, MISSING FACTOR: {missing_factor}") 

        min_time = float('inf') 
        seed_main_loop = None 
        number_of_example_dropped = None 
        
        scaler_std_main = MinMaxScaler() 
        X_train_scaled_std_main = scaler_std_main.fit_transform(X_train_np) 
        X_test_scaled_std_main = scaler_std_main.transform(X_test_np)     

        examples_cleaned_AC, accuracy_AC ,training_time_AC = active_clean_driver(train_data_scaled, test_data_scaled)
        print(f"Accuracy_AC: {accuracy_AC}") 
        print(f"Examples Cleaned AC: {examples_cleaned_AC}")
        print(f"Training_Time_AC: {training_time_AC}\n")

        print("--------------------Evaluating SGDClassifier on Original (Clean) Data--------------------------")
        X_train_OG_np, Y_train_OG_np = get_Xy(OG_train_data_df) 
        X_test_OG_np, Y_test_OG_np = get_Xy(OG_test_data_df)       

        train_accuracies_orig_eval = [] 
        test_accuracies_orig_eval = []  
        
        for i in range(0, 10):
            train_acc, test_acc, _ = SGD_class(X_train_OG_np, Y_train_OG_np, X_test_OG_np, Y_test_OG_np) 
            train_accuracies_orig_eval.append(train_acc)
            test_accuracies_orig_eval.append(test_acc)
            
        avg_test_acc_orig_eval = np.mean(test_accuracies_orig_eval) 
        std_test_acc_orig_eval = np.std(test_accuracies_orig_eval)   
        avg_train_acc_orig_eval = np.mean(train_accuracies_orig_eval) 

        print(f"\nAverage train accuracy: {avg_train_acc_orig_eval:.4f}")
        print(f"Average test accuracy: {avg_test_acc_orig_eval:.4f}")
        print(f"Test accuracy standard deviation: {std_test_acc_orig_eval:.4f}")
        
        print("--------------------Evaluating SGDClassifier on KNN Impute Data--------------------------")
        train_mask_complete_knn = ~np.isnan(X_train_np).any(axis=1)
        X_train_drop_knn = X_train_np[train_mask_complete_knn] 
        
        model_knn_main = None 
        if X_train_drop_knn.shape[0] > 0: 
            model_knn_main, knn_fit_time_main = knn_Repair(X_train_drop_knn, condition="Fit") 
            X_train_imputed_knn_main, knn_infer_time_main = knn_Repair(X_train_np.copy(), condition="Transform", imputer=model_knn_main) 
            
            train_accuracies_knn_eval = [] 
            test_accuracies_knn_eval = []  
            durations_knn_eval = []        

            for i in range(0, 10):
                train_acc, test_acc, duration = SGD_class(X_train_imputed_knn_main, Y_train_np, X_test_np, Y_test_np)
                train_accuracies_knn_eval.append(train_acc)
                test_accuracies_knn_eval.append(test_acc)
                durations_knn_eval.append(duration)            

            avg_test_acc_knn_eval = np.mean(test_accuracies_knn_eval) 
            std_test_acc_knn_eval = np.std(test_accuracies_knn_eval)   
            avg_train_acc_knn_eval = np.mean(train_accuracies_knn_eval) 
            avg_duration_knn_eval = np.mean(durations_knn_eval)         
            
            total_time_knn_main = knn_fit_time_main + knn_infer_time_main + avg_duration_knn_eval 

            print(f"\nAverage train accuracy: {avg_train_acc_knn_eval:.4f}")
            print(f"Average test accuracy: {avg_test_acc_knn_eval:.4f}")
            print(f"Test accuracy standard deviation: {std_test_acc_knn_eval:.4f}")
            print(f"\nAverage duration: {avg_duration_knn_eval:.4f}")
            print(f"Duration standard deviation: {np.std(durations_knn_eval):.4f}") 
            print(f"Total time using KNN-imputed data: {total_time_knn_main:.4f}")
        else:
            print("KNN Repair skipped: No fully complete rows in training data to fit imputer.")

        model_mice_main = None 
        if name != "malware":
            print("--------------------Evaluating SGDClassifier on MICE Impute Data--------------------------")
            train_mask_complete_mice = ~np.isnan(X_train_np).any(axis=1)
            X_train_drop_mice = X_train_np[train_mask_complete_mice] 
            
            if X_train_drop_mice.shape[0] > 0:
                model_mice_main, mice_fit_time_main = mice_impute(X_train_drop_mice, condition="Fit") 
                X_train_imputed_mice_main, mice_infer_time_main = mice_impute(X_train_np.copy(), condition="Transform", imputer=model_mice_main) 
                
                train_accuracies_mice_eval = [] 
                test_accuracies_mice_eval = []  
                durations_mice_eval = []        

                for i in range(0, 10):
                    train_acc, test_acc, duration = SGD_class(X_train_imputed_mice_main, Y_train_np, X_test_np, Y_test_np)
                    train_accuracies_mice_eval.append(train_acc)
                    test_accuracies_mice_eval.append(test_acc)
                    durations_mice_eval.append(duration)

                avg_test_acc_mice_eval = np.mean(test_accuracies_mice_eval) 
                std_test_acc_mice_eval = np.std(test_accuracies_mice_eval)   
                avg_train_acc_mice_eval = np.mean(train_accuracies_mice_eval) 
                avg_duration_mice_eval = np.mean(durations_mice_eval)         
                
                total_time_mice_main = mice_fit_time_main + mice_infer_time_main + avg_duration_mice_eval 

                print(f"\nAverage train accuracy: {avg_train_acc_mice_eval:.4f}")
                print(f"Average test accuracy: {avg_test_acc_mice_eval:.4f}")
                print(f"Test accuracy standard deviation: {std_test_acc_mice_eval:.4f}")
                print(f"\nAverage duration: {avg_duration_mice_eval:.4f}")
                print(f"Duration standard deviation: {np.std(durations_mice_eval):.4f}")
                print(f"Total time using MICE-imputed data: {total_time_mice_main:.4f}")
            else:
                print("MICE Repair skipped: No fully complete rows for fitting.")
        else: 
             print("MICE Repair skipped for 'malware' dataset.")

        for seed_iter_loop in seeds_to_try: 
            print(f"--------------------------------------------- Seed: {seed_iter_loop} ------------------------------------------")
            X_train_filled_iter = X_train_np.copy() 
            Y_train_filled_iter = Y_train_np.copy() 
            
            iter_loop_start_time = time.time() 
            accuracy_time_total_iter = 0 
            Repair_total_iter = 0    
            missing_value_initial_iter = rows_with_missing_values 

            max_iter_loop = 10 
            itter_method_loop = "GT" 

            if itter_method_loop == "GT": 
                for i_loop in range(0,max_iter_loop): 
                    print(f"\n--- Iteration {i_loop+1}/{max_iter_loop} ---")
                    minimal_Repair_res, Repair_required_indices_res, _ = findminimalRepair(
                        X_train_filled_iter, Y_train_filled_iter, seed=seed_iter_loop
                    ) 
                    print("need to be imputed: ", len(Repair_required_indices_res))
                    num_dropped_sanity = sanity_check(X_train_filled_iter, Repair_required_indices_res) 
                    print("number of example dropped: ", num_dropped_sanity)

                    X_train_filled_iter[Repair_required_indices_res] = X_train_OG_np[Repair_required_indices_res] # Use X_train_OG_np
                
                    current_number_missing_iter = np.isnan(X_train_filled_iter).any(axis=1).sum() 
                    Repair_total_iter += len(Repair_required_indices_res)

                    print("Repair total: ", Repair_total_iter)
                    print("missing value initial: ", missing_value_initial_iter)

                    if i_loop == 0 or (i_loop + 1) % 3 == 0:
                        acc_start_iter = time.time() 
                        train_accs_iter_eval, test_accs_iter_eval, durs_iter_eval = [], [], [] 
                        
                        mask_iter_eval = ~np.isnan(X_train_filled_iter).any(axis=1) 
                        X_train_clean_iter_eval = X_train_filled_iter[mask_iter_eval] 
                        Y_train_clean_iter_eval = Y_train_filled_iter[mask_iter_eval] 
                        
                        if X_train_clean_iter_eval.shape[0] > 0: 
                            for j_loop in range(0, 10): 
                                train_acc, test_acc, duration = SGD_class(X_train_clean_iter_eval, Y_train_clean_iter_eval, X_test_np, Y_test_np)
                                train_accs_iter_eval.append(train_acc)
                                test_accs_iter_eval.append(test_acc)
                                durs_iter_eval.append(duration) 
                            top_k_accs = sorted(test_accs_iter_eval, reverse=True)[:]
                            avg_test_acc_iter_eval = np.mean(top_k_accs) 
                            std_test_acc_iter_eval = np.std(top_k_accs)   
                            acc_end_iter = time.time() 

                            acc_time_iter = acc_end_iter - acc_start_iter 
                            accuracy_time_total_iter += acc_time_iter
                            current_time_iter = time.time() - iter_loop_start_time - accuracy_time_total_iter 
                            print(f"Current time: {current_time_iter:.4f} seconds")  
                            print(f"Average test accuracy: {avg_test_acc_iter_eval:.4f}")
                            print(f"Test accuracy standard deviation: {std_test_acc_iter_eval:.4f}")
                        else:
                            print(f"Iter {i_loop+1}: No non-NaN data to evaluate.")

                    if Repair_total_iter >= missing_value_initial_iter or len(Repair_required_indices_res) == 0:
                        print("Repair total: ", Repair_total_iter)
                        print("Stopping condition met.")
                        break

            elif itter_method_loop == "KNN": 
                for i_loop in range(0,max_iter_loop): 
                    print(f"\n--- Iteration {i_loop+1}/{max_iter_loop} ---")
                    minimal_Repair_res, Repair_required_indices_res, _ = findminimalRepair(
                        X_train_filled_iter, Y_train_filled_iter, seed=seed_iter_loop
                    )
                    print("need to be imputed: ", len(Repair_required_indices_res))
                    num_dropped_sanity = sanity_check(X_train_filled_iter, Repair_required_indices_res)
                    print("number of example dropped: ", num_dropped_sanity)

                    if not Repair_required_indices_res: print("Stopping: No indices to impute."); break

                    if model_knn_main: 
                        X_imputed_full_knn_iter, _ = knn_Repair(X_train_filled_iter.copy(), condition="Transform", imputer=model_knn_main)
                        X_train_filled_iter[Repair_required_indices_res] = X_imputed_full_knn_iter[Repair_required_indices_res]
                    else:
                        print("KNN model (model_knn_main) not available. Stopping KNN iterative method.")
                        break

                    current_number_missing_iter = np.isnan(X_train_filled_iter).any(axis=1).sum()
                    Repair_total_iter += len(Repair_required_indices_res)

                    print("Repair total: ", Repair_total_iter)
                    print("missing value initial: ", missing_value_initial_iter)

                    if Repair_total_iter >= missing_value_initial_iter or len(Repair_required_indices_res) == 0 :
                        print("Repair total: ", Repair_total_iter)
                        print("Stopping condition met.")
                        break
            
            elif itter_method_loop == "MICE": 
                for i_loop in range(0, max_iter_loop): 
                    print(f"\n--- Iteration {i_loop+1}/{max_iter_loop} ---")
                    minimal_Repair_res, Repair_required_indices_res, _ = findminimalRepair(
                        X_train_filled_iter, Y_train_filled_iter, seed=seed_iter_loop)

                    print("need to be imputed: ", len(Repair_required_indices_res))
                    num_dropped_sanity = sanity_check(X_train_filled_iter, Repair_required_indices_res)
                    print("number of example dropped: ", num_dropped_sanity)
                    
                    if not Repair_required_indices_res: print("Stopping: No indices to impute."); break

                    if model_mice_main: 
                        X_imputed_full_mice_iter, _ = mice_impute(X_train_filled_iter.copy(), condition="Transform", imputer=model_mice_main)
                        X_train_filled_iter[Repair_required_indices_res] = X_imputed_full_mice_iter[Repair_required_indices_res]
                    else:
                        print("MICE model (model_mice_main) not available. Stopping MICE iterative method.")
                        break

                    current_number_missing_iter = np.isnan(X_train_filled_iter).any(axis=1).sum()
                    Repair_total_iter += len(Repair_required_indices_res)

                    print("Repair total: ", Repair_total_iter)
                    print("missing value initial: ", missing_value_initial_iter)

                    if Repair_total_iter >= missing_value_initial_iter or len(Repair_required_indices_res) == 0:
                        print("Repair total: ", Repair_total_iter)
                        print("Stopping condition met.")
                        break


            iterative_GT_time_final = time.time() - iter_loop_start_time - accuracy_time_total_iter 
            print(f"Time to finish iteration: {iterative_GT_time_final:.4f} seconds")

            final_mask_iter = ~np.isnan(X_train_filled_iter).any(axis=1) 
            X_train_clean_final_iter = X_train_filled_iter[final_mask_iter] 
            Y_train_clean_final_iter = Y_train_filled_iter[final_mask_iter] 
            # save_clean_train_data(X_train_clean_final_iter, Y_train_clean_final_iter, seed_iter_loop, missingness, name)  # Uses 'name'

            if X_train_clean_final_iter.shape[0] > 0: 
                train_accs_final_iter, test_accs_final_iter, durs_final_iter = [], [], [] 
                for i_final_eval in range(0, 10): 
                    train_acc, test_acc, duration = SGD_class(X_train_clean_final_iter, Y_train_clean_final_iter, X_test_np, Y_test_np)
                    train_accs_final_iter.append(train_acc)
                    test_accs_final_iter.append(test_acc)
                    durs_final_iter.append(duration)
                top_k_accs_test = sorted(test_accs_final_iter, reverse=True)[:]
                top_k_accs_train = sorted(train_accs_final_iter, reverse=True)[:]
                avg_test_acc_final_iter = np.mean(top_k_accs_test) 
                std_test_acc_final_iter = np.std(top_k_accs_test)

                avg_train_acc_final_iter = np.mean(top_k_accs_train)
                std_train_acc_final_iter = np.std(top_k_accs_train) 

                avg_duration_final_iter = np.mean(durs_final_iter)       
                std_duration_final_iter = np.std(durs_final_iter)       

                print(f"\nAverage train accuracy: {avg_train_acc_final_iter:.4f}")
                print(f"Average test accuracy: {avg_test_acc_final_iter:.4f}")
                print(f"Test accuracy standard deviation: {std_test_acc_final_iter:.4f}")
                print(f"\nAverage duration: {avg_duration_final_iter:.4f}")
                print(f"Duration standard deviation: {std_duration_final_iter:.4f}")

                total_time_final_iter = iterative_GT_time_final + avg_duration_final_iter 

                print(f"Total time itterative GT-imputed: {total_time_final_iter:.4f}")
                print("\n\n")
            else:
                print("Final Iterative Repair: No non-NaN data to evaluate.\n\n")
