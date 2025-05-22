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
from datetime import datetime
from sklearn import svm
# Generate random repairs and replace with edge repairs if available
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--itter_data_loop", type=str, default="breast", help="Repair method to use in iteration loop")
args = parser.parse_args()
itter_data_loop = args.itter_data_loop


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
    start_time = time.time() 
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


def SGD_class(X_train, Y_train, X_test, Y_test, iter=10000, tolerance=1e-3, seed=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SGDClassifier(loss='hinge', max_iter=iter, tol=tolerance, random_state=seed)
    start = time.time()
    model.fit(X_train_scaled, Y_train)
    duration = time.time() - start


    train_accuracy = model.score(X_train_scaled, Y_train)
    test_accuracy = model.score(X_test_scaled, Y_test)

    return train_accuracy, test_accuracy, duration


def SVC_class(X_train, Y_train, X_test, Y_test, iter=10000, tolerance=1e-4, seed=None):
    # Generate random seed if not provided
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM with SGD
    model = svm.LinearSVC(penalty="l2", loss="hinge",random_state=seed, C=0.01)
    start = time.time()
    model.fit(X_train_scaled, Y_train)
    duration = time.time() - start
    # print(f"Training time: {duration:.2f} seconds")

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

    # Apply Cleaning to the Initial Batch

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
    AC_records_2, AC_score_2 =activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_3, AC_score_3 =activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_4, AC_score_4 =activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_5, AC_score_5 =activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )

    end_time = time.time()

    elapsed_time = end_time - start_time
    AC_time = elapsed_time / 5

    AC_records = (
                        AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5
                 ) / 5
    AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5

    return AC_records, AC_score, AC_time



#----------------------------------------------------Utility Function---------------------------------------------------
def sanity_check(X, minimal_Repair_examples):
    missing_rows = np.where(np.isnan(X).any(axis=1))[0]

    examples_drop = [example for example in missing_rows if example not in minimal_Repair_examples]
    return len(examples_drop)


def make_dirty(df, random_seed, missing_factor, dirty_cols=1):
    np.random.seed(random_seed)
    num_dirty_cols = dirty_cols 

    num_rows = df.shape[0]
    num_dirty_rows = int(missing_factor * num_rows)

    selected_dirty_cols = np.random.choice(df.columns[:-1], num_dirty_cols, replace=False)

    dirty_rows = np.random.choice(df.index, num_dirty_rows, replace=False)

    df_dirty = df.copy()

    df_dirty.loc[dirty_rows, selected_dirty_cols] = np.nan

    return df_dirty


#############################################################################################
#                                                                                           #
#                                   MAIN STARTS HERE                                        #
#                                                                                           #
#############################################################################################


if __name__ == '__main__':
    print("Run started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    seeds_to_try = [171]

    DATASET_CHOICE = itter_data_loop
    if DATASET_CHOICE == "breast":
        print("----------------------------------breast--------------------------------------")
        file_path = './MI/SVM/real/data/original/breast.csv'
        OG_train_path = './MI/SVM/real/data/original/df_train_breast'
        OG_test_path = './MI/SVM/real/data/original/df_test_breast'
        df = pd.read_csv(file_path).iloc[:, 1:]

    elif DATASET_CHOICE == "water":
        print("-----------------------------------water-------------------------------------------")
        file_path = './MI/SVM/real/data/original/water.csv'
        df = pd.read_csv(file_path, header=0)

    elif DATASET_CHOICE == "online":
        print("-----------------------------------online education-------------------------------------------")
        df = pd.read_csv("./MI/SVM/real/data/original/online.csv", header=0)

    elif DATASET_CHOICE == "bankrupt":
        print("----------------------------------bankrupt--------------------------------------")
        df = pd.read_csv('./MI/SVM/real/data/original/bankrupt.csv', header=0)

    else:
        raise ValueError(f"Unknown dataset choice: {DATASET_CHOICE}")




    #--------------------------------------Preprocessing-------------------------------
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    test_data = test_data.dropna()

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    print(train_data.shape)

    #-----------------------------Initialization---------------------------
    label_col = train_data.columns[-1]

    # Separate features and labels
    X_train = train_data.drop(columns=[label_col])
    y_train = train_data[label_col]

    X_test = test_data.drop(columns=[label_col])
    y_test = test_data[label_col]


    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_data_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=train_data.index)
    train_data_scaled[label_col] = y_train

    test_data_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=test_data.index)
    test_data_scaled[label_col] = y_test



    X_train, Y_train = get_Xy(train_data) 
    X_test, Y_test = get_Xy(test_data) 
    #-----------------------------Info about the dataset---------------------------
    total_examples = len(X_train)
    rows_with_missing_values = np.isnan(X_train).any(axis=1).sum()
    missing_factor = rows_with_missing_values / total_examples

    print("Number of rows with missing values:", rows_with_missing_values)
    print(f"Total example {X_train.shape}, MISSING FACTOR: {missing_factor}")

    #-----------------------------Initialize saving parameter---------------------------
    min_time = float('inf')
    seed = None
    number_of_example_dropped = None
    test_set = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #-----------------------------Active Clean------------------------------------------
    examples_cleaned_AC, accuracy_AC ,training_time_AC = active_clean_driver(train_data_scaled, test_data_scaled)
    print(f"Accuracy_AC: {accuracy_AC}") 
    print(f"Examples Cleaned AC: {examples_cleaned_AC}")
    print(f"Training_Time_AC: {training_time_AC}\n")


    #-------------------------------Calculate Imputing All------------------------------


    train_mask_complete = ~np.isnan(X_train).any(axis=1)
    test_mask_complete  = ~np.isnan(X_test).any(axis=1)

    # Split training data
    X_tr_drop = X_train[train_mask_complete]
    Y_tr_drop = Y_train[train_mask_complete]
    X_tr_missing  = X_train[~train_mask_complete]
    Y_tr_missing  = Y_train[~train_mask_complete]

    print(Y_tr_drop.shape)
    cheating_class = np.concatenate([X_tr_drop, Y_tr_drop.reshape(-1, 1)], axis=1)
    cheating_missing = np.concatenate([X_tr_missing, Y_tr_missing.reshape(-1, 1)], axis=1)
    imputer_test = IterativeImputer(max_iter=2, random_state=51, skip_complete=False)
    imputer_test.fit(cheating_class)
    cheating_missing_complete = imputer_test.transform(cheating_missing)
    full_data = np.concatenate([cheating_class, cheating_missing_complete], axis=0)

    X_tr = full_data[:, :-1]
    Y_tr = full_data[:, -1]
    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=2)
    model.fit(X_tr, Y_tr)
    accuracy_KNN = model.score(X_test, Y_test)
    print(accuracy_KNN)





    #-------------------------------Using KNN----------------------
    print("--------------------Evaluating SGDClassifier on KNN Impute Data--------------------------")

    train_mask_complete = ~np.isnan(X_train).any(axis=1)
    test_mask_complete  = ~np.isnan(X_test).any(axis=1)

    X_train_drop     = X_train[train_mask_complete]
    Y_train_drop     = Y_train[train_mask_complete]
    X_train_missing  = X_train[~train_mask_complete]
    Y_train_missing  = Y_train[~train_mask_complete]

    X_test_drop      = X_test[test_mask_complete]
    Y_test_drop      = Y_test[test_mask_complete]


    model_knn, train_time = knn_Repair(X_train_drop, condition="Fit")


    X_train_imputed, infer_time_train = knn_Repair(X_train_missing, condition="Transform", imputer=model_knn)

    X_train_KNN = np.concatenate([X_train_drop, X_train_imputed], axis=0)
    Y_train_KNN = np.concatenate([Y_train_drop, Y_train_missing], axis=0)

    X_test_KNN = X_test_drop.copy()
    Y_test_KNN = Y_test_drop.copy()

    train_accuracies = []
    test_accuracies = []
    durations = []

    for i in range(0, 10):
        train_acc, test_acc, duration = SGD_class(X_train_KNN, Y_train_KNN, X_test, Y_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        durations.append(duration)

    avg_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies)
    avg_train_acc = np.mean(train_accuracies)
    avg_duration = np.mean(durations)
    std_duration = np.std(durations)

    total_time = train_time + infer_time_train + avg_duration

    print(f"\nAverage train accuracy: {avg_train_acc:.4f}")
    print(f"Average test accuracy: {avg_test_acc:.4f}")
    print(f"Test accuracy standard deviation: {std_test_acc:.4f}")
    print(f"\nAverage duration: {avg_duration:.4f}")
    print(f"Duration standard deviation: {std_duration:.4f}")

    print(f"Total time using KNN-imputed data: {total_time:.4f}")


    #-------------------------------Using MICE----------------------
    print("--------------------Evaluating SGDClassifier on MICE Impute Data--------------------------")

    train_mask_complete = ~np.isnan(X_train).any(axis=1)
    test_mask_complete  = ~np.isnan(X_test).any(axis=1)

    X_train_drop     = X_train[train_mask_complete]
    Y_train_drop     = Y_train[train_mask_complete]
    X_train_missing  = X_train[~train_mask_complete]
    Y_train_missing  = Y_train[~train_mask_complete]

    X_test_drop      = X_test[test_mask_complete]
    Y_test_drop      = Y_test[test_mask_complete]


    model_mice, train_time = mice_impute(X_train_drop, condition="Fit")

    X_train_imputed, infer_time = mice_impute(X_train_missing, condition="Transform", imputer=model_mice)


    X_train_mice = np.concatenate([X_train_drop, X_train_imputed], axis=0)
    Y_train_mice = np.concatenate([Y_train_drop, Y_train_missing], axis=0)

    X_test_mice = X_test_drop.copy()
    Y_test_mice = Y_test_drop.copy()

    train_accuracies = []
    test_accuracies = []
    durations = []

    for i in range(0, 10):
        train_acc, test_acc, duration = SGD_class(X_train_mice, Y_train_mice, X_test, Y_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        durations.append(duration)

    avg_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies)
    avg_train_acc = np.mean(train_accuracies)
    avg_duration = np.mean(durations)
    std_duration = np.std(durations)


    print(f"\nAverage train accuracy: {avg_train_acc:.4f}")
    print(f"Average test accuracy: {avg_test_acc:.4f}")
    print(f"Test accuracy standard deviation: {std_test_acc:.4f}")
    print(f"\nAverage duration: {avg_duration:.4f}")
    print(f"Duration standard deviation: {std_duration:.4f}")

    total_time = train_time + infer_time_train+ avg_duration

    print(f"Total time using MICE-imputed data: {total_time:.4f}")

    
    for seed in seeds_to_try:
        print(f"--------------------------------------------- Seed: {seed} ------------------------------------------")
        X_train_filled = X_train.copy() 
        Y_train_filled = Y_train.copy() 
        start_time = time.time()
        accuracy_time_total = 0
        Repair_total = 0
        missing_value_initial = rows_with_missing_values
        max_iter = 10 # YOU COULD PLAY WITH THIS
        itter_method = "KNN" # CHANGE THIS FOR DIFFERENT VERSION

        print(f"---------------------------------- itteration method: {itter_method}---------------------------------")

        if itter_method == "KNN":
            for i in range(0,max_iter):
                print(f"\n--- Iteration {i+1}/{max_iter} ---")
                minimal_Repair, Repair_required_examples_index, index_and_edge_repair = findminimalRepair(X_train_filled, Y_train_filled, seed=seed)
                print("need to be imputed: ", len(Repair_required_examples_index))
                number_of_examples_dropped = sanity_check(X_train_filled, Repair_required_examples_index)
                print("number of example dropped: ", number_of_examples_dropped)

                X_imputed_full, _ = knn_Repair(X_train_filled, condition="Transform", imputer=model_knn)
                X_train_filled[Repair_required_examples_index] = X_imputed_full[Repair_required_examples_index]

                current_number_missing = np.isnan(X_train_filled).any(axis=1).sum()
                Repair_total += len(Repair_required_examples_index)

                print("Repair total: ", Repair_total)
                print("missing value initial: ", missing_value_initial)

                if i == 0 or (i + 1) % 3 == 0:
                    acc_start = time.time()
                    train_accuracies = []
                    test_accuracies = []
                    mask = ~np.isnan(X_train_filled).any(axis=1)
                    X_train_clean = X_train_filled[mask]
                    Y_train_clean = Y_train_filled[mask]
                    for j in range(0, 10):
                        train_acc, test_acc, duration = SGD_class(X_train_clean, Y_train_clean, X_test, Y_test)
                        train_accuracies.append(train_acc)
                        test_accuracies.append(test_acc)
                        durations.append(duration)
                    avg_test_acc = np.mean(test_accuracies)
                    std_test_acc = np.std(test_accuracies)
                    acc_end = time.time()

                    acc_time = acc_end - acc_start
                    accuracy_time_total += acc_time
                    current_time = time.time() - start_time - accuracy_time_total
                    print(f"Current time: {current_time:.4f} seconds")  
                    print(f"Average test accuracy: {avg_test_acc:.4f}")
                    print(f"Test accuracy standard deviation: {std_test_acc:.4f}")

                if Repair_total == missing_value_initial or len(Repair_required_examples_index) == 0:
                    print("Repair total: ", Repair_total)
                    print("Stopping condition met.")
                    break

        elif itter_method == "MICE":
            for i in range(0, max_iter):
                print(f"\n--- Iteration {i+1}/{max_iter} ---")
                minimal_Repair, Repair_required_examples_index, index_and_edge_repair = findminimalRepair(
                    X_train_filled, Y_train_filled, seed=seed)

                print("need to be imputed: ", len(Repair_required_examples_index))
                number_of_examples_dropped = sanity_check(X_train_filled, Repair_required_examples_index)
                print("number of example dropped: ", number_of_examples_dropped)

                X_imputed_full, _ = mice_impute(X_train_filled, condition="Transform", imputer=model_mice)
                X_train_filled[Repair_required_examples_index] = X_imputed_full[Repair_required_examples_index]

                current_number_missing = np.isnan(X_train_filled).any(axis=1).sum()
                Repair_total += len(Repair_required_examples_index)

                print("Repair total: ", Repair_total)
                print("missing value initial: ", missing_value_initial)

                if i == 0 or (i + 1) % 3 == 0:
                    acc_start = time.time()
                    train_accuracies = []
                    test_accuracies = []
                    mask = ~np.isnan(X_train_filled).any(axis=1)
                    X_train_clean = X_train_filled[mask]
                    Y_train_clean = Y_train_filled[mask]
                    for j in range(0, 10):
                        train_acc, test_acc, duration = SGD_class(X_train_clean, Y_train_clean, X_test, Y_test)
                        train_accuracies.append(train_acc)
                        test_accuracies.append(test_acc)
                        durations.append(duration)
                    avg_test_acc = np.mean(test_accuracies)
                    std_test_acc = np.std(test_accuracies)
                    acc_end = time.time()

                    acc_time = acc_end - acc_start
                    accuracy_time_total += acc_time
                    current_time = time.time() - start_time - accuracy_time_total
                    print(f"Current time: {current_time:.4f} seconds")  
                    print(f"Average test accuracy: {avg_test_acc:.4f}")
                    print(f"Test accuracy standard deviation: {std_test_acc:.4f}")                

                if Repair_total == missing_value_initial or len(Repair_required_examples_index) == 0:
                    print("Repair total: ", Repair_total)
                    print("Stopping condition met.")
                    break

        
        iterative_time = time.time() - start_time
        print(f"Time to finish iteration: {iterative_time:.4f} seconds")

        mask = ~np.isnan(X_train_filled).any(axis=1)
        X_train_clean = X_train_filled[mask]
        Y_train_clean = Y_train_filled[mask]
        for j in range(0, 10):
            if itter_method == "KNN":
                train_acc, test_acc, duration = SGD_class(X_train_clean, Y_train_clean, X_test_KNN, Y_test_KNN)
            elif itter_method == "MICE":
                train_acc, test_acc, duration = SGD_class(X_train_clean, Y_train_clean, X_test_mice, Y_test_mice)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            durations.append(duration)
        
        print(f"\nAverage train accuracy: {avg_train_acc:.4f}")
        print(f"Average test accuracy: {avg_test_acc:.4f}")
        print(f"Test accuracy standard deviation: {std_test_acc:.4f}")
        print(f"\nAverage duration: {avg_duration:.4f}")
        print(f"Duration standard deviation: {std_duration:.4f}")   

        total_time = iterative_time + avg_duration

        # total_time = iterative_time + duration #NEED TO ADD THE INFER TIME OF TEST FOREACH METHOD
        # print(f"Train accuracy using itterative GT-imputed data: {train_acc:.4f}")
        # print(f"Test accuracy using itterative GT-imputed data: {test_acc:.4f}")
        print(f"Total time itterative imputed: {total_time:.4f}")
        print("\n\n")





