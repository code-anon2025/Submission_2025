import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import time
import os
from sklearn.impute import SimpleImputer, KNNImputer
from hyperimpute.plugins.imputers import Imputers
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cp
import csv
import math
import inspect

def get_Xy_svm(data, label_column_name):
    if label_column_name is None:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    else:
        X = data.drop(label_column_name, axis=1).values
        y = data[label_column_name].values
    return X, y

def make_dirty_svm(df, label_column_name, random_seed, missing_factor):
    np.random.seed(random_seed)
    df_dirty = df.copy()
    if label_column_name in df.columns:
        feature_columns = df.drop(label_column_name, axis=1).columns
    else:
        feature_columns = df.columns
    if not feature_columns.any():
        return df_dirty
    num_cols_to_select = min(1, len(feature_columns))
    if num_cols_to_select < 1:
         return df_dirty
    dirty_col_indices = np.random.choice(len(feature_columns), num_cols_to_select, replace=False)
    dirty_cols = feature_columns[dirty_col_indices]
    num_rows = df.shape[0]
    num_dirty_rows = int(missing_factor * num_rows)
    if num_dirty_rows > 0 and num_rows > 0:
        num_dirty_rows = min(num_dirty_rows, num_rows)
        dirty_row_indices = np.random.choice(df.index, num_dirty_rows, replace=False)
        df_dirty.loc[dirty_row_indices, dirty_cols] = np.nan
    elif num_rows == 0:
         pass
    return df_dirty

def generate_sample_edge_repairs_svm(X_current_state, mve_indices, num_samples,
                                      col_min_full, col_max_full, random_seed=None):
    if random_seed is not None: np.random.seed(random_seed)
    if not mve_indices or num_samples == 0:
        return [X_current_state.copy()] * num_samples if num_samples > 0 else []
    n_examples, n_features = X_current_state.shape
    sample_repairs = []
    missing_locations = {}
    for ex_idx in mve_indices:
        if 0 <= ex_idx < n_examples:
            row_data = X_current_state[ex_idx]
            if hasattr(row_data, '__iter__'):
                 missing_locations[ex_idx] = [
                     feat_idx for feat_idx in range(n_features)
                     if np.isnan(row_data[feat_idx])
                 ]
            else:
                 missing_locations[ex_idx] = []
    for _ in range(num_samples):
        repaired_dataset = X_current_state.copy()
        for ex_idx in mve_indices:
            if not (0 <= ex_idx < n_examples): continue
            if ex_idx in missing_locations:
                for feat_idx in missing_locations[ex_idx]:
                    if not (0 <= feat_idx < n_features and feat_idx < len(col_min_full) and feat_idx < len(col_max_full)):
                        min_v, max_v = 0,1
                    else:
                        min_v, max_v = col_min_full[feat_idx], col_max_full[feat_idx]
                    if np.isnan(min_v) or np.isnan(max_v): min_v, max_v = 0, 1
                    repaired_dataset[ex_idx, feat_idx] = np.random.choice([min_v, max_v])
        sample_repairs.append(repaired_dataset)
    return sample_repairs

def calculate_svm_loss_subgradient(w, X_repair, y_repair, C_svm, fit_intercept=False):
    n_samples, n_features_in_data = X_repair.shape
    expected_features = w.shape[0] - 1 if fit_intercept else w.shape[0]
    if n_samples == 0:
        return np.inf, np.zeros(expected_features + (1 if fit_intercept else 0))
    if n_features_in_data != expected_features:
         raise ValueError(f"Shape mismatch in calculate_svm_loss_subgradient: w implies {expected_features} features, data has {n_features_in_data}")
    w_feat = w[:-1] if fit_intercept else w
    b = w[-1] if fit_intercept else 0.0
    if y_repair.ndim > 1: y_repair = y_repair.ravel()
    margins = y_repair * (X_repair @ w_feat + b)
    hinge_loss_terms = np.maximum(0, 1 - margins)
    loss = 0.5 * (w_feat @ w_feat) + C_svm * np.sum(hinge_loss_terms)
    sv_indices = np.where(margins <= 1)[0]
    subgradient_w = w_feat.copy()
    subgradient_b = 0.0
    if sv_indices.size > 0:
        subgradient_w -= C_svm * np.sum(y_repair[sv_indices, np.newaxis] * X_repair[sv_indices], axis=0)
        if fit_intercept:
            subgradient_b = -C_svm * np.sum(y_repair[sv_indices])
    return loss, (np.append(subgradient_w, subgradient_b) if fit_intercept else subgradient_w)

def solve_svm_model(X_repair, y_repair, C_svm, random_seed=None, fit_intercept=False, max_iter=1000000, tol=1e-9):
    n_samples, n_features = X_repair.shape
    if n_samples == 0 or (n_samples > 0 and len(np.unique(y_repair)) < 2) :
        return np.zeros(n_features + (1 if fit_intercept else 0)), np.inf
    alpha_sgd = 1.0 / (C_svm * n_samples) if C_svm > 0 and n_samples > 0 else 1e-4
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha_sgd,
                          max_iter=max_iter, tol=tol, fit_intercept=fit_intercept,
                          random_state=random_seed, warm_start=False,
                          learning_rate='optimal', class_weight=None)
    try:
        X_clean = X_repair.copy()
        if np.isnan(X_clean).any():
             X_clean = np.nan_to_num(X_clean)
        model.fit(X_clean, y_repair)
        w_star_feat = model.coef_[0]
        w_star_b = model.intercept_[0] if fit_intercept else 0.0
        w_star = np.append(w_star_feat, w_star_b) if fit_intercept else w_star_feat
        L_star, _ = calculate_svm_loss_subgradient(w_star, X_clean, y_repair, C_svm, fit_intercept)
    except ValueError as e:
        return np.zeros(n_features + (1 if fit_intercept else 0)), np.inf
    return w_star, L_star

def calculate_max_margin_svm(w_candidate, original_incomplete_example, y_j, col_min_full, col_max_full, fit_intercept=False):
    x_j_best_margin_repair = original_incomplete_example.copy()
    w_feat = w_candidate[:-1] if fit_intercept else w_candidate
    b = w_candidate[-1] if fit_intercept else 0.0
    n_features_w = len(w_feat)
    if len(original_incomplete_example) != n_features_w:
        raise ValueError(f"Dimension mismatch in calculate_max_margin_svm: example has {len(original_incomplete_example)} features, w_feat implies {n_features_w}")
    for l_idx in range(n_features_w):
        if np.isnan(original_incomplete_example[l_idx]):
            term_yj_wl = y_j * w_feat[l_idx]
            if l_idx >= len(col_min_full) or l_idx >= len(col_max_full):
                min_v, max_v = 0,1
            else:
                min_v, max_v = col_min_full[l_idx], col_max_full[l_idx]
            if np.isnan(min_v) or np.isnan(max_v): min_v, max_v = 0,1
            x_j_best_margin_repair[l_idx] = max_v if term_yj_wl > 0 else min_v
    if np.isnan(x_j_best_margin_repair).any():
       x_j_best_margin_repair = np.nan_to_num(x_j_best_margin_repair)
    if len(w_feat) != len(x_j_best_margin_repair):
         raise ValueError(f"Critical Dimension mismatch before dot product in calculate_max_margin_svm: w_feat {len(w_feat)}, x_j {len(x_j_best_margin_repair)}")
    return y_j * (w_feat @ x_j_best_margin_repair + b)

def impute_examples_svm(X_current_state,
                        example_indices_to_impute,
                        imputation_method,
                        original_dataset_context=None,
                        col_stats=None,
                        fitted_knn_imputer=None,
                        X_original_complete_for_gt=None):
    X_new_state = X_current_state.copy()
    if not example_indices_to_impute:
        return X_new_state
    indices_to_impute_arr = np.array(list(example_indices_to_impute))
    if indices_to_impute_arr.size == 0:
        return X_new_state
    valid_indices = indices_to_impute_arr[indices_to_impute_arr < X_new_state.shape[0]]
    if valid_indices.size == 0:
        return X_new_state
    subset_to_impute = X_new_state[valid_indices]
    if subset_to_impute.size == 0:
        return X_new_state
    imputed_subset = subset_to_impute.copy()
    if imputation_method == 'mean':
        if col_stats is None or 'means' not in col_stats:
            if original_dataset_context is None or original_dataset_context.shape[0] == 0:
                raise ValueError("original_dataset_context (non-empty) is required for mean imputation if col_stats are not provided/valid.")
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(original_dataset_context)
            imputed_subset = imputer.transform(imputed_subset)
        else:
            col_means = col_stats['means']
            for i in range(imputed_subset.shape[0]):
                for feat_idx in range(imputed_subset.shape[1]):
                    if np.isnan(imputed_subset[i, feat_idx]):
                        if feat_idx < len(col_means):
                            imputed_subset[i, feat_idx] = col_means[feat_idx]
                        else:
                            imputed_subset[i, feat_idx] = 0
    elif imputation_method == 'knn':
        if fitted_knn_imputer is not None:
            imputed_subset = fitted_knn_imputer.transform(imputed_subset)
        else:
            print("Warning: fitted_knn_imputer not provided to impute_examples_svm. Falling back to mean imputation for affected subset.")
            if original_dataset_context is None or original_dataset_context.shape[0] == 0:
                raise ValueError("original_dataset_context (non-empty) is required for KNN's mean fallback.")
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(original_dataset_context)
            imputed_subset = imputer.transform(imputed_subset)
    elif imputation_method == 'ground_truth':
        if X_original_complete_for_gt is None:
            raise ValueError("X_original_complete_for_gt is required for 'ground_truth' imputation.")
        if X_original_complete_for_gt.shape[0] != X_current_state.shape[0] or \
           X_original_complete_for_gt.shape[1] != X_current_state.shape[1]:
            print(f"Warning: Shape mismatch between X_current_state ({X_current_state.shape}) and X_original_complete_for_gt ({X_original_complete_for_gt.shape}). Ensure correct GT data is passed.")
        for i, original_row_idx in enumerate(valid_indices):
            if original_row_idx < X_original_complete_for_gt.shape[0]:
                for feat_idx in range(imputed_subset.shape[1]):
                    if np.isnan(imputed_subset[i, feat_idx]):
                        if feat_idx < X_original_complete_for_gt.shape[1]:
                            imputed_subset[i, feat_idx] = X_original_complete_for_gt[original_row_idx, feat_idx]
    else:
        raise ValueError(f"Unsupported imputation method: {imputation_method}")
    if imputed_subset is not None:
        X_new_state[valid_indices] = imputed_subset
    return X_new_state

def approximate_BB1_svm_cvxpy(X_current_state, y_labels, mve_indices, C_svm,
                              s_opt_sample_size,
                              col_min_full, col_max_full,
                              fit_intercept=False, random_seed=None, solver=None, verbose=False):
    num_features = X_current_state.shape[1] if X_current_state.ndim > 1 else (X_current_state.shape[0] if X_current_state.size > 0 else 0)
    if not mve_indices:
        w_final = np.zeros(num_features + (1 if fit_intercept else 0))
        g_k_est = 0.0
        L_final_val = 0.0
        X_final_state = X_current_state.copy()
        if np.isnan(X_final_state).any():
            X_final_state = np.nan_to_num(X_final_state)
        if X_final_state.shape[0] > 0 and len(np.unique(y_labels)) >=2:
            w_final_cand, L_final_val_cand = solve_svm_model(X_final_state, y_labels, C_svm, random_seed, fit_intercept)
            if L_final_val_cand != np.inf:
                 w_final, L_final_val = w_final_cand, L_final_val_cand
        return w_final, g_k_est, [X_current_state.copy()], {0: L_final_val if L_final_val != np.inf else 0.0}

    X_E_opt = generate_sample_edge_repairs_svm(
        X_current_state, mve_indices, s_opt_sample_size,
        col_min_full, col_max_full, random_seed
    )
    if not X_E_opt:
        return np.zeros(num_features + (1 if fit_intercept else 0)), np.inf, [], {}
    
    L_stars_sample = {}
    valid_repairs_indices = []
    valid_X_e_list = []
    for i, X_e_candidate in enumerate(X_E_opt):
        X_e_for_L_star = X_e_candidate.copy()
        if np.isnan(X_e_for_L_star).any(): # Skip if repair still has NaNs, L* would be inf
            # This can happen if generate_sample_edge_repairs_svm produces NaNs due to all-NaN columns
            # or if col_min/max were NaN and defaulted to 0,1 but random choice hit NaN (unlikely with np.random.choice([0,1]))
            # A more robust generate_sample_edge_repairs_svm might be needed or nan_to_num here.
            # For now, we skip to avoid issues with solve_svm_model if it can't handle internal NaNs for L* calc.
            # Alternatively, nan_to_num before solve_svm_model:
            # X_e_for_L_star = np.nan_to_num(X_e_for_L_star)
            continue # Skipping problematic repair for L* calculation
        
        _, L_star_e = solve_svm_model(X_e_for_L_star, y_labels, C_svm, random_seed, fit_intercept)
        if L_star_e != np.inf:
            L_stars_sample[i] = L_star_e
            valid_repairs_indices.append(i)
            # For CVXPY, ensure data is complete. If X_e_for_L_star was already nan_to_num'd, this is fine.
            # If not, and it was skipped above, it won't be in valid_X_e_list.
            # If it passed solve_svm_model, it should be complete.
            valid_X_e_list.append(X_e_for_L_star) 
    
    if not valid_repairs_indices: # No valid repairs for which L* could be computed
         return np.zeros(num_features + (1 if fit_intercept else 0)), np.inf, X_E_opt, L_stars_sample

    w_var_feat = cp.Variable(num_features, name="w_features")
    b_var = cp.Variable(name="bias") if fit_intercept else 0.0
    t_var = cp.Variable(name="t_sup_gap")
    constraints = []
    
    for idx_in_valid_list, original_repair_idx in enumerate(valid_repairs_indices):
        X_e_cvx = valid_X_e_list[idx_in_valid_list] # This X_e_cvx should be complete
        y_e_cvx = y_labels
        margins = cp.multiply(y_e_cvx, (X_e_cvx @ w_var_feat + b_var))
        hinge_loss_terms = cp.sum(cp.pos(1 - margins))
        svm_loss_i = 0.5 * cp.sum_squares(w_var_feat) + C_svm * hinge_loss_terms
        L_star_e_i = L_stars_sample[original_repair_idx]
        constraints.append(svm_loss_i - L_star_e_i <= t_var) # h(w, X^e) <= t
    
    if not constraints: # Should not happen if valid_repairs_indices is not empty
         return np.zeros(num_features + (1 if fit_intercept else 0)), np.inf, X_E_opt, L_stars_sample

    objective = cp.Minimize(t_var)
    problem = cp.Problem(objective, constraints)
    w_approx_k_out = np.zeros(num_features + (1 if fit_intercept else 0))
    g_k_est_out = np.inf
    try:
        solve_args = {'solver': solver} if solver else {}
        if verbose: solve_args['verbose'] = True
        problem.solve(**solve_args)
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            w_val_feat = w_var_feat.value if w_var_feat.value is not None else np.zeros(num_features)
            b_val = 0.0
            if fit_intercept and hasattr(b_var, 'value') and b_var.value is not None:
                b_val = b_var.value
            w_approx_k_out = np.append(w_val_feat, b_val) if fit_intercept else w_val_feat
            g_k_est_out = t_var.value if t_var.value is not None else np.inf
        else:
            print(f"  BB1_SVM_CVXPY (MinSupH): CVXPY solver failed or problem status not optimal: {problem.status}")
    except Exception as e:
        print(f"  BB1_SVM_CVXPY (MinSupH): Error during CVXPY solve: {e}.")
    
    if np.isnan(w_approx_k_out).any():
         w_approx_k_out = np.zeros(num_features + (1 if fit_intercept else 0))
         g_k_est_out = np.inf
    return w_approx_k_out, g_k_est_out, X_E_opt, L_stars_sample


def approximate_BB2_svm(X_current_state, y_labels, mve_indices, w_approx_k, C_svm,
                        h_selection_size, epsilon_prime_grad_threshold,
                        col_min_full, col_max_full,
                        X_E_sample_from_BB1,
                        fit_intercept=False, random_seed=None):
    if w_approx_k is None or np.isnan(w_approx_k).any() or not w_approx_k.any():
        return []
    if not mve_indices: return []
    X_E_grad_problematic = []
    subgradients_problematic_map = {}
    if not X_E_sample_from_BB1:
        return []
    for i, X_e_candidate in enumerate(X_E_sample_from_BB1):
        X_e = X_e_candidate.copy()
        if np.isnan(X_e).any():
             X_e = np.nan_to_num(X_e)
        if X_e.shape[0] != y_labels.shape[0] and X_e.shape[0] > 0: continue
        if X_e.shape[0] == 0: continue
        try:
            _, g_e = calculate_svm_loss_subgradient(w_approx_k, X_e, y_labels, C_svm, fit_intercept)
            norm_g_e = np.linalg.norm(g_e)
            if norm_g_e > epsilon_prime_grad_threshold:
                X_E_grad_problematic.append(X_e)
                subgradients_problematic_map[len(X_E_grad_problematic)-1] = g_e
        except ValueError:
            continue
    if not X_E_grad_problematic:
        return []
    total_potential_grad_reduction_score = {}
    w_feat_bb2 = w_approx_k[:-1] if fit_intercept else w_approx_k
    b_val_bb2 = w_approx_k[-1] if fit_intercept else 0.0
    num_features_model = len(w_feat_bb2)
    for i, X_e_problematic_loop in enumerate(X_E_grad_problematic):
        g_e_current = subgradients_problematic_map[i]
        g_e_feat_current = g_e_current[:-1] if fit_intercept else g_e_current
        if len(g_e_feat_current) != num_features_model: continue
        for original_example_idx in mve_indices:
            if not (0 <= original_example_idx < X_e_problematic_loop.shape[0] and \
                    0 <= original_example_idx < X_current_state.shape[0]): continue
            original_incomplete_example_j = X_current_state[original_example_idx]
            if len(original_incomplete_example_j) != num_features_model: continue
            example_vector_repaired_in_Xe_loop = X_e_problematic_loop[original_example_idx]
            if len(example_vector_repaired_in_Xe_loop) != num_features_model: continue
            y_j = y_labels[original_example_idx]
            current_margin = y_j * (w_feat_bb2 @ example_vector_repaired_in_Xe_loop + b_val_bb2)
            if current_margin < 1:
                try:
                    margin_j_max = calculate_max_margin_svm(w_approx_k, original_incomplete_example_j,
                                                            y_j, col_min_full, col_max_full, fit_intercept)
                    if margin_j_max >= 1:
                        score_j_Xe = g_e_feat_current @ (C_svm * y_j * example_vector_repaired_in_Xe_loop)
                        total_potential_grad_reduction_score[original_example_idx] = \
                            total_potential_grad_reduction_score.get(original_example_idx, 0.0) + score_j_Xe
                except ValueError as e:
                    continue
    if not total_potential_grad_reduction_score:
        return []
    ranked_examples = sorted(total_potential_grad_reduction_score.items(), key=lambda item: item[1], reverse=True)
    S_prime_k_indices = [item[0] for item in ranked_examples[:h_selection_size]]
    return S_prime_k_indices

def findminimalImputation(original_dataset, labels, seed=None,
                          X_test_eval=None, y_test_eval=None,
                          X_initial_dirty_for_eval_context=None,
                          X_original_complete_for_gt_imputation=None,
                          log_file_path=None):
    C_svm = 1.0
    e_acm_threshold = 0.05 
    epsilon_prime_grad_threshold = 0.2
    s_sample_size_bb1 = 20
    h_selection_ratio = 0.02
    imputation_strategy = 'ground_truth'
    max_iterations = 100 
    fit_intercept_svm = False
    cvxpy_solver = None
    cvxpy_verbose = False
    knn_k_for_imputation = 5

    print("Starting Iterative ACM Imputation for Linear SVM (MinSupH strategy)...")
    if seed is not None: np.random.seed(seed)

    X_current = np.array(original_dataset, dtype=float)
    initial_context_for_imputers = np.array(X_initial_dirty_for_eval_context, dtype=float) if X_initial_dirty_for_eval_context is not None else X_current.copy()
    y_labels = np.array(labels, dtype=float).ravel()

    if X_current.shape[0] != y_labels.shape[0]:
        raise ValueError(f"Shape mismatch: X_current has {X_current.shape[0]} samples, y_labels has {y_labels.shape[0]}.")

    log_writer = None
    log_file_handle = None
    if log_file_path:
        try:
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            log_file_handle = open(log_file_path, 'w', newline='')
            log_writer = csv.writer(log_file_handle)
            log_writer.writerow(['iteration', 'time_bb1_s', 'time_bb2_s', 'time_imputation_s', 'time_total_iter_s',
                                 'total_imputed_count', 'eval_accuracy', 'g_k_est_bb1'])
        except IOError as e:
            print(f"Warning: Could not open log file {log_file_path}. Error: {e}. Iterative logging will be disabled.")
            log_writer = None
            if log_file_handle: log_file_handle.close()

    col_min_full, col_max_full = None, None
    col_stats_for_mean = None
    fitted_knn_imputer_once = None

    if X_current.ndim > 1 and X_current.shape[1] == 0 and X_current.shape[0] > 0:
        col_min_full, col_max_full = np.array([]), np.array([])
        col_stats_for_mean = {'means': np.array([])}
    elif X_current.shape[0] == 0:
        if log_file_handle: log_file_handle.close()
        return [], [], np.inf
    else:
        col_min_full = np.nanmin(X_current, axis=0)
        col_max_full = np.nanmax(X_current, axis=0)
        for i in range(X_current.shape[1]):
            if np.isnan(col_min_full[i]): col_min_full[i] = 0.0
            if np.isnan(col_max_full[i]): col_max_full[i] = 1.0
        if initial_context_for_imputers.shape[0] > 0:
            temp_mean_imputer = SimpleImputer(strategy='mean')
            temp_mean_imputer.fit(initial_context_for_imputers)
            col_stats_for_mean = {'means': temp_mean_imputer.statistics_}
            if imputation_strategy == 'knn':
                initially_complete_mask_for_knn = ~np.isnan(initial_context_for_imputers).any(axis=1)
                X_initially_complete_for_knn = initial_context_for_imputers[initially_complete_mask_for_knn]
                knn_fit_context = None
                if X_initially_complete_for_knn.shape[0] >= knn_k_for_imputation + 1 :
                    knn_fit_context = X_initially_complete_for_knn
                elif initial_context_for_imputers.shape[0] > 0:
                    knn_fit_context = initial_context_for_imputers
                if knn_fit_context is not None and knn_fit_context.shape[0] > 0:
                    n_fit_samples_knn = knn_fit_context.shape[0]
                    k_fit_knn = knn_k_for_imputation
                    if n_fit_samples_knn <= k_fit_knn:
                        k_fit_knn = max(1, n_fit_samples_knn - 1 if n_fit_samples_knn > 1 else 1)
                    if k_fit_knn > 0 :
                        fitted_knn_imputer_once = KNNImputer(n_neighbors=k_fit_knn)
                        try:
                            fitted_knn_imputer_once.fit(knn_fit_context)
                        except ValueError as e_knn_fit:
                            print(f"Warning: KNNImputer fit failed: {e_knn_fit}. KNN strategy will fallback to mean for subsets if this imputer is used directly.")
                            fitted_knn_imputer_once = None
                    else:
                        print(f"Warning: Context too small ({n_fit_samples_knn} samples) for KNNImputer. KNN strategy will fallback to mean for subsets.")
                else:
                     print(f"Warning: No suitable context to fit KNNImputer. KNN strategy will fallback to mean for subsets.")
        else:
            num_cols = X_current.shape[1] if X_current.ndim > 1 else (1 if X_current.ndim==1 and X_current.size > 0 else 0)
            col_stats_for_mean = {'means': np.zeros(num_cols)}

    S_iter_ACM_indices = set()
    w_approx_final = None
    g_k_est_final = np.inf
    start_time_total_svm_algo = time.time()

    for k_iter in range(max_iterations):
        iter_start_time = time.time()
        current_mve_indices = [i for i in range(X_current.shape[0]) if np.isnan(X_current[i]).any()]

        if not current_mve_indices:
            print("No more incomplete examples. ACM terminates.")
            if X_current.shape[0] > 0 and len(np.unique(y_labels)) >=2 :
                 w_approx_final, _ = solve_svm_model(np.nan_to_num(X_current), y_labels, C_svm, seed, fit_intercept_svm)
            else:
                num_features = X_current.shape[1] if X_current.ndim > 1 else 0
                w_approx_final = np.zeros(num_features + (1 if fit_intercept_svm else 0))
            g_k_est_final = 0.0
            break

        g_k_est_bb1_current_iter = np.nan
        time_bb1_s_iter = 0.0
        X_E_sample_for_bb2 = []
        try:
            time_start_bb1 = time.time()
            w_approx_k, g_k_est_bb1_current_iter, X_E_sample_for_bb2, _ = approximate_BB1_svm_cvxpy(
                X_current, y_labels, current_mve_indices, C_svm, s_sample_size_bb1,
                col_min_full, col_max_full, fit_intercept_svm,
                random_seed=seed + k_iter if seed is not None else None,
                solver=cvxpy_solver, verbose=cvxpy_verbose
            )
            time_bb1_s_iter = time.time() - time_start_bb1
        except Exception as bb1_err:
            print(f"Error in BB1 at iteration {k_iter}: {bb1_err}. Terminating.")
            break
        
        g_k_est_bb1_current_iter = max(0.0, g_k_est_bb1_current_iter) if g_k_est_bb1_current_iter is not None and not np.isnan(g_k_est_bb1_current_iter) else np.inf
        w_approx_final = w_approx_k
        g_k_est_final = g_k_est_bb1_current_iter

        time_bb2_s_iter = 0.0
        time_imputation_s_iter = 0.0
        S_prime_k_indices = []

        if g_k_est_final is not None and g_k_est_final <= e_acm_threshold:
            print(f"Iteration {k_iter}: ACM condition met (g_k_est = {g_k_est_final:.4f} <= e_acm_threshold = {e_acm_threshold}). Terminating.")
            if log_writer:
                log_writer.writerow([k_iter, time_bb1_s_iter, time_bb2_s_iter, time_imputation_s_iter,
                                     time.time() - iter_start_time,
                                     len(S_iter_ACM_indices), np.nan, g_k_est_final])
                if log_file_handle: log_file_handle.flush()
            break
        
        current_h_selection_size = 0
        if current_mve_indices and h_selection_ratio > 0:
            current_h_selection_size = math.ceil(len(current_mve_indices) * h_selection_ratio)
            current_h_selection_size = max(1, int(current_h_selection_size))
        elif current_mve_indices and h_selection_ratio == 0:
            current_h_selection_size = 0
        
        if current_h_selection_size == 0 and current_mve_indices :
             print(f"Iteration {k_iter}: h_selection_size is 0 due to h_selection_ratio. BB2 will not select examples. Terminating.")
             if log_writer:
                 log_writer.writerow([k_iter, time_bb1_s_iter, time_bb2_s_iter, time_imputation_s_iter,
                                     time.time() - iter_start_time,
                                     len(S_iter_ACM_indices), np.nan, g_k_est_final])
                 if log_file_handle: log_file_handle.flush()
             break
        try:
            time_start_bb2 = time.time()
            S_prime_k_indices = approximate_BB2_svm(
                X_current, y_labels, current_mve_indices, w_approx_k, C_svm,
                current_h_selection_size,
                epsilon_prime_grad_threshold,
                col_min_full, col_max_full,
                X_E_sample_for_bb2,
                fit_intercept_svm,
                random_seed=seed + k_iter if seed is not None else None
            )
            time_bb2_s_iter = time.time() - time_start_bb2
        except Exception as bb2_err:
             print(f"Error in BB2 at iteration {k_iter}: {bb2_err}. Terminating.")
             if log_writer:
                 log_writer.writerow([k_iter, time_bb1_s_iter, time_bb2_s_iter, time_imputation_s_iter,
                                     time.time() - iter_start_time,
                                     len(S_iter_ACM_indices), np.nan, g_k_est_final])
                 if log_file_handle: log_file_handle.flush()
             break

        if not S_prime_k_indices:
            print(f"Iteration {k_iter}: BB2 returned no candidate examples. Terminating (g_k_est {g_k_est_final:.4f}).")
            if log_writer:
                log_writer.writerow([k_iter, time_bb1_s_iter, time_bb2_s_iter, time_imputation_s_iter,
                                     time.time() - iter_start_time,
                                     len(S_iter_ACM_indices), np.nan, g_k_est_final])
                if log_file_handle: log_file_handle.flush()
            break
        try:
            time_start_imputation = time.time()
            X_current = impute_examples_svm(X_current, S_prime_k_indices, imputation_strategy,
                                            original_dataset_context=initial_context_for_imputers,
                                            col_stats=col_stats_for_mean,
                                            fitted_knn_imputer=fitted_knn_imputer_once,
                                            X_original_complete_for_gt=X_original_complete_for_gt_imputation)
            time_imputation_s_iter = time.time() - time_start_imputation
        except Exception as impute_err:
            print(f"Error during imputation at iteration {k_iter}: {impute_err}. Terminating.")
            if log_writer:
                log_writer.writerow([k_iter, time_bb1_s_iter, time_bb2_s_iter, time_imputation_s_iter,
                                     time.time() - iter_start_time,
                                     len(S_iter_ACM_indices), np.nan, g_k_est_final])
                if log_file_handle: log_file_handle.flush()
            break
        S_iter_ACM_indices.update(S_prime_k_indices)
        eval_accuracy_iter = np.nan
        if log_writer and X_test_eval is not None and y_test_eval is not None and X_initial_dirty_for_eval_context is not None:
            X_eval_train_parts = []
            y_eval_train_parts = []
            initially_complete_mask_log = ~np.isnan(X_initial_dirty_for_eval_context).any(axis=1)
            if np.sum(initially_complete_mask_log) > 0:
                X_eval_train_parts.append(X_initial_dirty_for_eval_context[initially_complete_mask_log])
                y_eval_train_parts.append(y_labels[initially_complete_mask_log])
            imputed_indices_list_for_eval_log = sorted(list(S_iter_ACM_indices))
            if imputed_indices_list_for_eval_log:
                valid_imputed_indices_log = [idx for idx in imputed_indices_list_for_eval_log if idx < X_current.shape[0]]
                if valid_imputed_indices_log:
                    X_eval_train_parts.append(X_current[valid_imputed_indices_log])
                    y_eval_train_parts.append(y_labels[valid_imputed_indices_log])
            if X_eval_train_parts:
                X_for_iter_eval_log = np.vstack(X_eval_train_parts)
                y_for_iter_eval_log = np.concatenate(y_eval_train_parts)
                if X_for_iter_eval_log.shape[0] > 0 and len(np.unique(y_for_iter_eval_log)) >= 2:
                    if np.isnan(X_for_iter_eval_log).any():
                        # More robust imputation for logging
                        if np.all(np.isnan(X_for_iter_eval_log), axis=0).any(): # Check for all-NaN columns
                             X_for_iter_eval_log = np.nan_to_num(X_for_iter_eval_log)
                        if np.isnan(X_for_iter_eval_log).any(): # If still NaNs after nan_to_num (e.g. object dtype issues)
                            temp_eval_imputer_log = SimpleImputer(strategy='mean')
                            try:
                                X_for_iter_eval_log = temp_eval_imputer_log.fit_transform(X_for_iter_eval_log)
                            except ValueError: # If imputer fails, final fallback
                                X_for_iter_eval_log = np.nan_to_num(X_for_iter_eval_log)
                    
                    model_iter_eval = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=seed, fit_intercept=fit_intercept_svm)
                    try:
                        model_iter_eval.fit(X_for_iter_eval_log, y_for_iter_eval_log)
                        y_pred_iter_eval = model_iter_eval.predict(X_test_eval)
                        eval_accuracy_iter = accuracy_score(y_test_eval, y_pred_iter_eval)
                    except ValueError as e_fit:
                        print(f"Warning (iter log eval): Could not fit/evaluate SVM at iter {k_iter}. Error: {e_fit}")
            log_writer.writerow([k_iter, time_bb1_s_iter, time_bb2_s_iter, time_imputation_s_iter,
                                 time.time() - iter_start_time,
                                 len(S_iter_ACM_indices), eval_accuracy_iter, g_k_est_final])
            if log_file_handle: log_file_handle.flush()
        if k_iter == max_iterations - 1:
            print("Max iterations reached.")

    end_time_total_svm_algo = time.time()
    total_time_algo = end_time_total_svm_algo - start_time_total_svm_algo
    print(f"Total Iterative ACM algorithm time: {total_time_algo:.2f} seconds")
    print(f"Final estimated gap g_k_est: {g_k_est_final}")

    if log_file_handle:
        log_file_handle.close()
    minimal_imputation_list = []
    minimal_imputation_indices = sorted(list(S_iter_ACM_indices))
    return minimal_imputation_list, minimal_imputation_indices, g_k_est_final

def get_Xy(data, label):
    if label is None:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    else:
        X = data.drop(label, axis=1)
        y = data[label]
    return np.array(X), np.array(y)

def sanity_check(X, minimal_imputation_examples):
    missing_rows = np.where(np.isnan(X).any(axis=1))[0]
    examples_saved_count = 0
    if len(missing_rows) > len(minimal_imputation_examples):
        examples_saved_count = len(missing_rows) - len(minimal_imputation_examples)
    elif not minimal_imputation_examples and len(missing_rows) > 0:
        examples_saved_count = len(missing_rows)
    mve_set = set(missing_rows)
    imputed_set = set(minimal_imputation_examples)
    saved_explicitly = len(mve_set - imputed_set)
    return saved_explicitly

def miwae_imputation(X_train, X_test, y_train, y_test, seed):
    start_time_miwae = time.time()
    method='miwae'
    X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train.copy()
    try:
        plugin = Imputers().get(method)
        imputed_X_df = plugin.fit_transform(X_train_df)
        imputed_X = imputed_X_df.values
    except Exception as e:
        print(f"Warning: MIWAE imputation failed with error: {e}. Falling back to mean imputation.")
        imputer = SimpleImputer(strategy='mean')
        imputed_X = imputer.fit_transform(X_train)
    if pd.DataFrame(imputed_X).isnull().any().any():
        imputer = SimpleImputer(strategy='mean')
        imputed_X = imputer.fit_transform(imputed_X)
    if imputed_X.shape[0] == 0 or len(np.unique(y_train)) < 2:
        return 0,0, time.time() - start_time_miwae
    clf = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    clf.fit(imputed_X, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    f1_miwae = f1_score(y_test, y_pred, zero_division=0)
    miwae_time = time.time() - start_time_miwae
    return score, f1_miwae, miwae_time

def mean_imputation(X_train, X_test, y_train, y_test, seed):
    start_time = time.time()
    if X_train.shape[0] == 0: return 0,0,0
    mean_imputer = SimpleImputer(strategy='mean')
    X_train_mean_imputed = mean_imputer.fit_transform(X_train)
    if X_train_mean_imputed.shape[0] == 0 or len(np.unique(y_train)) < 2:
        return 0,0, time.time() - start_time
    model_mean = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    model_mean.fit(X_train_mean_imputed, y_train)
    y_pred = model_mean.predict(X_test)
    accuracy_MI = model_mean.score(X_test, y_test)
    f1_MI = f1_score(y_test, y_pred, zero_division=0)
    training_time_MI = time.time() - start_time
    return accuracy_MI, f1_MI, training_time_MI

def knn_imputation(X_train, X_test, y_train, y_test, seed):
    start_time = time.time()
    n_neighbors_knn = 5
    if X_train.shape[0] == 0: return 0,0,0
    current_k = n_neighbors_knn
    if X_train.shape[0] <= current_k :
        if X_train.shape[0] <= 1:
             return mean_imputation(X_train, X_test, y_train, y_test, seed)
        current_k = max(1, X_train.shape[0]-1)
    if current_k == 0:
        return mean_imputation(X_train, X_test, y_train, y_test, seed)
    knn_imputer = KNNImputer(n_neighbors=current_k)
    try:
        X_train_knn_imputed = knn_imputer.fit_transform(X_train)
    except ValueError as e:
        return mean_imputation(X_train, X_test, y_train, y_test, seed)
    if X_train_knn_imputed.shape[0] == 0 or len(np.unique(y_train)) < 2:
        return 0,0, time.time() - start_time
    model_knn = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    model_knn.fit(X_train_knn_imputed, y_train)
    y_pred = model_knn.predict(X_test)
    accuracy_KNN = model_knn.score(X_test, y_test)
    f1_KNN = f1_score(y_test, y_pred, zero_division=0)
    training_time_KNN = time.time() - start_time
    return accuracy_KNN, f1_KNN, training_time_KNN

def original_zero_imputation(OG_X_train, X_test, OG_y_train, y_test, seed):
    start_time = time.time()
    X_train_clean = OG_X_train.copy()
    if np.isnan(X_train_clean).any():
        X_train_clean = np.nan_to_num(X_train_clean, nan=0.0)
    if X_train_clean.shape[0] == 0 or len(np.unique(OG_y_train)) < 2:
        return 0,0, time.time() - start_time
    model_OG = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    model_OG.fit(X_train_clean, OG_y_train)
    y_pred = model_OG.predict(X_test)
    accuracy_OG = model_OG.score(X_test, y_test)
    f1_OG = f1_score(y_test, y_pred, zero_division=0)
    training_time_OG = time.time() - start_time
    return accuracy_OG, f1_OG, training_time_OG

def make_dirty(df, random_seed, missing_factor, dirty_cols=1):
    np.random.seed(random_seed)
    num_dirty_cols_param = dirty_cols
    num_rows = df.shape[0]
    num_dirty_rows = int(missing_factor * num_rows)
    df_dirty = df.copy()
    feature_columns_list = list(df.columns)
    if 'label' in feature_columns_list:
        feature_columns_list.remove('label')
    if not feature_columns_list : return df_dirty
    actual_num_dirty_cols = min(num_dirty_cols_param, len(feature_columns_list))
    if actual_num_dirty_cols == 0: return df_dirty
    selected_dirty_cols = np.random.choice(feature_columns_list, actual_num_dirty_cols, replace=False)
    if num_rows > 0 and num_dirty_rows > 0:
        actual_num_dirty_rows = min(num_dirty_rows, num_rows)
        if actual_num_dirty_rows > 0 :
            dirty_rows_indices = np.random.choice(df.index, actual_num_dirty_rows, replace=False)
            df_dirty.loc[dirty_rows_indices, selected_dirty_cols] = np.nan
    return df_dirty

def saved_MI(data_frame, required, missing_factor):
    pass

if __name__ == '__main__':
    rng = np.random.RandomState(42)
    n_samples_svm_example, n_features_svm_example = 200, 10
    X_true_svm_example = rng.rand(n_samples_svm_example, n_features_svm_example) * 2 - 1
    true_coef_svm_example = rng.rand(n_features_svm_example) * 2 - 1
    y_svm_scores_example = X_true_svm_example @ true_coef_svm_example
    y_true_svm_example = np.ones(n_samples_svm_example)
    y_true_svm_example[y_svm_scores_example < np.median(y_svm_scores_example)] = -1
    df_full_svm_example = pd.DataFrame(X_true_svm_example, columns=[f'f{i}' for i in range(n_features_svm_example)])
    df_full_svm_example['label'] = y_true_svm_example
    df_train_full_svm_example, df_test_full_svm_example = train_test_split(df_full_svm_example, test_size=0.25, random_state=42)
    X_train_full_svm_example_np = df_train_full_svm_example.drop('label', axis=1).values
    noise_level_svm_example = 0.3
    df_train_dirty_svm_example = make_dirty_svm(df_train_full_svm_example.copy(), 'label', random_seed=42, missing_factor=noise_level_svm_example)
    X_train_dirty_svm_example, y_train_dirty_svm_example = get_Xy_svm(df_train_dirty_svm_example, 'label')
    X_test_svm_example, y_test_svm_example = get_Xy_svm(df_test_full_svm_example, 'label')
    print(f"Shape of X_train_dirty_svm (example): {X_train_dirty_svm_example.shape}")
    print(f"Number of NaNs in X_train_dirty_svm (example): {np.sum(np.isnan(X_train_dirty_svm_example))}")
    _, imputed_indices_svm_example, final_metric_example = findminimalImputation(
        X_train_dirty_svm_example, y_train_dirty_svm_example, seed=42,
        X_test_eval=X_test_svm_example, y_test_eval=y_test_svm_example,
        X_initial_dirty_for_eval_context=X_train_dirty_svm_example.copy(),
        X_original_complete_for_gt_imputation=X_train_full_svm_example_np
    )
    print(f"\n--- Iterative ACM Linear SVM Results (Example Data) ---")
    print(f"Indices of examples chosen for imputation: {sorted(imputed_indices_svm_example)}")
    print(f"Number of examples imputed: {len(imputed_indices_svm_example)}")
    print(f"Final Estimated Gap (g_k_est): {final_metric_example}")
    if imputed_indices_svm_example is not None:
        X_train_acm_imputed_svm_example = X_train_dirty_svm_example.copy()
        if imputed_indices_svm_example:
            X_train_acm_imputed_svm_example = impute_examples_svm(
                X_train_acm_imputed_svm_example, imputed_indices_svm_example, 'mean',
                original_dataset_context=X_train_dirty_svm_example.copy(),
                X_original_complete_for_gt=X_train_full_svm_example_np
            )
        if np.isnan(X_train_acm_imputed_svm_example).any():
            global_imputer_example = SimpleImputer(strategy='mean')
            X_train_acm_imputed_svm_example = global_imputer_example.fit_transform(X_train_acm_imputed_svm_example)
        if X_train_acm_imputed_svm_example.shape[0] > 0 and len(np.unique(y_train_dirty_svm_example)) >= 2:
            C_param_eval = 1.0
            fit_intercept_eval = False
            alpha_eval = 1.0/(C_param_eval * X_train_acm_imputed_svm_example.shape[0]) if C_param_eval > 0 and X_train_acm_imputed_svm_example.shape[0] > 0 else 1e-4
            model_acm_eval_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha_eval,
                                               max_iter=1000, tol=1e-3,
                                               fit_intercept=fit_intercept_eval, random_state=42)
            model_acm_eval_svm.fit(X_train_acm_imputed_svm_example, y_train_dirty_svm_example)
            y_pred_test_acm = model_acm_eval_svm.predict(X_test_svm_example)
            acc_acm = accuracy_score(y_test_svm_example, y_pred_test_acm)
            f1_acm = f1_score(y_test_svm_example, y_pred_test_acm, zero_division=0)
            print(f"Accuracy of model trained on ACM-imputed data (example, on test set): {acc_acm:.4f}")
            print(f"F1-score of model trained on ACM-imputed data (example, on test set): {f1_acm:.4f}")

    seeds_to_try = [42]
    s_sample_size_bb1 = 20
    h_selection_ratio = 0.02
    fit_intercept_svm = False
    ACM_STRATEGY_FOR_REPORTING = 'ground_truth'

    iter_log_dir = './Minimal-Imputation/Iter_Logs'
    os.makedirs(iter_log_dir, exist_ok=True)
    results_output_dir = './Minimal-Imputation/Synthetic-Final-Results'
    os.makedirs(results_output_dir, exist_ok=True)

    DATASET_NAME = "TUADROMD" 
    col_num_for_make_dirty = 48 # For TUADROMD, make_dirty affects up to 48 columns

    for noise_level in [0.2, 0.4, 0.6]:
        print(f"\n############################### Started Missing Factor : {noise_level} for {DATASET_NAME} Dataset ########################################")
        file_path_current_dataset = './Minimal-Imputation/Synthetic-Datasets/TUANDROMD.csv'
        try:
            df_current_raw = pd.read_csv(file_path_current_dataset)
        except FileNotFoundError:
            print(f"ERROR: {DATASET_NAME} dataset file not found at {file_path_current_dataset}. Skipping this noise level.")
            continue
        df_current = df_current_raw.copy()
        if not df_current.empty:
            last_column_name = df_current.columns[-1]
            df_current[last_column_name] = df_current[last_column_name].replace({'malware': 1, 'goodware': -1})
            df_current[last_column_name] = pd.to_numeric(df_current[last_column_name], errors='coerce').astype(float)
            df_current = df_current.dropna(how='all')
            df_current.rename(columns={last_column_name: 'label'}, inplace=True)
        else:
            print(f"Warning: {DATASET_NAME} DataFrame is empty after loading for noise level {noise_level}. Skipping.")
            continue
        if 'label' not in df_current.columns:
            print(f"ERROR: 'label' column not found in {DATASET_NAME} DataFrame after processing for noise level {noise_level}. Skipping.")
            continue

        X_main_df_orig = df_current.drop('label', axis=1)
        y_main_series_orig = df_current['label']
        if X_main_df_orig.shape[0] < 2 or y_main_series_orig.shape[0] < 2 or len(y_main_series_orig.unique()) < 2 :
            print(f"Not enough data or classes in {DATASET_NAME} dataset after processing for noise level {noise_level}. Skipping.")
            continue
        stratify_opt = y_main_series_orig if len(y_main_series_orig.unique()) > 1 else None
        X_train_orig_main_df, X_test_orig_main_df, y_train_orig_main_series, y_test_orig_main_series = \
            train_test_split(X_main_df_orig, y_main_series_orig, test_size=0.2, random_state=42, stratify=stratify_opt)
        scaler = MinMaxScaler()
        if X_train_orig_main_df.empty:
            print(f"Training data for {DATASET_NAME} is empty after split for noise level {noise_level}. Skipping.")
            continue
        X_train_scaled_main_np = scaler.fit_transform(X_train_orig_main_df.values)
        X_test_scaled_main_np = scaler.transform(X_test_orig_main_df.values)
        df_train_main_scaled = pd.DataFrame(X_train_scaled_main_np, columns=X_train_orig_main_df.columns)
        df_train_main_scaled['label'] = y_train_orig_main_series.values
        df_test_main_scaled = pd.DataFrame(X_test_scaled_main_np, columns=X_test_orig_main_df.columns)
        df_test_main_scaled['label'] = y_test_orig_main_series.values
        OG_X_train_main_np, OG_y_train_main_np = get_Xy(df_train_main_scaled, 'label')
        X_test_eval_np, y_test_eval_np = get_Xy(df_test_main_scaled, 'label')
        OG_df_train_main_for_dirty = df_train_main_scaled.copy()
        df_train_dirty_main_pd = make_dirty(OG_df_train_main_for_dirty, 42, noise_level, dirty_cols=col_num_for_make_dirty)
        X_train_loop_np = df_train_dirty_main_pd.drop('label', axis=1).values
        y_train_loop_np = df_train_dirty_main_pd['label'].values.astype(float)
        total_examples = len(X_train_loop_np)
        if total_examples == 0:
            print(f"No examples in training loop for {DATASET_NAME} at noise {noise_level}. Skipping.")
            continue
        missing_values_per_row = pd.DataFrame(X_train_loop_np).isnull().sum(axis=1)
        rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])
        print(f"Number of rows with missing values in {DATASET_NAME} training data: {rows_with_missing_values}")
        missing_factor_calc = rows_with_missing_values / total_examples if total_examples > 0 else 0
        print(f"Total examples {X_train_loop_np.shape}, Calculated Missing Factor : {missing_factor_calc:.4f}")

        min_time_loop = float('inf')
        best_seed_loop = None
        best_imputed_indices_loop = []
        best_final_metric_loop = np.inf
        best_number_of_examples_saved_loop = None
        best_accuracy_MM_loop = -1.0
        best_f1_MM_loop = -1.0
        accuracy_MI, f1_MI, training_time_MI = 0,0,0
        accuracy_KNN, f1_KNN, training_time_KNN = 0,0,0
        accuracy_MIWA, f1_MIWA, training_time_MIWA = 0,0,0
        accuracy_OG, f1_OG, training_time_OG = 0,0,0

        for seed_val_loop in seeds_to_try:
            print(f"################### Seed {seed_val_loop} for {DATASET_NAME} ###################")
            iter_log_file_path = os.path.join(iter_log_dir, f'{DATASET_NAME}_noise_{noise_level}_s{s_sample_size_bb1}_hr{h_selection_ratio}_ACM_{ACM_STRATEGY_FOR_REPORTING}_iter_log.csv')
            start_time_acm_loop = time.time()
            _, current_imputed_indices_loop, current_final_metric_loop = findminimalImputation(
                X_train_loop_np, y_train_loop_np,
                seed=seed_val_loop,
                X_test_eval=X_test_eval_np,
                y_test_eval=y_test_eval_np,
                X_initial_dirty_for_eval_context=X_train_loop_np.copy(),
                X_original_complete_for_gt_imputation=OG_X_train_main_np,
                log_file_path=iter_log_file_path
            )
            minimal_imputation_time_loop = time.time() - start_time_acm_loop
            print(f"Minimal imputation time for {DATASET_NAME} (ACM): {minimal_imputation_time_loop:.2f} seconds")
            X_final_acm_train_list = []
            y_final_acm_train_list = []
            initial_dirty_mask_main_loop = np.isnan(X_train_loop_np).any(axis=1)
            initially_complete_indices_in_loop = np.where(~initial_dirty_mask_main_loop)[0]
            if initially_complete_indices_in_loop.size > 0:
                X_final_acm_train_list.extend(X_train_loop_np[initially_complete_indices_in_loop])
                y_final_acm_train_list.extend(y_train_loop_np[initially_complete_indices_in_loop])
            current_eval_imputation_strategy = ACM_STRATEGY_FOR_REPORTING
            acm_selected_originally_dirty_indices_for_eval = []
            data_to_impute_for_acm_eval_list = []
            for idx_acm_selected in current_imputed_indices_loop:
                if 0 <= idx_acm_selected < X_train_loop_np.shape[0] and initial_dirty_mask_main_loop[idx_acm_selected]:
                    acm_selected_originally_dirty_indices_for_eval.append(idx_acm_selected)
                    if current_eval_imputation_strategy == 'ground_truth':
                        data_to_impute_for_acm_eval_list.append(OG_X_train_main_np[idx_acm_selected])
                    else:
                        data_to_impute_for_acm_eval_list.append(X_train_loop_np[idx_acm_selected])
            if data_to_impute_for_acm_eval_list:
                data_to_impute_np_for_eval = np.array(data_to_impute_for_acm_eval_list)
                imputed_data_for_acm_eval = None
                if data_to_impute_np_for_eval.size > 0:
                    if current_eval_imputation_strategy == 'mean':
                        imputer_final_acm = SimpleImputer(strategy='mean')
                        imputer_final_acm.fit(X_train_loop_np)
                        imputed_data_for_acm_eval = imputer_final_acm.transform(data_to_impute_np_for_eval)
                    elif current_eval_imputation_strategy == 'knn':
                        n_fit_samples = X_train_loop_np.shape[0]
                        k_fit = knn_k_for_imputation
                        if n_fit_samples <= k_fit: k_fit = max(1, n_fit_samples - 1 if n_fit_samples > 1 else 1)
                        if k_fit > 0 and n_fit_samples > 0 :
                            imputer_final_acm = KNNImputer(n_neighbors=k_fit)
                            imputer_final_acm.fit(X_train_loop_np)
                            imputed_data_for_acm_eval = imputer_final_acm.transform(data_to_impute_np_for_eval)
                        else:
                            imputer_final_acm = SimpleImputer(strategy='mean')
                            if X_train_loop_np.shape[0] > 0: imputer_final_acm.fit(X_train_loop_np)
                            imputed_data_for_acm_eval = imputer_final_acm.transform(data_to_impute_np_for_eval)
                    elif current_eval_imputation_strategy == 'ground_truth':
                        imputed_data_for_acm_eval = data_to_impute_np_for_eval
                    if imputed_data_for_acm_eval is not None:
                        X_final_acm_train_list.extend(imputed_data_for_acm_eval)
                        y_final_acm_train_list.extend(y_train_loop_np[acm_selected_originally_dirty_indices_for_eval])
            num_feat = X_train_loop_np.shape[1] if X_train_loop_np.ndim > 1 and X_train_loop_np.shape[1] > 0 else (OG_X_train_main_np.shape[1] if OG_X_train_main_np.ndim > 1 and OG_X_train_main_np.shape[1] > 0 else 1)
            X_final_acm_train_np = np.array(X_final_acm_train_list) if X_final_acm_train_list else np.empty((0, num_feat))
            y_final_acm_train_np = np.array(y_final_acm_train_list) if y_final_acm_train_list else np.empty((0,))
            if X_final_acm_train_np.ndim == 1 and X_final_acm_train_np.size > 0 and num_feat > 0 :
                X_final_acm_train_np = X_final_acm_train_np.reshape(-1, num_feat)
            elif X_final_acm_train_np.size == 0 and num_feat > 0:
                 X_final_acm_train_np = np.empty((0, num_feat))
            current_accuracy_MM = 0.0
            current_f1_MM = 0.0
            if X_final_acm_train_np.shape[0] > 0 and len(np.unique(y_final_acm_train_np)) >= 2:
                if np.isnan(X_final_acm_train_np).any():
                    temp_final_eval_imputer = SimpleImputer(strategy='mean')
                    X_final_acm_train_np = temp_final_eval_imputer.fit_transform(X_final_acm_train_np)
                clf_acm_final_eval = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed_val_loop, fit_intercept=fit_intercept_svm)
                clf_acm_final_eval.fit(X_final_acm_train_np, y_final_acm_train_np)
                current_accuracy_MM = clf_acm_final_eval.score(X_test_eval_np, y_test_eval_np)
                y_pred_MM_final = clf_acm_final_eval.predict(X_test_eval_np)
                current_f1_MM = f1_score(y_test_eval_np, y_pred_MM_final, zero_division=0)
            print(f"Minimal Method (ACM with {ACM_STRATEGY_FOR_REPORTING}) for {DATASET_NAME}: accuracy_MM: {current_accuracy_MM:.4f}, f1_MM: {current_f1_MM:.4f}")
            if current_accuracy_MM > best_accuracy_MM_loop:
                best_seed_loop = seed_val_loop
                best_accuracy_MM_loop = current_accuracy_MM
                best_f1_MM_loop = current_f1_MM
                best_imputed_indices_loop = current_imputed_indices_loop
                best_final_metric_loop = current_final_metric_loop
                min_time_loop = minimal_imputation_time_loop
            current_number_examples_saved = sanity_check(X_train_loop_np, current_imputed_indices_loop)
            if best_number_of_examples_saved_loop is None or current_number_examples_saved > best_number_of_examples_saved_loop :
                 best_number_of_examples_saved_loop = current_number_examples_saved
            print("-----------------------------------------------------------------------------")

        print(f"\n############################## Results for {DATASET_NAME}, Missing Factor: {noise_level} ##############################")
        print(f"Best seed: {best_seed_loop}")
        print(f"Best accuracy_MM ({ACM_STRATEGY_FOR_REPORTING} for ACM): {best_accuracy_MM_loop:.4f}")
        print(f"Best f1_MM ({ACM_STRATEGY_FOR_REPORTING} for ACM): {best_f1_MM_loop:.4f}")
        print(f"Best number of examples saved (ACM): {best_number_of_examples_saved_loop}")
        print(f"Best minimal_imputation_time (ACM): {min_time_loop:.2f}s")
        print(f"Imputed indices (ACM, best seed run): {sorted(best_imputed_indices_loop)}")
        print(f"Final estimated gap g_k_est (ACM, best seed run): {best_final_metric_loop:.4f}")
        print("#########################################################################################################")

        output_file_path = os.path.join(results_output_dir, f'{DATASET_NAME}_noise_level_{noise_level}_s{s_sample_size_bb1}_hr{h_selection_ratio}_ACM_{ACM_STRATEGY_FOR_REPORTING}.txt')
        with open(output_file_path, 'w') as f:
            f.write(f"################ Results for Missing Factor: {noise_level} ({DATASET_NAME} Dataset) ################\n")
            internal_acm_imputation_strategy = 'ground_truth'
            try:
                sig = inspect.signature(findminimalImputation)
                if 'imputation_strategy' in sig.parameters and sig.parameters['imputation_strategy'].default is not inspect.Parameter.empty:
                    internal_acm_imputation_strategy = sig.parameters['imputation_strategy'].default
            except Exception:
                pass
            f.write(f"ACM Imputation Strategy (in findminimalImputation): {internal_acm_imputation_strategy}\n")
            f.write(f"ACM Imputation Strategy (for this eval & reporting): {ACM_STRATEGY_FOR_REPORTING}\n")
            f.write(f"Rows with missing values in initial dirty train set: {rows_with_missing_values}\n")
            f.write(f"Best seed (last run for this noise level): {best_seed_loop}\n")
            f.write(f"Accuracy_MM (ACM with reported strategy): {best_accuracy_MM_loop:.4f}\n")
            f.write(f"F1_MM (ACM with reported strategy): {best_f1_MM_loop:.4f}\n")
            f.write(f"Number of examples saved (ACM): {best_number_of_examples_saved_loop}\n")
            f.write(f"Minimal_imputation_time (ACM): {min_time_loop:.2f}s\n")
            f.write(f"Imputed Indices (ACM): {sorted(best_imputed_indices_loop)}\n")
            f.write(f"Final Estimated Gap g_k_est (ACM): {best_final_metric_loop:.4f}\n")
            f.write("#########################################################################################################\n")

        print(f"Results for {DATASET_NAME} noise {noise_level} written to {output_file_path}")
        print("\n\n")

