import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

def cv_select_lv_subset(Xtr, ytr, Kfold=10, maxLV=9):
    """
    Select the best number of PLS components using K-fold CV.
    
    Parameters:
    Xtr (np.ndarray):  Calibration matrix
    ytr (np.ndarray): Calibration RUL vector
    Kfold (int): Number of folds for cross-validation
    maxLV (int): Maximum number of latent variables to consider

    Returns:
    best_lv (int): Optimal number of latent variables (components)
    Q2_mean (np.ndarray): Mean Q2 for each LV
    PRESS_mean (np.ndarray): Mean PRESS for each LV
    """
    n_samples = Xtr.shape[0]
    indices = np.arange(n_samples)
    fold_sizes = n_samples // Kfold
    Q2_folds = np.zeros((Kfold, maxLV))
    PRESS_folds = np.zeros((Kfold, maxLV))

    for lv in range(1, maxLV + 1):
        for fold in range(Kfold):
            start = fold * fold_sizes
            end = n_samples if fold == Kfold-1 else (fold+1) * fold_sizes
            val_idx = indices[start:end]
            cal_idx = np.setdiff1d(indices, val_idx)

            X_cal, y_cal = Xtr[cal_idx, :], ytr[cal_idx]
            X_val, y_val = Xtr[val_idx, :], ytr[val_idx]

            # Standardize using calibration data
            mu, sigma = X_cal.mean(axis=0), X_cal.std(axis=0)
            sigma[sigma == 0] = 1
            X_calZ = (X_cal - mu) / sigma
            X_valZ = (X_val - mu) / sigma

            # Fit PLS
            n_comp = min(lv, np.linalg.matrix_rank(X_calZ), X_calZ.shape[1])
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_calZ, y_cal)
            y_pred = pls.predict(X_valZ).ravel()

            PRESS_folds[fold, lv-1] = np.sum((y_val - y_pred) ** 2)
            TSS = np.sum((y_val - np.mean(y_cal)) ** 2)
            Q2_folds[fold, lv-1] = 1 - PRESS_folds[fold, lv-1] / max(TSS, np.finfo(float).eps)

    PRESS_mean = np.mean(PRESS_folds, axis=0)
    Q2_mean = np.mean(Q2_folds, axis=0)
    best_lv = int(np.argmax(Q2_mean) + 1)
    
    return best_lv, Q2_mean, PRESS_mean

def evaluate_vip_subset(X_cal, y_cal, X_val, y_val, selected_features, Kfold=10, maxLV=9, show=False):
    """
    Evaluate the VIP-selected features subset.
    
    Parameters:
    X_cal (pd.DataFrame): Calibration dataframe.
    y_cal (pd.DataFrame): Calibration RUL dataframe.
    X_val (pd.DataFrame): Validation dataframe.
    y_val (pd.DataFrame): Validation RUL dataframe.
    selected_features (list): List of selected feature names.
    Kfold (int): Number of folds for cross-validation.
    maxLV (int): Maximum number of latent variables to consider.
    
    Returns:
    pls (PLSRegression): Trained PLS model on the selected features.
    """
    col_idx = [X_cal.columns.get_loc(f) for f in selected_features]
    # Convert to arrays for PLS
    X_cal_arr = X_cal.iloc[:, col_idx].to_numpy()
    X_val_arr = X_val.iloc[:, col_idx].to_numpy()
    y_cal_arr = y_cal.to_numpy().ravel()
    y_val_arr = y_val.to_numpy().ravel()

    # CV on calibration set to select best LV
    best_lv, _, _ = cv_select_lv_subset(X_cal_arr, y_cal_arr, Kfold=Kfold, maxLV=maxLV)

    # Fit final model on full calibration set with best LV
    pls = PLSRegression(n_components=best_lv)
    pls.fit(X_cal_arr, y_cal_arr)

    # Predict on validation/test
    y_pred = pls.predict(X_val_arr).ravel()
    PRESS_val = np.sum((y_val_arr - y_pred) ** 2)
    RMSE_val = np.sqrt(mean_squared_error(y_val_arr, y_pred))
    TSS_val = np.sum((y_val_arr - np.mean(y_cal_arr)) ** 2)
    Q2_val = 1 - PRESS_val / max(TSS_val, np.finfo(float).eps)

    if show:
        # Print results
        print(f"\nOptimal number of components (LV): {best_lv}")
        print(f"Validation PRESS: {PRESS_val:.3f}")
        print(f"Validation RMSE:  {RMSE_val:.3f}")
        print(f"Validation Q2:   {Q2_val:.3f}\n")
    
    return pls


def plot_predictions(y_true, y_pred, title="Predictions"):
    """
    Plot true vs predicted RUL values.
    
    Parameters:
    y_true (np.ndarray): True RUL values.
    y_pred (np.ndarray): Predicted RUL values.
    title (str): Title of the plot.
    """
    
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', label='Ideal prediction')
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.legend()
    plt.grid(True)