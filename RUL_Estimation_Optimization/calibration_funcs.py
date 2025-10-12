import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


def expanding_window_cv(X, y, max_components=10, n_folds=10, show=False):
    '''
    Perform expanding window cross-validation to evaluate PLS regression models with varying number of components.
    
    Parameters:
    X (pd.DataFrame): Calibration matrix.
    y (pd.DataFrame): Target RUL values.
    max_components (int): Maximum number of PLS components to evaluate.
    n_folds (int): Number of folds for cross-validation.

    Returns:
    PRESS_med (np.ndarray): Median PRESS values for each component.
    Q2_med (np.ndarray): Median Q2 values for each component.
    best_press (int): Optimal number of components based on PRESS.
    best_q2 (int): Optimal number of components based on Q2.
    '''
    
    n_samples = X.shape[0]
    block = n_samples // (n_folds + 1)
    PRESS_folds = np.full((n_folds, max_components), np.nan)
    Q2_folds = np.full((n_folds, max_components), np.nan)

    for a in range(1, max_components + 1):
        for f in range(n_folds):
            cal_end = block * (f + 1)
            val_beg = cal_end
            val_end = min(cal_end + block, n_samples)
            if val_beg >= n_samples:
                break
            cal_idx = np.arange(0, cal_end)
            val_idx = np.arange(val_beg, val_end)

            Xc, yc = X.iloc[cal_idx], y.iloc[cal_idx]
            Xv, yv = X.iloc[val_idx], y.iloc[val_idx]

            # Standardize within the calibration window
            mu = Xc.mean(axis=0)
            std = Xc.std(axis=0).replace(0, 0.001)
            XcZ = (Xc - mu) / std
            XvZ = (Xv - mu) / std

            a = min([a, np.linalg.matrix_rank(XcZ), XcZ.shape[1]])
            pls = calibrate_pls(XcZ, yc, n_components=a)
            yhat_v = pls.predict(XvZ).ravel()

            press_f = np.sum((yv - yhat_v) ** 2)
            ybar_cal = np.mean(yc)
            tss_f = np.sum((yv - ybar_cal) ** 2)

            PRESS_folds[f, a - 1] = press_f
            Q2_folds[f, a - 1] = 1 - press_f / max(tss_f, np.finfo(float).eps)

    PRESS_med = np.nanmean(PRESS_folds, axis=0)
    Q2_med = np.nanmean(Q2_folds, axis=0)
    best_press = np.argmin(PRESS_med) + 1
    best_q2 = np.argmax(Q2_med) + 1
    
    if show:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, max_components + 1), PRESS_med, marker='o')
        plt.axvline(best_press, color='r', linestyle='--', label=f'Best n={best_press}')
        plt.title('PRESS vs Number of PLS Components')
        plt.xlabel('Number of PLS Components')
        plt.ylabel('PRESS')
        plt.xticks(range(1, max_components + 1))
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, max_components + 1), Q2_med, marker='o')
        plt.axvline(best_q2, color='r', linestyle='--', label=f'Best n={best_q2}')
        plt.title('Q^2 vs Number of PLS Components')
        plt.xlabel('Number of PLS Components')
        plt.ylabel('Q^2')
        plt.xticks(range(1, max_components + 1))
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return PRESS_med, Q2_med, best_press, best_q2


def calibrate_pls(X, y, n_components=10):
    '''
    Calibrate a PLS regression model.
    
    Parameters:
    X (pd.DataFrame): Calibration dataframe.
    y (pd.DataFrame): Target RUL values for calibration.
    n_components (int): Number of components to use.
    
    Returns:
    pls (PLSRegression): PLS regression model.
    '''
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    
    return pls


def select_sensors(pls_model, X, threshold=1, show=False):
    '''
    Calculation of VIP scores and selection of important sensors based on a threshold.
    
    Parameters:
    pls_model (PLSRegression): PLS regression model.
    X (pd.DataFrame): Calibration dataframe.
    threshold (float): Threshold for selecting important sensors.
    
    Returns:
    selected_sensors (list): List of selected sensor column names.
    '''
    
    w = pls_model.x_weights_
    p = w.shape[0]
    if hasattr(pls_model, 'explained_variance_ratio_'):
        SSYa = pls_model.explained_variance_ratio_.flatten()
    else:
        SSYa = np.var(pls_model.y_scores_, axis=0)

    vip_scores = np.sqrt(
        p * (np.sum((w ** 2) * SSYa, axis=1)) / max(np.sum(SSYa), np.finfo(float).eps)
    )
    
    # Select sensors with VIP scores above the threshold
    feature_names = X.columns
    sorted_idx = np.argsort(-vip_scores)
    sorted_vip = vip_scores[sorted_idx]
    sorted_features = feature_names[sorted_idx]

    selected_sensors = [sorted_features[i] for i in range(len(sorted_features)) if sorted_vip[i] >= threshold]

    if show:
        # Plot VIP scores
        plt.figure(figsize=(10, 4))
        plt.bar(selected_sensors, [sorted_vip[i] for i in range(len(sorted_features)) if sorted_vip[i] >= threshold])
        plt.axhline(1, color='r', linestyle='--', label='VIP = 1')
        plt.title('Variable Importance in Projection (VIP)')
        plt.ylabel('VIP Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return selected_sensors, vip_scores