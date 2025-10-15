import calibration_funcs as cf
import validation_funcs as vf
import prep_funcs as pf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define column names
column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sens_meas_{i}' for i in range(1, 22)]  # total 26 columns

# Load train datasets
train_df1 = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=column_names)
train_df2 = pd.read_csv('data/train_FD002.txt', sep=r'\s+', header=None, names=column_names)
train_df3 = pd.read_csv('data/train_FD003.txt', sep=r'\s+', header=None, names=column_names)
train_df4 = pd.read_csv('data/train_FD004.txt', sep=r'\s+', header=None, names=column_names)

c_perc = 0.6  # 60% for calibration
v_perc = 0.2  # 20% for validation
t_perc = 0.2  # 20% for testing

train_dfs = [train_df1, train_df2, train_df3, train_df4]

for i, train_df in enumerate(train_dfs):
    print(f"\n\nProcessing Train FD00{i+1}\n\n")
    
    ## PRE-PROCESSING PHASE
    cal_X, val_X, test_X, cal_y, val_y, test_y = pf.split_dataframes(train_df, c_perc, v_perc, t_perc)


    cal_X, cal_features = pf.build_X(cal_X, drop_constant=True, drop_op_setting=True)
    val_X, val_features = pf.build_X(val_X, drop_constant=True, drop_op_setting=True)
    test_X, test_features = pf.build_X(test_X, drop_constant=True, drop_op_setting=True)

    cal_y = pf.compute_RUL(cal_y)
    val_y = pf.compute_RUL(val_y)
    test_y = pf.compute_RUL(test_y)

    # Standardize data based on calibration set statistics
    cal_X, cal_mean, cal_std = pf.standardize_data(cal_X)
    val_X, _, _ = pf.standardize_data(val_X, cal_mean, cal_std)
    test_X, _, _ = pf.standardize_data(test_X, cal_mean, cal_std)



    ## CALIBRATION / TRAINING PHASE
    PRESS_med, Q2_med, best_press, best_q2 = cf.expanding_window_cv(cal_X, cal_y['RUL'], max_components=15, n_folds=10, show=True)
    best_n = best_q2
    print(f"Optimal number of components based on Q^2: {best_n}")

    model = cf.calibrate_pls(cal_X, cal_y['RUL'], n_components=best_n)

    selected_sensors, vip_scores = cf.select_sensors(model, cal_X, threshold=1, show=False)
    print(f"Selected sensors ({len(selected_sensors)}): {selected_sensors}")

    cal_X_reduced = cal_X[selected_sensors]
    val_X_reduced = val_X[selected_sensors]
    test_X_reduced = test_X[selected_sensors]

    cal_X_reduced, cal_mean, cal_std = pf.standardize_data(cal_X_reduced)
    val_X_reduced, _, _ = pf.standardize_data(val_X_reduced, mean=cal_mean, std=cal_std)
    test_X_reduced, _, _ = pf.standardize_data(test_X_reduced, mean=cal_mean, std=cal_std)



    ## VALIDATION PHASE
    pls_model_reduced = vf.evaluate_vip_subset(
        cal_X_reduced, cal_y['RUL'],
        val_X_reduced, val_y['RUL'],
        selected_sensors, Kfold=10, maxLV=9, show=True
    )

    val_pred = pls_model_reduced.predict(val_X_reduced.to_numpy()).ravel()


    ## TESTING PHASE
    test_pred = pls_model_reduced.predict(test_X_reduced.to_numpy()).ravel()
    test_y_true = test_y['RUL'].to_numpy().ravel()
    test_rmse = np.sqrt(np.mean((test_y_true - test_pred)**2))
    test_press = np.sum((test_y_true - test_pred)**2)
    test_q2 = 1 - test_press / max(np.sum((test_y_true - np.mean(cal_y['RUL']))**2), np.finfo(float).eps)

    print(f"\nTest PRESS: {test_press:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}")
    print(f"Test Q²:    {test_q2:.3f}\n\n")

    vf.plot_predictions(val_y['RUL'].to_numpy().ravel(), val_pred, title="Validation Results")
    vf.plot_predictions(test_y_true, test_pred, title="Testing Results")
    plt.show()