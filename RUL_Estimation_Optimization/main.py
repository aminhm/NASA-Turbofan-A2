import calibration_funcs as cf
import validation_funcs as vf
import testing_funcs as tf
import prep_funcs as pf

import pandas as pd
import numpy as np

# Define column names
column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sens_meas_{i}' for i in range(1, 22)]  # total 26 columns

# Load train datasets
train_df1 = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=column_names)
train_df2 = pd.read_csv('data/train_FD002.txt', sep=r'\s+', header=None, names=column_names)
train_df3 = pd.read_csv('data/train_FD003.txt', sep=r'\s+', header=None, names=column_names)
train_df4 = pd.read_csv('data/train_FD004.txt', sep=r'\s+', header=None, names=column_names)



## PRE-PROCESSING PHASE

# For each dataset, select random units for calibration, validation, and testing
# Percentages for splitting the data
c_perc = 0.6  # 60% for calibration
v_perc = 0.2  # 20% for validation
t_perc = 0.2  # 20% for testing

train_dfs = [train_df1, train_df2, train_df3, train_df4]
cal_X, val_X, test_X, cal_y, val_y, test_y = pf.split_dataframes(train_dfs, c_perc, v_perc, t_perc)


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

# Print shape and number of units of each df
for i, df in enumerate(train_dfs):
    print(f"Train FD00{i+1}: {df.shape} and units: {df['unit_number'].nunique()}")

# Print shapes of the feature matrices
print(f"\nCalibration feature matrix shape: {cal_X.shape}")
print(f"Validation feature matrix shape: {val_X.shape}")
print(f"Testing feature matrix shape: {test_X.shape}")
print("\n\nBuild completed successfully.\n\n")



## CALIBRATION / TRAINING PHASE
PRESS_med, Q2_med, best_press, best_q2 = cf.expanding_window_cv(cal_X, cal_y['RUL'], max_components=10, show=False)
best_n = best_q2
print(f"Optimal number of components based on Q2: {best_n}")

model = cf.calibrate_pls(cal_X, cal_y['RUL'], n_components=best_n)
selected_sensors, vip_scores = cf.select_sensors(model, cal_X, threshold=1, show=False)
print(f"Selected sensors ({len(selected_sensors)}): {selected_sensors}")

cal_X_reduced = cal_X[selected_sensors]
pls_model_reduced = cf.calibrate_pls(cal_X_reduced, cal_y['RUL'], n_components=best_n)



## VALIDATION PHASE
# ...
    

## TESTING PHASE
# ...