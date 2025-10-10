import calibration_funcs as cf
import validation_funcs as vf
import testing_funcs as tf

import pandas as pd
import numpy as np

# Define column names
column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sens_meas_{i}' for i in range(1, 22)]  # total 26 columns

train_df1 = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=column_names)
train_df2 = pd.read_csv('data/train_FD002.txt', sep=r'\s+', header=None, names=column_names)
train_df3 = pd.read_csv('data/train_FD003.txt', sep=r'\s+', header=None, names=column_names)
train_df4 = pd.read_csv('data/train_FD004.txt', sep=r'\s+', header=None, names=column_names)

# Percentages for splitting the data
c_perc = 0.6  # 60% for calibration
v_perc = 0.2  # 20% for validation
t_perc = 0.2  # 20% for testing

def build_X(df, drop_constant=False, drop_op_setting=False):
    """
    Build the feature matrix X from the dataframe by selecting relevant columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    drop_constant (bool): If True, drop columns with constant values.
    drop_op_setting (bool): If True, drop operational setting columns.
    
    Returns:
    X (np.ndarray): Feature matrix.
    feature_cols (list): List of feature column names used.
    """

    feature_cols = ['op_setting_1','op_setting_2','op_setting_3'] + [f'sens_meas_{i}' for i in range(1, 22)]

    if drop_constant:
        # Drop columns with number of unique values = 1
        feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
    
    if drop_op_setting:
        # Drop operational setting columns
        feature_cols = [c for c in feature_cols if 'op_setting' not in c]

    X = df[feature_cols].to_numpy()
    return X, feature_cols

# For each dataset, select random units for calibration, validation, and testing
train_dfs = [train_df1, train_df2, train_df3, train_df4]
splits = []

for df in train_dfs:
    # Get unique unit numbers
    unique_units = df['unit_number'].unique()
    np.random.shuffle(unique_units)

    # Split the unique units into calibration, validation, and testing sets
    calibration_units = unique_units[:int(len(unique_units) * c_perc)]
    validation_units = unique_units[int(len(unique_units) * c_perc):int(len(unique_units) * (c_perc + v_perc))]
    testing_units = unique_units[int(len(unique_units) * (c_perc + v_perc)):]

    splits.append({
        'calibration': df[df['unit_number'].isin(calibration_units)],
        'validation': df[df['unit_number'].isin(validation_units)],
        'testing': df[df['unit_number'].isin(testing_units)]
    })

# Print the number of units in each split for each dataset
for i, split in enumerate(splits, start=1):
    print(f"\nDataset FD00{i}:")
    print(f"  Total unique units:", train_dfs[i-1]['unit_number'].nunique())
    print(f"  Calibration units: {split['calibration']['unit_number'].nunique()}")
    print(f"  Validation units: {split['validation']['unit_number'].nunique()}")
    print(f"  Testing units: {split['testing']['unit_number'].nunique()}")
    
## CALIBRATION / TRAINING PHASE
# ...

## VALIDATION PHASE
# ...

## TESTING PHASE
# ...