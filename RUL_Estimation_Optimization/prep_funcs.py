from operator import index
import pandas as pd
import numpy as np

def compute_RUL(df):
    """
    Compute the Remaining Useful Life (RUL) for each unit in the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing 'unit_number' and 'time_in_cycles' columns.
    
    Returns:
    pd.DataFrame: Dataframe with an additional 'RUL' column.
    """
    
    # Add RUL column
    df = df.copy()
    df['RUL'] = 0
    
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        max_cycle = unit_data['time_in_cycles'].max()
        df.loc[df['unit_number'] == unit, 'RUL'] = max_cycle - unit_data['time_in_cycles']
    
    return df


def build_X(df, drop_constant=False, drop_op_setting=False):
    """
    Build the feature matrix X from the dataframe by selecting relevant columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    drop_constant (bool): If True, drop columns with constant values.
    drop_op_setting (bool): If True, drop operational setting columns.
    
    Returns:
    X (pd.DataFrame): Feature matrix.
    feature_cols (list): List of feature column names used.
    """

    feature_cols = ['unit_number','op_setting_1','op_setting_2','op_setting_3'] + [f'sens_meas_{i}' for i in range(1, 22)]

    if drop_constant:
        # Drop columns with number of unique values = 1
        feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
        
        # Print names of dropped columns
        dropped_cols = [c for c in ['op_setting_1','op_setting_2','op_setting_3'] + [f'sens_meas_{i}' for i in range(1, 22)] if c not in feature_cols]
        if dropped_cols:
            print(f"Dropped constant columns: {dropped_cols}")
    
    if drop_op_setting:
        # Drop operational setting columns
        feature_cols = [c for c in feature_cols if 'op_setting' not in c]
        
    X = df[feature_cols].copy()
    return X, feature_cols


def split_dataframes(dfs, c_perc=0.6, v_perc=0.2, t_perc=0.2):
    '''
    Split each dataframe in the list into calibration, validation, and testing sets based on unique unit numbers.
    
    Parameters:
    dfs (list of pd.DataFrame): List of dataframes to be split.
    c_perc (float): Percentage of data to be used for calibration.
    v_perc (float): Percentage of data to be used for validation.
    t_perc (float): Percentage of data to be used for testing.
    
    Returns:
    cal_X, val_X, test_X (list of pd.DataFrame): Lists of dataframes for calibration, validation, and testing sets.
    cal_y, val_y, test_y (list of pd.DataFrame): Lists of RUL dataframes for calibration, validation, and testing sets.
    '''
    
    cal_X, val_X, test_X = [], [], []
    cal_y, val_y, test_y = [], [], []

    for df_X in dfs:
        # shuffle unique units
        units = df_X['unit_number'].unique()
        np.random.shuffle(units)

        n_cal = int(len(units) * c_perc)
        n_val = int(len(units) * v_perc)

        cal_units = units[:n_cal]
        val_units = units[n_cal:n_cal+n_val]
        test_units = units[n_cal+n_val:]

        # split rows by units e aggiungi alle liste
        cal_X.append(df_X[df_X['unit_number'].isin(cal_units)])
        val_X.append(df_X[df_X['unit_number'].isin(val_units)])
        test_X.append(df_X[df_X['unit_number'].isin(test_units)])
        
        # split RUL dataframes by units e aggiungi alle liste
        cal_y.append(df_X[df_X['unit_number'].isin(cal_units)][['unit_number', 'time_in_cycles']].copy())
        val_y.append(df_X[df_X['unit_number'].isin(val_units)][['unit_number', 'time_in_cycles']].copy())
        test_y.append(df_X[df_X['unit_number'].isin(test_units)][['unit_number', 'time_in_cycles']].copy())

    # Concatena DOPO il loop
    cal_X = pd.concat(cal_X, ignore_index=True)
    val_X = pd.concat(val_X, ignore_index=True)
    test_X = pd.concat(test_X, ignore_index=True)

    cal_y = pd.concat(cal_y, ignore_index=True)
    val_y = pd.concat(val_y, ignore_index=True)
    test_y = pd.concat(test_y, ignore_index=True)

    return cal_X, val_X, test_X, cal_y, val_y, test_y


def standardize_data(data, mean=None, std=None):
    """
    Standardize the data to have zero mean and unit variance.
    
    Parameters:
    data (pd.DataFrame): Input data to be standardized.
    mean (np.ndarray): Mean values for each feature (optional).
    std (np.ndarray): Standard deviation values for each feature (optional).

    Returns:
    standardized_data (pd.DataFrame): Standardized data.
    mean (np.ndarray): Mean values for each feature.
    std (np.ndarray): Standard deviation values for each feature.
    """

    if mean is None or std is None:
        mean = np.mean(data.values, axis=0)
        std = np.std(data.values, axis=0)

    # Avoid division by zero
    std[std == 0] = 1
    standardized_values = (data.values - mean) / std

    return pd.DataFrame(standardized_values, columns=data.columns, index=data.index), mean, std

