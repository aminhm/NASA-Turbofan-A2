# NASA-Turbofan-A2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names
column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]  # total 26 columns

def build_X(df, drop_constant=False):
    
    feature_cols = ['op_setting_1','op_setting_2','op_setting_3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]
    df_used = df.sort_values('time_in_cycles').groupby('unit_number').tail(1)

    if drop_constant:
        feature_cols = [c for c in feature_cols if df_used[c].nunique() > 1]

    X = df_used[feature_cols].to_numpy()
    return X, feature_cols, df_used

# --- Keep your exploratory code guarded to avoid side-effects on import ---
if __name__ == "__main__":
    # Read the data to different datasets
    train_df1 = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=column_names)
    train_df2 = pd.read_csv('data/train_FD002.txt', sep=r'\s+', header=None, names=column_names)
    train_df3 = pd.read_csv('data/train_FD003.txt', sep=r'\s+', header=None, names=column_names)
    train_df4 = pd.read_csv('data/train_FD004.txt', sep=r'\s+', header=None, names=column_names)
    test_df1 = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=column_names)
    test_df2 = pd.read_csv('data/test_FD002.txt', sep=r'\s+', header=None, names=column_names)
    test_df3 = pd.read_csv('data/test_FD003.txt', sep=r'\s+', header=None, names=column_names)
    test_df4 = pd.read_csv('data/test_FD004.txt', sep=r'\s+', header=None, names=column_names)
    RUL_df1 = pd.read_csv('data/RUL_FD001.txt', header=None, names=['remaining_useful_life_FD001'])
    RUL_df2 = pd.read_csv('data/RUL_FD002.txt', header=None, names=['remaining_useful_life_FD002'])
    RUL_df3 = pd.read_csv('data/RUL_FD003.txt', header=None, names=['remaining_useful_life_FD003'])
    RUL_df4 = pd.read_csv('data/RUL_FD004.txt', header=None, names=['remaining_useful_life_FD004'])

    # Plot cycles per engine
    train_df1.groupby('unit_number')['time_in_cycles'].max().plot(kind='hist', bins=50)
    plt.title('Distribution of Engine Cycles in the Dataset train_FD001')
    plt.xlabel('Max Cycles per Engine')
    plt.show()

    # NaN checks
    print(train_df1.isna().sum())
    print(train_df2.isna().sum())
    print(train_df3.isna().sum())
    print(train_df4.isna().sum())
    print(test_df1.isna().sum())
    print(test_df2.isna().sum())
    print(test_df3.isna().sum())
    print(test_df4.isna().sum())
    print(RUL_df1.isna().sum())
    print(RUL_df2.isna().sum())
    print(RUL_df3.isna().sum())
    print(RUL_df4.isna().sum())

    # Check for constant values
    constant_cols = [col for col in train_df1.columns if train_df1[col].nunique() <= 1]
    print("train_FD001 Constant Columns:", constant_cols)
    constant_cols = [col for col in train_df2.columns if train_df2[col].nunique() <= 1]
    print("train_FD002 Constant Columns:", constant_cols)
    constant_cols = [col for col in train_df3.columns if train_df3[col].nunique() <= 1]
    print("train_FD003 Constant Columns:", constant_cols)
    constant_cols = [col for col in train_df4.columns if train_df4[col].nunique() <= 1]
    print("train_FD004 Constant Columns:", constant_cols)
    constant_cols = [col for col in test_df1.columns if test_df1[col].nunique() <= 1]
    print("test_FD001 Constant Columns:", constant_cols)
    constant_cols = [col for col in test_df2.columns if test_df2[col].nunique() <= 1]
    print("test_FD002 Constant Columns:", constant_cols)
    constant_cols = [col for col in test_df3.columns if test_df3[col].nunique() <= 1]
    print("test_FD003 Constant Columns:", constant_cols)
    constant_cols = [col for col in test_df4.columns if test_df4[col].nunique() <= 1]
    print("test_FD004 Constant Columns:", constant_cols)
    constant_cols = [col for col in RUL_df1.columns if RUL_df1[col].nunique() <= 1]
    print("RUL_FD001 Constant Columns:", constant_cols)
    constant_cols = [col for col in RUL_df2.columns if RUL_df2[col].nunique() <= 1]
    print("RUL_FD002 Constant Columns:", constant_cols)
    constant_cols = [col for col in RUL_df3.columns if RUL_df3[col].nunique() <= 1]
    print("RUL_FD003 Constant Columns:", constant_cols)
    constant_cols = [col for col in RUL_df4.columns if RUL_df4[col].nunique() <= 1]
    print("RUL_FD004 Constant Columns:", constant_cols)
