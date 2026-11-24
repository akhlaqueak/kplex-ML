import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max',
                'core_min', 'shell_num', 'shell_avgSize',
                'shell_maxSize', 'shell_minSize', 'shell_stdSize']
algorithms = ['UMKP-S', 'UMKP-R', 'UMKP-P', 'UMKP+-S', 'UMKP+-R', 'UMKP+-P', 'kPEX', 'kPEX-S', 'kPEX-R', 'kPEX-P']

#------- Splits with TIES and Tie Tolerance (optional) -------#

###------ Balanced train-test split based on Y ------###

def prepare_train_test_strat_Y(cat_df,
                       weight_col='weight',
                       test_size=0.2,
                       random_state=42,
                       for_nn=False,
                       kcore_list=None,
                       tie_tolerance=0.00):  # <-- new
    """
    Expands rows into multiple labeled, weighted rows.
    Handles approximate ties: any algorithm within (1+tie_tolerance)*min_time.
    Then performs stratified train-test split on Y (target variable).
    """

    # Prepare the features columns
    if kcore_list:
        feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max', 
                        'core_min', 'shell_num', 'shell_avgSize', 
                        'shell_maxSize', 'shell_minSize', 'shell_stdSize', 'kcore_list']
    elif for_nn:
        feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max', 
                        'core_min', 'shell_num', 'shell_avgSize', 
                        'shell_maxSize', 'shell_minSize', 'shell_stdSize', 'fastest_time']
    else:
        feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max', 
                        'core_min', 'shell_num', 'shell_avgSize', 
                        'shell_maxSize', 'shell_minSize', 'shell_stdSize']

    # Build expanded rows with approximate ties
    expanded_rows = []
    num_tied_expansions = 0

    for idx, row in cat_df.iterrows():
        values = [row[alg] for alg in algorithms]
        min_val = min(values)
        # Any algorithm within tie_tolerance * min_val
        tied_algs = [alg for alg in algorithms if row[alg] <= min_val * (1 + tie_tolerance)]

        
        for alg in tied_algs:
            if kcore_list:
                new_row = row[feature_cols].copy()
            elif for_nn:
                new_row = row[feature_cols].copy()
            else:
                new_row = row[feature_cols].copy()
            new_row['target'] = alg
            new_row['weight'] = row[weight_col] / len(tied_algs) # Split weight among ties
            expanded_rows.append(new_row)
        
        if len(tied_algs) > 1:
            num_tied_expansions += 1

    expanded_df = pd.DataFrame(expanded_rows)
    print(f"Expanded {num_tied_expansions} rows with near ties using tolerance {tie_tolerance*100:.1f}%.")

    # Prepare X, y, weights
    X = expanded_df.drop(['target', 'weight'], axis=1)
    y = expanded_df['target']
    weights = expanded_df['weight']

    # Stratified train-test split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, weights_train, weights_test


###------ Balanced train-test split based on both Y and k ------###

from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_train_test_strat_Y_K(cat_df,
                       weight_col='weight',
                       test_size=0.2,
                       random_state=42,
                       for_nn=False,
                       kcore_list=None,
                       tie_tolerance=0.00):  # <-- new
    """
    Expands rows into multiple labeled, weighted rows.
    Handles approximate ties: any algorithm within (1+tie_tolerance)*min_time.
    Then performs stratified train-test split on Y (target variable).
    """

    # Prepare the features columns
    if kcore_list:
        feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max', 
                        'core_min', 'shell_num', 'shell_avgSize', 
                        'shell_maxSize', 'shell_minSize', 'shell_stdSize', 'kcore_list']
    elif for_nn:
        feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max', 
                        'core_min', 'shell_num', 'shell_avgSize', 
                        'shell_maxSize', 'shell_minSize', 'shell_stdSize', 'fastest_time']
    else:
        feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max', 
                        'core_min', 'shell_num', 'shell_avgSize', 
                        'shell_maxSize', 'shell_minSize', 'shell_stdSize']

    # Build expanded rows with approximate ties
    expanded_rows = []
    num_tied_expansions = 0

    for idx, row in cat_df.iterrows():
        values = [row[alg] for alg in algorithms]
        min_val = min(values)
        # Any algorithm within tie_tolerance * min_val
        tied_algs = [alg for alg in algorithms if row[alg] <= min_val * (1 + tie_tolerance)]

        
        for alg in tied_algs:
            if kcore_list:
                new_row = row[feature_cols].copy()
            elif for_nn:
                new_row = row[feature_cols].copy()
            else:
                new_row = row[feature_cols].copy()
            new_row['target'] = alg
            new_row['weight'] = row[weight_col] / len(tied_algs) # Split weight among ties
            expanded_rows.append(new_row)
        
        if len(tied_algs) > 1:
            num_tied_expansions += 1

    expanded_df = pd.DataFrame(expanded_rows)
    print(f"Expanded {num_tied_expansions} rows with near ties using tolerance {tie_tolerance*100:.1f}%.")

    # 3) create combined stratification key
    expanded_df['stratify_var'] = (
        expanded_df['target'].astype(str)
        + "_k"
        + expanded_df['k_value'].astype(str)
    )
    
    # 4) Prepare X, y, weights
    X = expanded_df.drop(columns=['target', 'weight', 'stratify_var'])
    y = expanded_df['target']
    weights = expanded_df['weight']
    
    # 5) Stratified split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights,
        test_size=test_size,
        random_state=random_state,
        stratify=expanded_df['stratify_var']   # Stratify by BOTH target and k_value
    )
    
    return X_train, X_test, y_train, y_test, weights_train, weights_test


##------------- Scheme 2: For Hierarchial experiments -----------------## 
###------ Balanced train-test split based on Y ------###



###------ Balanced train-test split based on both Y and K------###


