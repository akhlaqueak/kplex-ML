import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

feature_cols = ['k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max',
                'core_min', 'shell_num', 'shell_avgSize',
                'shell_maxSize', 'shell_minSize', 'shell_stdSize']
algorithms = ['UMKP', 'AltRB', 'kPEX', 'Pivot_Br', 'S_Br']

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


##------------- For Holdout test set experiments -----------------## 

###--------- Splits with TIES, works on a clean dataframe (duplicate rows with different labels) -------###
##### --- multi-label aware training set with duplicate lines (but different labels)

def expand_with_ties_and_prepare(
    df,
    weight_col='weight',
    target_col_name='target',
    for_nn=False,
    tie_tolerance: float = 0.0        # new: relative tolerance for “near‐ties”
):
    """
    Given a DataFrame with multiple algorithm runtimes,
    expand rows to duplicate ties (within tie_tol) and adjust weights, then
    return separate X, y, weights DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature columns, algorithm runtimes, and weight.
    weight_col : str
        Column name for the weights (default: 'weight').
    target_col_name : str
        Name of the target column to create (default: 'target').
    for_nn : bool
        If True, also include 'fastest_time' in features.
    tie_tol : float
        Relative tolerance for near‐ties (0.0 = exact only).

    Returns
    -------
    X_out : pd.DataFrame
    y_out : pd.Series
    weights_out : pd.Series
    """
    expanded_rows = []

    for _, row in df.iterrows():
        # 1) collect all algorithm runtimes
        values = np.array([row[alg] for alg in algorithms], dtype=float)
        min_val = values.min()
        threshold = (1 + tie_tolerance) * min_val

        # 2) any algorithm within tolerance counts as a “tie”
        tied_algs = [alg for alg, t in zip(algorithms, values) if t <= threshold]

        # 3) expand one copy per tied algorithm
        for alg in tied_algs:
            if 'kcore_list' in row:
                new_row = row[feature_cols + ['kcore_list']].copy()
            elif for_nn:
                new_row = row[feature_cols + ['fastest_time']].copy()
            else:
                new_row = row[feature_cols].copy()

            new_row[target_col_name] = alg
            # split original weight among all tied‐within‐tol
            new_row[weight_col] = row[weight_col] / len(tied_algs)
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    X_out       = expanded_df.drop([target_col_name, 'weight'], axis=1)
    y_out       = expanded_df[target_col_name]
    weights_out = expanded_df['weight']

    return X_out, y_out, weights_out



def prepare_test_with_ties(
    test_df: pd.DataFrame,
    weight_col: str = 'weight',
    target_col_name: str = 'target'
):
    """
    Given a test DataFrame and a list of algorithm‐runtime columns,
    expand each row into a single record whose 'target' is the list
    of all algorithms tied for minimum runtime, and carry over the weight.

    Parameters
    ----------
    test_df : pd.DataFrame
        Original test set containing feature columns, algorithm run‐times, and weights.
    weight_col : str
        Column name holding the sample weight. Defaults to 'weight'.
    target_col_name : str
        Name to assign to the multi‐label target column. Defaults to 'target'.

    Returns
    -------
    X_out : pd.DataFrame
        Feature matrix for the test set.
    y_out : pd.Series
        Multi‐label targets (each entry is a list of tied algorithms).
    w_out : pd.Series
        Corresponding weights for each test instance.
    """
    rows = []
    for _, row in test_df.iterrows():
        # copy only the feature columns
        new_row = row[feature_cols].copy()
        # find tied fastest algorithms
        times = [row[alg] for alg in algorithms]
        m = min(times)
        tied = [alg for alg in algorithms if row[alg] == m]
        # assign multi‐label target and weight
        new_row[target_col_name] = tied
        new_row[weight_col] = row[weight_col]
        rows.append(new_row)

    expanded = pd.DataFrame(rows).reset_index(drop=True)

    X_out = expanded.drop([target_col_name, weight_col], axis=1)
    y_out = expanded[target_col_name]
    w_out = expanded[weight_col]

    return X_out, y_out, w_out


###------ Balanced train-test split based on Y or Y and K both  ------###


# -----------------------------------------------------------------------------
## - scikit-learn's `train_test_split(stratify=...)` only works with single labels.
# - Our target is a list (e.g. ['UMKP', 'Pivot_Br']) representing ties.
# - We also need to stratify on `k_value`.


#### ----------------------- Multi-Label iterative stratified sampling ------#

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def stratify_holdout_train_test_split(
    df,
    target_col: str = 'target',
    k_value_col: str = None,
    test_size: float = 0.1,
    random_state: int = 40
):
    """
    Produce a balanced hold-out split by iteratively stratifying on:
      - multi-label `target_col` (list of algorithms)
      - and optionally `k_value_col`.

    If `k_value_col` is None, stratifies only on `target_col`.

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    
    df = df.copy()

    rows = []
    for _, row in df.iterrows():
        # copy only the feature columns
        new_row = row.copy()
        # find tied fastest algorithms
        times = [row[alg] for alg in algorithms]
        m = min(times)
        tied = [alg for alg in algorithms if row[alg] == m]
        # assign multi‐label target list
        new_row[target_col] = tied
        rows.append(new_row)

    df = pd.DataFrame(rows).reset_index(drop=True)

    # 1) Build multilabel indicator matrix for the target lists
    Y_alg = np.vstack([
        [int(alg in row) for alg in algorithms]
        for row in df[target_col]
    ])

    # 2) If k_value_col is given, one-hot encode it and combine
    if k_value_col is not None:
        unique_ks = sorted(df[k_value_col].unique())
        Y_k = np.vstack([
            [int(k == kv) for kv in unique_ks]
            for k in df[k_value_col]
        ])
        Y_multi = np.hstack([Y_alg, Y_k])
    else:
        Y_multi = Y_alg

    # 3) Stratified shuffle split on multilabel
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X=np.zeros((len(df), 1)), y=Y_multi))

    # 4) Select DataFrames
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx] .reset_index(drop=True)

    print("Train set shape:", train_df.shape)
    print("Test set shape: ", test_df.shape)

    return train_df, test_df



# 💡 Alternative Strategy:
# “Expand‑Then‑Compress” workflow that
# 1️⃣ Expands each tied row into multiple single‑label rows (one per “fastest” algorithm), carrying along the original row index.
# 2️⃣ Stratifies & splits the expanded DataFrame on (algorithm, k_value) so that each algorithm‑k pair is balanced.
# 3️⃣ Re‑compresses test split, which serves as holdout set, back into the original multi‑label form by grouping on that original index.
# 4️⃣ After splitting, expand ties only in the training set (adjusting weights),
#     keeping the test/holdout set unchanged for fair evaluation.

# ✅ Result:
# Ensures balanced splits on both algorithm and `k_value`,
# while allowing tie expansion later without data leakage.
# 
# Still gives you a test set of whole original samples (with full tie lists) for honest evaluation.
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

def stratify_holdout_train_test_split_alt(
    df,
    target_col='target',
    weight_col: str = 'weight',
    k_value_col=None,
    test_size=0.1,
    random_state=42,
    how='first' # or 'concat'
):
    """
    1) Expand tied rows → one per algorithm in `algorithms` list
    2) Stratify+split on (alg, k_value) if k_value_col given, else on alg only
    3) Re‑compress each side back into multi‑label targets
    """
    df = cat_df.copy().reset_index().rename(columns={'index':'orig_idx'})

    # --- 1) Expand ties ---
    expanded = []
    for _, row in df.iterrows():
        times   = [row[alg] for alg in algorithms]
        min_val = min(times)
        tied    = [alg for alg in algorithms if row[alg] == min_val]
        for alg in tied:
            new = row[feature_cols + ([k_value_col] if k_value_col else []) + [weight_col, 'orig_idx']].copy()
            new['target'] = alg
            new[weight_col] = row[weight_col] / len(tied)
            expanded.append(new)
    expanded_df = pd.DataFrame(expanded)

    # --- 2) Stratify key ---
    if k_value_col:
        strat_key = expanded_df['target'].astype(str) + "_k" + expanded_df[k_value_col].astype(str)
    else:
        strat_key = expanded_df['target'].astype(str)

    # --- 3) Stratified split ---
    tr_exp, te_exp = train_test_split(
        expanded_df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key
    )

    # --- 4) Re‑compress by orig_idx ---
    def recompress(expanded_split):
        groups = []
        for orig, group in expanded_split.groupby('orig_idx'):
            rep = group.iloc[0].copy()
            rep['target'] = list(group['target'])
            rep[weight_col] = group[weight_col].sum()
            groups.append(rep)
        return pd.DataFrame(groups).reset_index(drop=True)

    train_df = recompress(tr_exp)
    test_df  = recompress(te_exp)

    print("Train set shape:", train_df.shape)
    print("Test set shape: ", test_df.shape)

    return train_df, test_df


