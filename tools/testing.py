import numpy as np
import pandas as pd


def split_ratings_dataset(ratings_df, seed=None, frac=0.7):
    if seed is not None:
        np.random.seed(seed)

    users = ratings_df['bgg_user_name'].unique()
    np.random.shuffle(users)
    train_size = int(frac*users.shape[0])

    train_df = ratings_df[ratings_df['bgg_user_name'].isin(users[:train_size])]
    test_df = ratings_df[ratings_df['bgg_user_name'].isin(users[train_size:])]

    return train_df, test_df


def split_testing_set(test_df, seed=None, frac=0.8):
    if seed is not None:
        np.random.seed(seed)

    grouped = test_df.groupby(by='bgg_user_name')
    test_known = []
    test_unknown = []
    for _, df in grouped:
        df_size = df.shape[0]

        known_size = int(round(frac*df_size))
        known_indices = np.random.choice(df_size, known_size, replace=False)
        known_data = df.iloc[known_indices]
        test_known.append(known_data)

        unknown_indices = np.setdiff1d(np.arange(df_size), known_indices)
        unknown_data = df.iloc[unknown_indices]
        test_unknown.append(unknown_data)

    return pd.concat(test_known), pd.concat(test_unknown)
