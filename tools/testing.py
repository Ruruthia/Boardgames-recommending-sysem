import numpy as np
import pandas as pd
from collections import defaultdict


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


def coverage(top_n_df, games_df):
    recommended_games = top_n_df['bgg_id'].unique()
    all_games = games_df['bgg_id'].unique()

    return recommended_games.size / all_games.size


def diversity(top_n_df, games_df, criterions=['category', 'mechanic']):
    games_df = games_df[['bgg_id'] + criterions].set_index('bgg_id')

    criterion_diversity = {}
    for criterion in criterions:
        criterion_diversity[criterion] = np.unique(np.hstack(games_df[criterion].dropna())).size

    top_n_df = top_n_df[['bgg_user_name', 'bgg_id']]
    df = top_n_df.join(games_df, on='bgg_id', how='left')

    diversity_per_user = defaultdict(list)

    for _, user_df in df.groupby(by='bgg_user_name'):
        for criterion in criterions:
            user_criterion_diversity = np.unique(np.hstack(user_df[criterion].dropna())).size
            diversity_per_user[criterion].append(user_criterion_diversity
                                                 / criterion_diversity[criterion])

    mean_diversity = {}
    for criterion in criterions:
        mean_diversity[criterion] = np.mean(diversity_per_user[criterion])

    return mean_diversity
