import pandas as pd
import numpy as np
from tqdm import tqdm
from lightfm.evaluation import precision_at_k, recall_at_k, reciprocal_rank, auc_score
from tools.testing import split_testing_set, split_ratings_dataset
from tools.testing import coverage, diversity



def prepare_interactions(df, dataset):
    
    print('Splitting test set')
    train_df, test_df = split_ratings_dataset(df)
    test_known, test_unknown = split_testing_set(test_df)
    interactions_df = train_df.append(test_known)
    print('Preparing training interactions')
    interactions = dataset.build_interactions(((val['bgg_user_name'], val['bgg_id'], val['value']) for idx, val in interactions_df.iterrows()))[1]
    print('Preparing testing interactions')
    test_interactions = dataset.build_interactions(((val['bgg_user_name'], val['bgg_id'], val['value']) for idx, val in test_unknown.iterrows()))[1]
    return interactions, test_interactions, interactions_df

def evaluate_model(model, interactions, test_interactions, k=5, num_threads=8, item_features = None):
    train_precision = precision_at_k(model, interactions, item_features = item_features, k=5, num_threads=8).mean()
    print('Precision: train %.2f' % (train_precision))

    test_precision = precision_at_k(model, test_interactions, train_interactions = interactions, item_features = item_features, k=5, num_threads=8).mean()
    print('Precision: test %.2f' % (test_precision))
    
    train_auc = auc_score(model, interactions, item_features = item_features, num_threads=8).mean()
    print('AUC: train %.2f' % (train_auc))

    test_auc = auc_score(model, test_interactions, train_interactions = interactions, item_features = item_features, num_threads=8).mean()
    print('AUC: test %.2f' % (test_auc))
    
def return_top_N(N, user_name, model, user_mapping, games_mapping, item_features, training):
    user_id = user_mapping[user_name]
    no_games = len(games_mapping)
    known_ids = training[user_name]
    known_ids = [games_mapping[i] for i in known_ids]
    unknown_ids = np.array([i for i in range(no_games) if i not in known_ids])
    ratings = model.predict(user_id, unknown_ids, item_features = item_features, num_threads = 8)
    games_ids = np.argsort(ratings)[::-1][:N]
    top_N = []
    for idx in games_ids:
        idx = unknown_ids[idx]
        top_N.append(list(games_mapping.keys())[list(games_mapping.values()).index(idx)])
    return top_N

def evaluate_diversity_and_coverage(model, users_sample_size, dataset, games_df,
                                    known, path, item_features = None, N = 10):
    users = dataset.mapping()[0].keys()[:user_sample_size]
    user_mapping = dataset.mapping()[0]
    games_mapping = dataset.mapping()[2]
    recommendations_df = pd.DataFrame()
    known_grouped = known.groupby('bgg_user_name')['bgg_id'].apply(list)
    for user in tqdm(users):
        rec = return_top_N(N, user, model, user_mapping, games_mapping, item_features, known_grouped)
        d = {'bgg_user_name': N*[user], 'bgg_id': rec}
        d = pd.DataFrame(data=d)
        recommendations_df = recommendations_df.append(d)
    recommendations_df.to_csv(path, compression='gzip', index = False)
    print('Diversity: ' +  str(diversity(recommendations_df, games_df)))
    print('Coverage: ' +  str(coverage(recommendations_df))) 
    