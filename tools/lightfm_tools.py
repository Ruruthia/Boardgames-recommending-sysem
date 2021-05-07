import pandas as pd
import numpy as np
from tqdm import tqdm
from lightfm import LightFM
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
    
# rho, epsilon - only for adadelta    
def gridsearch(interactions, test_interactions, item_features = None, k=5, num_threads=8,
               schedules = ['adagrad', 'adadelta'], components = [32, 64, 128, 256],
               learning_rates = [0.01, 0.05, 0.1], item_alphas = [1e-07, 1e-06, 1e-05], max_samples = [10, 15, 20],
               rhos = [0.85, 0.9, 0.95], epsilons = [1e-07, 1e-06, 1e-05], epochs = [20, 40, 60]):
    best_schedule, best_comp, best_lr, best_alpha, best_samples, best_rho, best_eps, best_epochs = schedules[0], components[0],
    learning_rates[0], item_alphas[0], max_samples[0], rhos[0], epsilons[0], epochs[0]
    best_train_precision = 0.00
    best_test_precision = 0.00
    
    for schedule in schedules:
        for component in components:
            for lr in learning_rates:
                for item_alpha in item_alphas:
                    for sample in max_samples:
                        if schedule is 'adagrad':
                            for rho in rhos:
                                for eps in epsilons:
                                    model = LightFM(loss = 'warp', no_components = component, learning_schedule = schedule,
                                                    learning_rate = lr, rho = rho, epsilon = eps, item_alpha = item_alpha,
                                                    max_sampled = sample)
                        if schedule is 'adadelta':
                                    model = LightFM(loss = 'warp', no_components = component, learning_schedule = schedule,
                                                    learning_rate = lr, item_alpha = item_alpha,
                                                    max_sampled = sample)                                  
                        for epoch in epochs:
                                print('-----Params-----')
                                print(f'Schedule: {schedule}, no_components: {component}, learning_rate: {lr}')
                                if schedule is 'adagrad':
                                    print(f'max_sampled: {sample}, rho: {rho}, epsilon: {eps}, epochs: {epoch}')
                                if schedule is 'adadelta':
                                    print(f'max_sampled: {sample}, epochs: {epoch}')
                                model.fit(interactions, item_features = item_features, epochs=epoch, num_threads=num_threads)
                                train_precision = precision_at_k(model, interactions,
                                                                 item_features = item_features,
                                                                 k=k, num_threads=num_threads).mean()
                                test_precision = precision_at_k(model, test_interactions, train_interactions = interactions,
                                                                item_features = item_features,
                                                                k=k, num_threads= num_threads).mean()
                                print(f'Train precision: {train_precision}, test precision: {test_precision}')
                                if best_train_precision < train_precision:
                                    best_train_precision = train_precision
                                if best_test_precision < test_precision:
                                    best_test_precision = test_precision
                                    if schedule is 'adagrad': 
                                        best_schedule, best_comp, best_lr, best_alpha, best_samples, best_rho, best_eps, best_epochs = schedule, component, lr, item_alpha, sample, rho, eps, epoch
                                    if schedule is 'adadelta':
                                        best_schedule, best_comp, best_lr, best_alpha, best_samples, best_epochs = schedule, component, lr, item_alpha, sample, epoch
    return best_schedule, best_comp, best_lr, best_alpha, best_samples, best_rho, best_eps, best_epochs
                            
                