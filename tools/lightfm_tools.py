import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, reciprocal_rank, auc_score
from tools.testing import split_testing_set, split_ratings_dataset
from tools.testing import coverage, diversity


def prepare_interactions(df, dataset):
    """ Prepare the interactions matrices for training and testing and dataframe with known ratings (appearing in interactions 
        for training).
    
    
        Parameters:
        df -- Dataframe of ratings, should contain 'bgg_user_name' column.
        dataset -- Dataset object created by LightFM Dataset() function.
    """
    

    print("Splitting test set")
    train_df, test_df = split_ratings_dataset(df)
    test_known, test_unknown = split_testing_set(test_df)
    interactions_df = train_df.append(test_known)
    print("Preparing training interactions")
    interactions = dataset.build_interactions(
        (
            (val["bgg_user_name"], val["bgg_id"], val["value"])
            for idx, val in interactions_df.iterrows()
        )
    )[1]
    print("Preparing testing interactions")
    test_interactions = dataset.build_interactions(
        (
            (val["bgg_user_name"], val["bgg_id"], val["value"])
            for idx, val in test_unknown.iterrows()
        )
    )[1]
    return interactions, test_interactions, interactions_df


def evaluate_model(
    model, interactions, test_interactions, k=5, num_threads=8, item_features=None
):
    """ Print training and testing precision_at_k and AUC measures for the model.
        
        
        Parameters:
        model -- Model from LightFM(), preferably not trained using 'logistic' loss.
        interactions -- Interaction matrix on which the model was trained.
        test_interactions -- Interaction matrix on which we intend to test the model.
        k -- For precision_at_k measure. (default 5)
        num_threads -- Number of threads to use while evaluating model. (default 8)
        item_features -- Item features to use while evaluating model. (default None)        
    """
    
    
    
    train_precision = precision_at_k(
        model, interactions, item_features=item_features, k=5, num_threads=8
    ).mean()
    print("Precision: train %.2f" % (train_precision))

    test_precision = precision_at_k(
        model,
        test_interactions,
        train_interactions=interactions,
        item_features=item_features,
        k=5,
        num_threads=8,
    ).mean()
    print("Precision: test %.2f" % (test_precision))

    train_auc = auc_score(
        model, interactions, item_features=item_features, num_threads=8
    ).mean()
    print("AUC: train %.2f" % (train_auc))

    test_auc = auc_score(
        model,
        test_interactions,
        train_interactions=interactions,
        item_features=item_features,
        num_threads=8,
    ).mean()
    print("AUC: test %.2f" % (test_auc))


def return_top_N(
    N, user_name, model, user_mapping, games_mapping, item_features, training
):
    """ Return top N recommendations made to user. Indexes of games are mapped to be consistent with BGG.
    
    
        Parameters:
        N -- Number of top recommendations to be returned.
        user_name -- Name of the user.
        model -- LightFM model.
        user_mapping -- Mapping from bgg_user_name to internal id of the user in LightFM Dataset.
        games_mapping -- Mapping from bgg_id to internal id of the game in LightFM Dataset.
        item_features -- Item features to be used while making predictions.
        training -- Dataframe containing "bgg_user_name" and list of corresponding bgg_ids used
            for that user while training the model. We will not include them in recommendations.
    """
    user_id = user_mapping[user_name]
    no_games = len(games_mapping)
    known_ids = training[user_name]
    known_ids = [games_mapping[i] for i in known_ids]
    unknown_ids = np.array([i for i in range(no_games) if i not in known_ids])
    ratings = model.predict(
        user_id, unknown_ids, item_features=item_features, num_threads=8
    )
    games_ids = np.argsort(ratings)[::-1][:N]
    top_N = []
    for idx in games_ids:
        idx = unknown_ids[idx]
        top_N.append(
            list(games_mapping.keys())[list(games_mapping.values()).index(idx)]
        )
    return top_N


def evaluate_diversity_and_coverage(
    model, users_sample_size, dataset, games_df, known, path, item_features=None, N=10
):
    """ Save to file top N recommendations for specified users. Print diverity and coverage of these recommendations.
    
    
        Parameters:
        model -- LightFM model for making recomenndations.
        user_sample_size -- Number of users for which we want to make recommendations,
            we will use first user_sample_size users from dataset.
        dataset -- LightFM dataset used while training model.
        games_df -- Dataframe of games, should contain 'bgg_id' column, and columns
            corresponding to each criterion used by diversity function.
        known -- Dataframe containing known ratings (used while training) for each user, should contain 'bgg_user_name'
            and 'bgg_id' columns.
        path -- Path for file where the recommendations are to be stored.
        item_features -- Item features to be used while making predictions. (default None)
        N -- How many top recommendations are supposed to be prepared for each user. (default 10)
        
    """
    users = list(dataset.mapping()[0].keys())[:users_sample_size]
    user_mapping = dataset.mapping()[0]
    games_mapping = dataset.mapping()[2]
    recommendations_df = pd.DataFrame()
    known_grouped = known.groupby("bgg_user_name")["bgg_id"].apply(list)
    for user in tqdm(users):
        rec = return_top_N(
            N, user, model, user_mapping, games_mapping, item_features, known_grouped
        )
        d = {"bgg_user_name": N * [user], "bgg_id": rec}
        d = pd.DataFrame(data=d)
        recommendations_df = recommendations_df.append(d)
    recommendations_df.to_csv(path, compression="gzip", index=False)
    print("Diversity: " + str(diversity(recommendations_df, games_df)))
    print("Coverage: " + str(coverage(recommendations_df)))


def sample_hyperparameters():
    """Iterator yielding random hyperparameters for fitting LightFM model."""
    while True:
        yield {
            "no_components": np.random.randint(50, 100),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": "warp",
            "learning_rate": np.random.uniform(0.01, 0.1),
            "item_alpha": np.random.uniform(1e-5, 1e-7),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(20, 50),
            "rho": np.random.uniform(0.8, 1),
            "epsilon": np.random.uniform(1e-5, 1e-7),
        }


def random_search(
    interactions, test_interactions, item_features, num_samples=20, num_threads=8
):
    """ Fit the model using specified number of random hyperparameters dictionaries, print their values
        and precision of the model and return the best hyperparameters and precision of model using them.
        
        
        Parameters:
        interactions -- Interaction matrix on which the model was trained.
        test_interactions -- Interaction matrix on which we intend to test the model.
        item_features -- Item features to use while evaluating model. (default None) 
        num_samples -- Determines how many random hyperparameters dictionaries are to be tested. (default 20)
        num_threads -- Number of threads to use while evaluating model. (default 8)
        
    """
    best_precision = 0.00
    best_hyperparams = {}
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        print("Params:")
        print(hyperparams)
        num_epochs = hyperparams.pop("num_epochs")
        model = LightFM(**hyperparams)
        model.fit(
            interactions,
            item_features=item_features,
            epochs=num_epochs,
            num_threads=num_threads,
            verbose=True,
        )
        train_precision = precision_at_k(
            model,
            interactions,
            item_features=item_features,
            k=5,
            num_threads=num_threads,
        ).mean()
        test_precision = precision_at_k(
            model,
            test_interactions,
            train_interactions=interactions,
            item_features=item_features,
            k=5,
            num_threads=num_threads,
        ).mean()
        hyperparams["num_epochs"] = num_epochs
        print(
            f"Train precision: {train_precision:4.4f}, test precision: {test_precision:4.4f}"
        )
        if test_precision > best_precision:
            best_precision = test_precision
            best_hyperparams = hyperparams.copy()
    return best_precision, best_hyperparams
