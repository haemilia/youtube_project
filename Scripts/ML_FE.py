#%%
import pathlib
import json
with open("../private.json") as json_file:
    priv = json.load(json_file)
DATA_PATH = pathlib.Path(priv["DATA_PATH"])
HOME_DIR = pathlib.Path(priv["HOME_DIR"])

import pandas as pd
import numpy as np
import ML_convenience as ml

## L1 (lasso) based feature selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, make_scorer
def lasso_feature_selection(X_train, y_train, X_test, y_test, 
                            feature_names:np.array, 
                            output_file:pathlib.Path,
                            scoring="accuracy"):
    # Make a pipeline
    pipe = Pipeline(
        [("scaler", MinMaxScaler()),
        ("lasso_selector", SelectFromModel(Lasso())),
        # Increase c to reduce regularization strength, since technically lasso was already regularizing
        # Increase tolerance to go faster, because we're just looking for feature selection, not making the optimum model here
        ("logistic_regression", LogisticRegression(C=2.0, tol=0.01, max_iter=1000)),
        ]
    )
    # hyperparameter
    hyperparam_grid = {
        "lasso_selector__estimator__alpha": np.logspace(-4, -2, 20)
    }
    score_methods = {"accuracy":make_scorer(accuracy_score),
                                     "f1": make_scorer(f1_score),}
    # grid search
    splitter = StratifiedKFold(n_splits=6, shuffle=True)
    grid_search = GridSearchCV(pipe, 
                            hyperparam_grid, 
                            cv=splitter, 
                            scoring=score_methods, 
                            n_jobs=-1,
                            refit=scoring)
    grid_search.fit(X_train, y_train)
    # best results
    
    best_alpha = grid_search.best_params_['lasso_selector__estimator__alpha']
    best_accuracy = grid_search.best_score_
    best_model = grid_search.best_estimator_
    train_accuracy = grid_search.score(X_train, y_train)
    best_train_score = train_accuracy
    test_accuracy = grid_search.score(X_test, y_test)
    best_test_score= test_accuracy
    selected_indices = best_model.named_steps["lasso_selector"].get_support(indices=True)
    lasso_coefficients = best_model.named_steps["lasso_selector"].estimator_.coef_
    all_coef = np.zeros(X_train.shape[1])
    all_coef[selected_indices] = lasso_coefficients[selected_indices]

    best = {}
    best["lasso_selected_indices"] = selected_indices
    best["all_lasso_coefficients"] = all_coef
    best_logistic_coefficients = best_model.named_steps["logistic_regression"].coef_[0]
    best["logistic_coefficients"] = best_logistic_coefficients
    best["alpha"] = best_alpha
    best[f"{scoring}"] = best_accuracy
    best["model"] = best_model
    best["train_score"] = best_train_score
    best["test_score"] = best_test_score

    # print report
    output_str = f"""Lasso Feature Selection Report\n
Best lasso alpha value : {best_alpha}
Best {scoring}: {best_accuracy}
Train {scoring} with best model: {best_train_score}
Test {scoring} with best model: {best_test_score}
Lasso coefficients based on original features (0 for unselected features):
{all_coef}
Logistic coefficients based on selected features:
{best_logistic_coefficients}
Selected feature indices:
{selected_indices}
Selected feature names:
{feature_names[selected_indices]}"""
    print(output_str)

    # save report
    with open(output_file, "w") as f:
        f.write(output_str)
    
    return best

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.family"] = "Freesentation"
## PCA feature selection
def pca_feature_selection(X_train, y_train, X_test, y_test, 
                          feature_names:np.array, 
                          output_file:pathlib.Path, plot_dir:pathlib.Path,
                          scoring="accuracy"):
    # Make a pipeline
    pipe = Pipeline(
        [("scaler", MinMaxScaler()),
        ("pca", PCA()),
        # Increase tolerance to go faster, because we're just looking for feature selection, not making the optimum model here
        ("logistic_regression", LogisticRegression(tol=0.01, max_iter=1000)),
        ]
    )
    # hyperparameter
    hyperparam_grid = {
        "pca__n_components": np.array(range(3, X_train.shape[1])),
    }
    score_methods = {"accuracy":make_scorer(accuracy_score),
                                     "f1": make_scorer(f1_score),}
    # grid search
    splitter = StratifiedKFold(n_splits=6, shuffle=True)
    grid_search = GridSearchCV(pipe, 
                            hyperparam_grid, 
                            cv=splitter, 
                            scoring=score_methods, 
                            n_jobs=-1,
                            refit=scoring)
    grid_search.fit(X_train, y_train)
    # best results
    best_n_components = grid_search.best_params_['pca__n_components']
    best_accuracy = grid_search.best_score_
    best_model = grid_search.best_estimator_
    train_accuracy = grid_search.score(X_train, y_train)
    best_train_score = train_accuracy
    test_accuracy = grid_search.score(X_test, y_test)
    best_test_score = test_accuracy
    best_pca = best_model.named_steps["pca"]
    best_components = best_model.named_steps["pca"].components_
    best_explained_var_ratio = best_model.named_steps["pca"].explained_variance_ratio_
    best_logistic_coefficients = best_model.named_steps["logistic_regression"].coef_[0]

    best = {}
    best["n_components"] = best_n_components
    best["accuracy"] = best_accuracy
    best["model"] = best_model
    best["train_score"] = best_train_score
    best["test_score"]= best_test_score
    best["pca"] = best_pca
    # Each row represents a principal component
    # Each column corresponds to original feature
    best["components"] = best_components
    best["explained_variance_ratio"] = best_explained_var_ratio
    best["logistic_coefficients"] = best_logistic_coefficients

    # print report
    output_str = f"""PCA Feature Analysis Report\n
Best number of PCA components: {best_n_components}
Best {scoring}: {best_accuracy}
Train {scoring} with best model: {best_train_score}
Test {scoring} with best model: {best_test_score}
Logistic coefficients based on selected features:
{best_logistic_coefficients}"""
    
    print(output_str)
    # save report
    with open(output_file, "w") as f:
        f.write(output_str)

    # feature analysis
    # heatmap of PCA components
    plt.figure(figsize=(15, 15))
    sns.heatmap(best["components"], annot=True, cmap="crest", center=0)
    plt.title("PCA Components")
    plt.xlabel("Features")
    plt.ylabel("Principal Component Vectors")
    plt.yticks(np.arange(best["components"].shape[0]), [f"PC{i+1}" for i in range(best["components"].shape[0])])
    plt.xticks(np.arange(best["components"].shape[1]), feature_names, rotation=90)
    plt.savefig(plot_dir / "pca_heatmap.png")
    plt.show()

    # bar chart of variance explained by different PCA components
    best_pca = best_model.named_steps["pca"]
    bars =plt.bar(np.arange(len(best_pca.explained_variance_ratio_)), best_pca.explained_variance_ratio_)
    plt.xlabel("Principal Component Vectors")
    plt.ylabel("Ratio of Variance Explained")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(np.arange(best["components"].shape[0]), [f"PC{i+1}" for i in range(best["components"].shape[0])])
    plt.title("Ratio of Variance Explained by Principal Component Vectors")
    plt.savefig(plot_dir / "pca_var_bar.png")
    plt.show()

    return best

#%%
## Split dataset and save
# train_set, test_set = ml.read_dataset_and_split(DATA_PATH / "dataset/youtube_trending_labeled.csv", 
#                                                 test_size=0.3, 
#                                                 target_col=-1,
#                                                 use_csv=True)
# ml.save_split_dataset(train_set, test_set)

#%%
## Retrieve train set
X_train, y_train = ml.load_train_set() # (9711, 21) (9711)

X_test, y_test = ml.load_test_set() # (4163, 21) (4163)


og = pd.read_csv(DATA_PATH / "dataset/youtube_trending_labeled.csv", index_col=0)
og_X = og.iloc[:, :-1]
feature_names = np.array(og_X.columns)

#%%

best_lasso = lasso_feature_selection(X_train, 
                                     y_train, 
                                     X_test, 
                                     y_test, 
                                     feature_names, 
                                     output_file=HOME_DIR/"Analysis/lasso_feature_selection.txt",)
best_lasso_f1 = lasso_feature_selection(X_train, 
                                     y_train, 
                                     X_test, 
                                     y_test, 
                                     feature_names, 
                                     output_file=HOME_DIR/"Analysis/lasso_f1_feature_selection.txt",
                                     scoring="f1")
#%%
lasso_acc_transformer = best_lasso["model"].named_steps['lasso_selector'].transform
lasso_acc_transformer(X_train)
lasso_acc_transformer(X_test)

lasso_f1_transformer = best_lasso_f1["model"].named_steps["lasso_selector"].transform
lasso_f1_transformer(X_train)
lasso_f1_transformer(X_test)

best_lasso["model"].get_params()

#%%
best_pca = pca_feature_selection(X_train,
                                 y_train,
                                 X_test,
                                 y_test,
                                 feature_names,
                                 output_file=HOME_DIR/"Analysis/pca_feature_selection.txt",
                                 plot_dir=HOME_DIR/"Plots/ML")

best_pca_f1 = pca_feature_selection(X_train,
                                 y_train,
                                 X_test,
                                 y_test,
                                 feature_names,
                                 output_file=HOME_DIR/"Analysis/pca_f1_feature_selection.txt",
                                 plot_dir=HOME_DIR/"Plots/ML",
                                 scoring="f1")
#%%
