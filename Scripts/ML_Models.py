#%%
import pathlib
import json
with open("../private.json") as json_file:
    priv = json.load(json_file)
DATA_PATH = pathlib.Path(priv["DATA_PATH"])
HOME_DIR = pathlib.Path(priv["HOME_DIR"])
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import accuracy_score, f1_score
import pickle
#%%
from itertools import product
minmax_scaler = MinMaxScaler()
pca = PCA()
lasso_selector = SelectFromModel(Lasso(max_iter=10000))
lr = LogisticRegression(solver='liblinear')
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
knn = KNeighborsClassifier()
object_scalers = [None, minmax_scaler]
object_feature_selectors = [None, pca, lasso_selector]
object_classifiers = {"logistic": lr, 
                      "random_forest": rf, 
                      "gradient_boost": gb,
                      "knn": knn}
scaler_ids = {
    "logistic": [1],
    "knn":[1],
    "random_forest": [0],
    "gradient_boost": [0, 1]
}
feature_selector_ids = {
    "gradient_boost": [0, 1, 2],
    "random_forest": [0, 1, 2],
    "logistic": [1, 2],
    "knn": [1, 2]
}
feature_params = {0: [],
    2: [('feature_selection__estimator__alpha', loguniform(1e-4, 1)), ('feature_selection__estimator', [Lasso()])],
    1: [('feature_selection__n_components', randint(2, 20))],}
classifier_params = {
    'logistic':[("classifier__C", loguniform(1e-4, 100))],
    'random_forest': [("classifier__n_estimators", randint(50, 500)), 
                      ("classifier__max_features", uniform(0.1, 0.9))],
    'gradient_boost': [("classifier__learning_rate", loguniform(1e-3, 1)),
                       ("classifier__n_estimators", randint(50, 500))],
    'knn': [("classifier__n_neighbors", randint(3, 20))]
}
def construct_pipelines(object_scalers=object_scalers, 
                        object_feature_selectors=object_feature_selectors,
                        object_classifiers=object_classifiers,
                        scaler_ids=scaler_ids,
                        feature_selector_ids=feature_selector_ids,
                        feature_params=feature_params,
                        classifier_params=classifier_params,):
    def retrieve_for_pipeline(si, fi, classifier_obj):
        pipeline = []
        if si > 0:
            pipeline.append(("scaler", object_scalers[si]))
        if fi > 0:
            pipeline.append(("feature_selection", object_feature_selectors[fi]))
        return pipeline
    def retrieve_for_param(fi, classifier_id):
        params = {}
        for name, dist in feature_params[fi]:
            params[name] = dist
        for name, dist in classifier_params[classifier_id]:
            params[name] = dist
        return params
    pipelines = []
    for classifier_id, classifier_obj in object_classifiers.items():
        scaler = scaler_ids[classifier_id]
        feature_selector = feature_selector_ids[classifier_id]
        combinations = list(product(scaler, feature_selector))
        for si, fi in combinations:
            print(si, fi)
            indiv = {}
            indiv["pipeline"] = retrieve_for_pipeline(si, fi, classifier_obj)
            indiv["pipeline"].append(("classifier", classifier_obj))
            indiv["params"] = retrieve_for_param(fi, classifier_id)
            pipelines.append(indiv)
    return pipelines
#%%
pipelines = construct_pipelines()


#%%
import ML_convenience as ml
X_train, y_train = ml.load_train_set() # (9711, 21) (9711)

X_test, y_test = ml.load_test_set() # (4163, 21) (4163)
X_test.shape
#%%
test_pipeline = pipelines[1]
splitter = StratifiedKFold(n_splits=6, shuffle=True, random_state=5)
pipe = Pipeline(test_pipeline["pipeline"])
rcv = RandomizedSearchCV(pipe,
                         test_pipeline["params"],
                         scoring="accuracy",
                         n_jobs=-1,
                         cv=splitter)
rcv.fit(X_train, y_train)
# %%
pipelines = construct_pipelines()
for i, p in enumerate(pipelines):
    pipe = Pipeline(p["pipeline"])
    splitter = StratifiedKFold(n_splits=6, shuffle=True, random_state=5)
    rcv = RandomizedSearchCV(pipe,
                            p["params"],
                            scoring="accuracy",
                            n_jobs=-1,
                            cv=splitter)
    rcv.fit(X_train, y_train)
    print(i, rcv.best_score_)
    with open(HOME_DIR / f"Models/randomized_search_{i}.pkl", "wb") as f:
        pickle.dump(rcv, f)
#%%
pipelines = construct_pipelines()
for i, p in enumerate(pipelines):
    pipe = Pipeline(p["pipeline"])
    splitter = StratifiedKFold(n_splits=6, shuffle=True, random_state=5)
    rcv = RandomizedSearchCV(pipe,
                            p["params"],
                            scoring="f1",
                            n_jobs=-1,
                            cv=splitter)
    rcv.fit(X_train, y_train)
    print(i, rcv.best_score_)
    with open(HOME_DIR / f"Models/randomized_search_f1_{i}.pkl", "wb") as f:
        pickle.dump(rcv, f)


#%%
with open(HOME_DIR / "Models/randomized_search_2.pkl", "rb") as f: #load the first model.
    loaded_rcv = pickle.load(f)

best_model = loaded_rcv.best_estimator_
best_model.named_steps.get("feature_selection")
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import ML_convenience as ml
X_train, y_train = ml.load_train_set() # (9711, 21) (9711)
X_test, y_test = ml.load_test_set() # (4163, 21) (4163)
og = pd.read_csv(DATA_PATH / "dataset/youtube_trending_labeled.csv", index_col=0)
og_X = og.iloc[:, :-1]
feature_names = np.array(og_X.columns)

def analyze_pickled_models(all_models, X_train, y_train, X_test, y_test, feature_names, output_txt_dir=HOME_DIR/"Analysis", plot_dir=HOME_DIR/"Plots/ML" ):
    all_train_accuracy = []
    all_train_f1 = []
    all_test_accuracy = []
    all_test_f1 = []
    model_cats = {}
    model_cats["pca"] = np.zeros(len(all_models))
    model_cats["lasso"] = np.zeros(len(all_models))
    model_cats["logistic"] = np.zeros(len(all_models))
    model_cats["random_forest"] = np.zeros(len(all_models))
    model_cats["knn"] = np.zeros(len(all_models))
    model_cats["gradient_boost"] = np.zeros(len(all_models))
    output_path = Path(output_txt_dir / "analysis_models.txt")
    with output_path.open("w") as f_out:  # Open the output file for writing
        for i, model_path_str in enumerate(all_models):
            model_path = Path(model_path_str)
            try:
                with model_path.open("rb") as f:
                    rcv = pickle.load(f)

                f_out.write(f"\nAnalysis of model: {model_path}\n")
                print(f"\nAnalysis of model: {model_path}")

                # Model Structure
                f_out.write(f"Model structure: {rcv.best_estimator_.steps}\n")
                print(f"Model structure: {rcv.best_estimator_.steps}")

                # Train and Test Metrics
                y_train_pred = rcv.predict(X_train)
                y_test_pred = rcv.predict(X_test)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                all_test_accuracy.append(test_accuracy)
                all_test_f1.append(test_f1)
                all_train_accuracy.append(train_accuracy)
                all_train_f1.append(train_f1)

                f_out.write(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}\n")
                f_out.write(f"Train F1-score: {train_f1}, Test F1-score: {test_f1}\n")
                print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
                print(f"Train F1-score: {train_f1}, Test F1-score: {test_f1}")

                # Feature Selector Analysis
                selected_features = feature_names  # Initialize with all feature names
                principal_vectors = None
                for name, step in rcv.best_estimator_.steps:
                    if isinstance(step, SelectFromModel) and isinstance(step.estimator, Lasso):
                        model_cats["lasso"][i] = 1
                        alpha = step.estimator.get_params()["alpha"]
                        f_out.write(f"Feature Selector: Lasso, Alpha: {alpha}\n")
                        print(f"Feature Selector: Lasso, Alpha: {alpha}")

                        lasso_coef = step.estimator_.coef_
                        selected_features = [feature_names[i] for i in range(len(lasso_coef)) if lasso_coef[i] != 0]

                        plt.figure()
                        plt.bar(selected_features, lasso_coef[lasso_coef != 0])
                        plt.xticks(rotation=90)
                        plt.title(f"Lasso Coefficients (Alpha={alpha})")
                        plt.tight_layout()
                        plt.savefig(plot_dir / f"{model_path.stem}_lasso_coef.png")
                        plt.close()

                        f_out.write(f"Selected Features (Lasso): {selected_features}\n")
                        print(f"Selected Features (Lasso): {selected_features}")

                    elif isinstance(step, PCA):
                        model_cats["pca"][i] = 1
                        n_components = step.n_components_
                        f_out.write(f"Feature Selector: PCA, Components: {n_components}\n")
                        print(f"Feature Selector: PCA, Components: {n_components}")

                        component_matrix = step.components_
                        plt.figure(figsize=(15, 15))
                        sns.heatmap(component_matrix, annot=True, cmap="viridis")
                        plt.title(f"PCA Component Matrix (Components={n_components})")
                        plt.xticks(np.arange(len(feature_names)), feature_names, rotation=90)
                        plt.tight_layout()
                        plt.savefig(plot_dir / f"{model_path.stem}_pca_matrix.png")
                        plt.close()

                        explained_variance_ratio = step.explained_variance_ratio_
                        plt.figure()
                        plt.bar(range(1, n_components + 1), explained_variance_ratio)
                        plt.xticks(range(1, n_components + 1))
                        plt.xlabel("Principal Component")
                        plt.ylabel("Explained Variance Ratio")
                        plt.title(f"PCA Explained Variance (Components={n_components})")
                        plt.tight_layout()
                        plt.savefig(plot_dir / f"{model_path.stem}_pca_variance.png")
                        plt.close()

                        principal_vectors = component_matrix

                        f_out.write(f"PCA Components: {n_components}\n")
                        print(f"PCA Components: {n_components}")


                # Classifier Analysis
                classifier = rcv.best_estimator_.named_steps["classifier"]
                if isinstance(classifier, LogisticRegression):
                    model_cats["logistic"][i] = 1
                    c_value = classifier.C
                    coef = classifier.coef_
                    intercept = classifier.intercept_
                    f_out.write(f"Classifier: Logistic Regression, C: {c_value}\n")
                    f_out.write(f"Coefficients: {coef}, Intercept: {intercept}\n")
                    print(f"Classifier: Logistic Regression, C: {c_value}")
                    print(f"Coefficients: {coef}, Intercept: {intercept}")
                elif isinstance(classifier, KNeighborsClassifier):
                    model_cats["knn"][i] = 1
                    n_neighbors = classifier.n_neighbors
                    f_out.write(f"Classifier: KNN, Neighbors: {n_neighbors}\n")
                    print(f"Classifier: KNN, Neighbors: {n_neighbors}")
                elif isinstance(classifier, RandomForestClassifier):
                    model_cats["random_forest"][i] = 1
                    n_estimators = classifier.n_estimators
                    max_features = classifier.max_features
                    if len(classifier.feature_importances_) == len(feature_names):
                        x_vals = feature_names
                    else:
                        x_vals = np.arange(len(classifier.feature_importances_))
                    plt.figure(tight_layout=True)
                    plt.bar(x_vals, classifier.feature_importances_)
                    plt.title("Random Forest Feature Importances")
                    plt.xticks(rotation=90)
                    plt.savefig(plot_dir / f"{model_path.stem}_rf_feature_importance.png")
                    f_out.write(f"Classifier: Random Forest, Estimators: {n_estimators}, Max Features: {max_features}\n")
                    print(f"Classifier: Random Forest, Estimators: {n_estimators}, Max Features: {max_features}")
                elif isinstance(classifier, GradientBoostingClassifier):
                    model_cats["gradient_boost"][i] = 1
                    learning_rate = classifier.learning_rate
                    n_estimators = classifier.n_estimators
                    if len(classifier.feature_importances_) == len(feature_names):
                        x_vals = feature_names
                    else:
                        x_vals = np.arange(len(classifier.feature_importances_))
                    plt.bar(x_vals, classifier.feature_importances_)
                    plt.title("Gradient Boosting Feature Importances")
                    plt.xticks(rotation=90)
                    plt.savefig(plot_dir / f"{model_path.stem}_gb_feature_importance.png")
                    f_out.write(f"Classifier: Gradient Boosting, Learning Rate: {learning_rate}, Estimators: {n_estimators}\n")
                    print(f"Classifier: Gradient Boosting, Learning Rate: {learning_rate}, Estimators: {n_estimators}")


            except Exception as e:
                f_out.write(f"Error processing {model_path}: {e}\n")
                print(f"Error processing {model_path}: {e}")
    return np.array(all_train_accuracy), np.array(all_test_accuracy), np.array(all_train_f1), np.array(all_test_f1), model_cats
all_models = []
for model_file in (HOME_DIR / "Models").iterdir():
    if model_file.is_dir():
        continue
    else:
        all_models.append(model_file)
train_acc, test_acc, train_f1, test_f1, model_cats = analyze_pickled_models(all_models, X_train, y_train, X_test, y_test,
                       feature_names)
model_cats
# %%
train_acc_pca = train_acc[model_cats["pca"] == 1]
test_acc_pca = test_acc[model_cats["pca"] == 1]
train_acc_lasso = train_acc[model_cats["lasso"] == 1]
test_acc_lasso = test_acc[model_cats["lasso"]==1]
other_f = model_cats["pca"] + model_cats["lasso"] == 0
train_acc_otherf = train_acc[other_f]
test_acc_otherf = test_acc[other_f]
plt.scatter(train_acc_pca, test_acc_pca, label="PCA")
plt.scatter(train_acc_lasso, test_acc_lasso, label="Lasso")
plt.scatter(train_acc_otherf, test_acc_otherf, label="No Feature Selection")
plt.title("Train Accuracy vs Test Accuracy")
plt.xlabel("Train Accuracy")
plt.ylabel("Test Accuracy")
plt.legend()
plt.savefig(HOME_DIR/"Plots/ML/Train_Test_Accuracy_F.png")
plt.show()

train_acc_log = train_acc[model_cats["logistic"] == 1]
test_acc_log = test_acc[model_cats["logistic"] == 1]
train_acc_knn = train_acc[model_cats["knn"] == 1]
test_acc_knn = test_acc[model_cats["knn"]==1]
train_acc_rf = train_acc[model_cats["random_forest"] == 1]
test_acc_rf = test_acc[model_cats["random_forest"]==1]
train_acc_gb = train_acc[model_cats["gradient_boost"] == 1]
test_acc_gb = test_acc[model_cats["gradient_boost"] == 1]
plt.scatter(train_acc_log, test_acc_log, label="Logistic Regression")
plt.scatter(train_acc_knn, test_acc_knn, label="K-NN")
plt.scatter(train_acc_rf, test_acc_rf, label="Random Forest")
plt.scatter(train_acc_gb, test_acc_gb, label="Gradient Boost")
plt.title("Train Accuracy vs Test Accuracy")
plt.xlabel("Train Accuracy")
plt.ylabel("Test Accuracy")
plt.legend()
plt.savefig(HOME_DIR/"Plots/ML/Train_Test_Accuracy_C.png")
plt.show()

train_acc_pca = train_f1[model_cats["pca"] == 1]
test_acc_pca = test_f1[model_cats["pca"] == 1]
train_acc_lasso = train_f1[model_cats["lasso"] == 1]
test_acc_lasso = test_f1[model_cats["lasso"]==1]
other_f = model_cats["pca"] + model_cats["lasso"] == 0
train_acc_otherf = train_f1[other_f]
test_acc_otherf = test_f1[other_f]
plt.scatter(train_acc_pca, test_acc_pca, label="PCA")
plt.scatter(train_acc_lasso, test_acc_lasso, label="Lasso")
plt.scatter(train_acc_otherf, test_acc_otherf, label="No Feature Selection")
plt.title("Train F1 vs Test F1")
plt.xlabel("Train F1")
plt.ylabel("Test F1")
plt.legend()
plt.savefig(HOME_DIR/"Plots/ML/Train_Test_F1_F.png")
plt.show()

train_acc_log = train_f1[model_cats["logistic"] == 1]
test_acc_log = test_f1[model_cats["logistic"] == 1]
train_acc_knn = train_f1[model_cats["knn"] == 1]
test_acc_knn = test_f1[model_cats["knn"]==1]
train_acc_rf = train_f1[model_cats["random_forest"] == 1]
test_acc_rf = test_f1[model_cats["random_forest"]==1]
train_acc_gb = train_f1[model_cats["gradient_boost"] == 1]
test_acc_gb = test_f1[model_cats["gradient_boost"] == 1]
plt.scatter(train_acc_log, test_acc_log, label="Logistic Regression")
plt.scatter(train_acc_knn, test_acc_knn, label="K-NN")
plt.scatter(train_acc_rf, test_acc_rf, label="Random Forest")
plt.scatter(train_acc_gb, test_acc_gb, label="Gradient Boost")
plt.title("Train F1 vs Test F1")
plt.xlabel("Train F1")
plt.ylabel("Test F1")
plt.legend()
plt.savefig(HOME_DIR/"Plots/ML/Train_Test_F1_C.png")
plt.show()
# %%
