import itertools
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pickle
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# Get dataset_name from CLI if provided, otherwise use input()
if len(sys.argv) > 1:
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate machine learning pipelines for a given dataset.")
    parser.add_argument("dataset_name", help="Name of the dataset to use (0:w/ vlc, 1:w/o vlc, 2: title, 3: thumbnail, 4: l1, 5: video)")
    args = parser.parse_args()
    dataset_name = args.dataset_name
else:
    dataset_name = input("Enter the dataset name (0:w/ vlc, 1:w/o vlc, 2: title, 3: thumbnail, 4: l1, 5: video): ")

# Read data
#### dataset name
# 0: tabular_with_vlc
# 1: tabular_without_vlc
# 2: title
# 3: thumbnail
# 4: video_l1
# 5: video
from dataset import dataset_construction
import useful as use
DATA_PATH = Path(use.get_priv()["DATA_PATH"])
locations = {
    "tabular_features_vlc": DATA_PATH / "dataset/tabular_features_vlc.pkl",
    "tabular_features_no_vlc": DATA_PATH / "dataset/tabular_features_no_vlc.pkl",
    "title_features": DATA_PATH/"dataset/title_features.pkl",
    "thumbnail_features": DATA_PATH/"dataset/thumbnail_features.pkl",
}
X, y = dataset_construction(dataset_name, locations)
print("Shape of dataset:")
print(X.shape, y.shape)

# Paths to save everything
models_dir = Path("../Models")
use.create_directory_if_not_exists(models_dir)
report_dir = Path("../LOG")
use.create_directory_if_not_exists(report_dir)


# Define the pipeline components
scalers = [('NoScaler', None), ('MinMaxScaler', MinMaxScaler())]
dimension_reducers = [('NoReducer', None), ('PCA', PCA(random_state=42))] # Added random_state for reproducibility
classifiers = [
    ('RandomForest', RandomForestClassifier(random_state=42)), # Added random_state for reproducibility
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)), # Added random_state for reproducibility
    ('LightGBM', LGBMClassifier(random_state=42)) # Added LightGBM with random_state
]

# Define hyperparameter grids for RandomizedSearchCV
param_grids = {
    'PCA': {'n_components': np.arange(10, min(X.shape[1], 501), 50)}, # Wider range for PCA
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 3, 5]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]},
    'LightGBM': {'n_estimators': [50, 100, 200], 'max_depth': [-1, 5, 10], 'learning_rate': [0.01, 0.1, 0.3], 'num_leaves': [31, 50, 100]}
}
# Generate all possible pipeline combinations
pipeline_set = set()
for reducer in dimension_reducers:
    for classifier in classifiers:
        if classifier[0] == 'RandomForest':
            pipeline_set.add((('NoScaler', None), reducer, classifier))
        else:
            for scaler in scalers:
                pipeline_set.add((scaler, reducer, classifier))

# Convert the set of unique pipelines back to a list
pipelines = list(pipeline_set)

# Store results
results = []
best_models = {}
pipeline_number = 0

# Create StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Starting pipeline evaluation...")



for scaler, reducer, classifier in pipelines:
    pipeline_number += 1
    pipeline_name = f"Pipeline_{pipeline_number}"
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nEvaluating {pipeline_name}: {scaler[0]} - {reducer[0]} - {classifier[0]}")

    steps = []
    if scaler[1]:
        steps.append(scaler)
    if reducer[1]:
        steps.append(reducer)
    steps.append(classifier)

    pipeline = Pipeline(steps)

    # Get relevant hyperparameters for the current pipeline
    search_params = {}
    for name, step in pipeline.named_steps.items():
        if name in param_grids:
            search_params[name + '__' + list(param_grids[name].keys())[0]] = list(param_grids[name].values())[0]
            for key, value in param_grids[name].items():
                search_params[name + '__' + key] = value

    random_search = RandomizedSearchCV(pipeline, param_distributions=search_params,
                                       cv=cv, scoring=['accuracy', 'f1'], refit='accuracy', n_iter=10, random_state=42, error_score='raise') # Increased n_iter for better exploration
    try:
        random_search.fit(X, y)

        best_model = random_search.best_estimator_
        train_accuracy = accuracy_score(y, best_model.predict(X))
        train_f1 = f1_score(y, best_model.predict(X))
        val_accuracy = random_search.best_score_
        val_f1 = random_search.cv_results_['mean_test_f1'][random_search.best_index_]

        results.append({
            'Pipeline Number': pipeline_number,
            'Scaler': scaler[0],
            'Dimension Reducer': reducer[0],
            'Classifier': classifier[0],
            'Best Train Accuracy': train_accuracy,
            'Best Validation Accuracy': val_accuracy,
            'Best Train F1': train_f1,
            'Best Validation F1': val_f1,
            'Best Hyperparameters': random_search.best_params_
        })
        best_models[pipeline_name] = best_model

        # Save the best model for this pipeline structure
        filename = f"best_model_{scaler[0]}_{reducer[0]}_{classifier[0]}_{dataset_name}.pkl"
        with open(models_dir / filename, 'wb') as file:
            pickle.dump(best_model, file)
        print(f"  Best model saved as {filename}")

    except Exception as e:
        print(f"  Error during training: {e}")

# Create the DataFrame of best scores
results_df = pd.DataFrame(results).set_index('Pipeline Number')
print("\nDataFrame of best scores:")
print(results_df)

# Save the DataFrame to pickle
results_df_path = f"best_pipeline_scores_{dataset_name}.pkl"
results_df.to_pickle(report_dir / results_df_path)
print(f"\nBest pipeline scores saved to {results_df_path}")

# Generate the markdown report
markdown_report = f"# Model Selection Report - dataset_{dataset_name}\n"
markdown_report += "# Best Performance by Validation Accuracy\n"
best_accuracy = results_df.sort_values(by='Best Validation Accuracy', ascending=False).iloc[0]
markdown_report += f"- Pipeline Name: Pipeline_{best_accuracy.name}\n"
markdown_report += f"- {best_accuracy['Scaler']} - {best_accuracy['Dimension Reducer']} - {best_accuracy['Classifier']}\n"
markdown_report += f"- Validation Accuracy: {best_accuracy['Best Validation Accuracy']:.4f}\n"
markdown_report += f"- Validation F1-Score: {best_accuracy['Best Validation F1']:.4f}\n"

markdown_report += "\n# Best Performance by Validation F1-Score\n"
best_f1 = results_df.sort_values(by='Best Validation F1', ascending=False).iloc[0]
markdown_report += f"- Pipeline Name: Pipeline_{best_f1.name}\n"
markdown_report += f"- {best_f1['Scaler']} - {best_f1['Dimension Reducer']} - {best_f1['Classifier']}\n"
markdown_report += f"- Validation Accuracy: {best_f1['Best Validation Accuracy']:.4f}\n"
markdown_report += f"- Validation F1-Score: {best_f1['Best Validation F1']:.4f}\n"

# Performance per classifier
classifier_types = results_df['Classifier'].unique()
for classifier_name in classifier_types:
    classifier_df = results_df[results_df['Classifier'] == classifier_name]
    pipeline_numbers = ', '.join(map(str, classifier_df.index.tolist()))
    mean_val_accuracy = classifier_df['Best Validation Accuracy'].mean()
    mean_val_f1 = classifier_df['Best Validation F1'].mean()

    markdown_report += f"\n# {classifier_name} Performance\n"
    markdown_report += f"- Pipelines: {pipeline_numbers}\n"
    markdown_report += f"- Mean Validation Accuracy: {mean_val_accuracy:.4f}\n"
    markdown_report += f"- Mean Validation F1-Score: {mean_val_f1:.4f}\n"

markdown_report += "\n---\n"

# Individual pipeline reports
for index, row in results_df.iterrows():
    markdown_report += f"\n# Pipeline_{index}\n"
    markdown_report += "## Pipeline Composition\n"
    markdown_report += f"- Scaler: {row['Scaler']}\n"
    markdown_report += f"- Dimension Reduction: {row['Dimension Reducer']}\n"
    markdown_report += f"- Classifier: {row['Classifier']}\n"
    markdown_report += "## Hyperparameters\n"
    for key, value in row['Best Hyperparameters'].items():
        markdown_report += f"- {key.split('__')[-1]} : {value}\n"
    markdown_report += "## Best Performance\n"
    markdown_report += "- Train\n"
    markdown_report += f"- Accuracy: {row['Best Train Accuracy']:.4f}\n"
    markdown_report += f"- F1-Score: {row['Best Train F1']:.4f}\n"
    markdown_report += "- Validation\n"
    markdown_report += f"- Accuracy: {row['Best Validation Accuracy']:.4f}\n"
    markdown_report += f"- F1-Score: {row['Best Validation F1']:.4f}\n"

# Save the report to a markdown file
report_path = f"{dataset_name}_pipeline_comparison_report.md"
with open(report_dir / report_path, 'w') as f:
    f.write(markdown_report)

print(f"\nMarkdown report saved to {report_path}")
print("\nReport Content:")
print(markdown_report)