import itertools
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pickle
import numpy as np

# Placeholder for the Keras model builder function
def build_fc_nn(input_shape):
    # You will replace this with your actual Keras model definition
    from tensorflow import keras
    from keras import layers
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Assuming binary classification for now
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Read data
dataset_name = ""
#### dataset name
# 0: tabular_with_vlc
# 1: tabular_without_vlc
# 2: title
# 3: thumbnail
# 4: video_l1
# 5: video
import dataset
X, y = dataset.construct_final_dataset(dataset_name)

# Wrap the Keras model for scikit-learn compatibility
from scikeras.wrappers import KerasClassifier

# Define the components
scalers = [('NoScaler', None), ('MinMaxScaler', MinMaxScaler())]
dimension_reducers = [('NoReducer', None), ('PCA', PCA()), ('LassoSelector', SelectFromModel(Lasso(max_iter=10000)))]
classifiers = [
    ('FC_NN', lambda: KerasClassifier(model=build_fc_nn, verbose=0)),
    ('RandomForest', RandomForestClassifier()),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

# Define hyperparameter grids for RandomizedSearchCV
param_grids = {
    'PCA': {'n_components': np.arange(1, 10)}, # Example range, adjust as needed
    'LassoSelector': {'threshold': np.logspace(-4, 0, 10)}, # Example range
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]},
    'FC_NN': {'model__epochs': [10, 20], 'model__batch_size': [32, 64]} # Keras wrapper uses 'model__' prefix
}

# Generate all possible pipeline combinations
pipelines = list(itertools.product(scalers, dimension_reducers, classifiers))

# Store results
results = []
best_models = {}
pipeline_number = 0

# Create StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Starting pipeline evaluation...")



for scaler, reducer, classifier in pipelines:
    pipeline_number += 1
    pipeline_name = f"Pipeline_{pipeline_number}"
    print(f"\nEvaluating {pipeline_name}: {scaler[0]} - {reducer[0]} - {classifier[0]}")

    steps = []
    if scaler[1]:
        steps.append(scaler)
    if reducer[1]:
        steps.append(reducer)
    # Handle the KerasClassifier separately for hyperparameter grid access
    if classifier[0] == 'FC_NN':
        steps.append(('FC_NN', classifier[1]()))
        param_grid = param_grids.get(classifier[0], {})
    else:
        steps.append(classifier)
        param_grid = param_grids.get(classifier[0], {})

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
        with open(filename, 'wb') as file:
            pickle.dump(best_model, file)
        print(f"  Best model saved as {filename}")

    except Exception as e:
        print(f"  Error during training: {e}")

# Create the DataFrame of best scores
results_df = pd.DataFrame(results).set_index('Pipeline Number')
print("\nDataFrame of best scores:")
print(results_df)

# Save the DataFrame to pickle
results_df_path = f"best_pipeline_scores{dataset_name}.pkl"
results_df.to_pickle(results_df_path)
print(f"\nBest pipeline scores saved to {results_df_path}")

# Generate the markdown report
markdown_report = "# Best Performance by Validation Accuracy\n"
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
    markdown_report += f"- F1-Score: ${row['Best Train F1']:.4f}$\n"
    markdown_report += "- Validation\n"
    markdown_report += f"- Accuracy: {row['Best Validation Accuracy']:.4f}\n"
    markdown_report += f"- F1-Score: ${row['Best Validation F1']:.4f}$\n"

# Save the report to a markdown file
report_path = f"{dataset_name}_pipeline_comparison_report.md"
with open(report_path, 'w') as f:
    f.write(markdown_report)

print(f"\nMarkdown report saved to {report_path}")
print("\nReport Content:")
print(markdown_report)