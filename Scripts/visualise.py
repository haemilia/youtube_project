#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import useful as use
import json
log_dir = Path("../LOG")
viz_dir = Path("../Visualisations")
model_dir = Path("../Models")
use.create_directory_if_not_exists(viz_dir)
score_dfs = {}
for score_df_path in log_dir.glob("best_pipeline_scores_*.pkl"):
    dataset_name = score_df_path.stem.split("_")[-1]
    score_dfs[dataset_name] = pd.read_pickle(score_df_path)

dataset_name_construction = {
    "0":"tabular_with_vlc",
    "1": "tabular_without_vlc",
    "2": "title",
    "3": "thumbnail",
    "4": "video"
}

best_models_acc = {}
best_models_f1 = {}

## Graph with val accuracy on x axis, and val f1 on y axis
# Identify unique classifiers and dimension reducers
# Get best performing model name per dataset...
for dataset_name, df in score_dfs.items():
    # Construct dataset full name
    full_name_list = [dataset_name_construction.get(cha, "") for cha in dataset_name]
    full_name = "-".join(full_name_list)
    # Identify categories
    classifiers = df['Classifier'].unique()
    reducers = df['Dimension Reducer'].unique()

    # Create color and marker mappings
    classifier_colors = {classifier: f'C{i}' for i, classifier in enumerate(classifiers)}
    reducer_markers = {reducer: marker for i, (reducer, marker) in enumerate(zip(reducers, ['o', 's']))}

    # Identify unique scalers
    scalers = df['Scaler'].unique()

    # Create scaler border color mapping
    scaler_borders = {scaler: f'gray' if scaler == 'NoScaler' else 'black' if scaler == 'MinMaxScaler' else 'red' for scaler in scalers}

    # Create the scatter plot with marker borders
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for index, row in df.iterrows():
        color = classifier_colors[row['Classifier']]
        marker = reducer_markers[row['Dimension Reducer']]
        border_color = scaler_borders[row['Scaler']]
        ax.scatter(row['Best Validation Accuracy'], row['Best Validation F1'], c=color, marker=marker, s=100, edgecolors=border_color, linewidths=1)

    # Add labels and title (same as before)
    ax.set_xlabel('Validation Accuracy')
    ax.set_ylabel('Validation F1-Score')
    ax.set_title(f'Validation Performance of {full_name}')

    # Create legend handles and labels for classifiers and reducers (same as before)
    classifier_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=classifier_colors[c], markersize=10, label=c) for c in classifiers]
    reducer_handles = [plt.Line2D([0], [0], marker=reducer_markers[r], color='k', linestyle='none', markersize=10, label=r) for r in reducers]

    # Add a legend for scalers
    scaler_handles = [plt.Line2D([0], [0], marker='o', color='w', markeredgecolor=scaler_borders[s], markersize=10, label=s, markeredgewidth=1) for s in scalers]

    # Combine and display the legends
    legend1 = ax.legend(handles=classifier_handles, title='Classifier', loc='upper left')
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=reducer_handles, title='Dimension Reducer', loc='center left')
    ax.add_artist(legend2)
    plt.legend(handles=scaler_handles, title='Scaler', loc='lower right')
    

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(viz_dir / f"val_performance_plot_{dataset_name}")
    plt.show()

    # Best performing models
    best_acc = df.sort_values(by='Best Validation Accuracy', ascending=False).iloc[0]
    scaler = best_acc["Scaler"]
    reducer = best_acc["Dimension Reducer"]
    classifier = best_acc["Classifier"]
    model_name = f"best_model_{scaler}_{reducer}_{classifier}_{dataset_name}.pkl"
    best_models_acc[dataset_name] = model_name
    best_f1 = df.sort_values(by='Best Validation Accuracy', ascending=False).iloc[0]
    scaler = best_f1["Scaler"]
    reducer = best_f1["Dimension Reducer"]
    classifier = best_f1["Classifier"]
    model_name = f"best_model_{scaler}_{reducer}_{classifier}_{dataset_name}.pkl"
    best_models_f1[dataset_name] = model_name

with open(log_dir / "best_models_acc.json", "w") as fw:
    json.dump(best_models_acc, fw)
with open(log_dir / "best_models_f1.json", "w") as fw:
    json.dump(best_models_f1, fw)