# Model Selection Report - dataset_02
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_1
- MinMaxScaler - NoReducer - XGBoost
- Validation Accuracy: 0.9199
- Validation F1-Score: 0.8842

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_1
- MinMaxScaler - NoReducer - XGBoost
- Validation Accuracy: 0.9199
- Validation F1-Score: 0.8842

# XGBoost Performance
- Pipelines: 1, 5, 8, 9
- Mean Validation Accuracy: 0.8856
- Mean Validation F1-Score: 0.8185

# RandomForest Performance
- Pipelines: 2, 7
- Mean Validation Accuracy: 0.9058
- Mean Validation F1-Score: 0.8585

# LightGBM Performance
- Pipelines: 3, 4, 6, 10
- Mean Validation Accuracy: 0.8842
- Mean Validation F1-Score: 0.8176

---

# Pipeline_1
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 1.0
- n_estimators : 200
- max_depth : 5
- learning_rate : 0.1
- colsample_bytree : 1.0
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9199
- F1-Score: 0.8842

# Pipeline_2
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 200
- min_samples_split : 2
- min_samples_leaf : 3
- max_depth : 20
- n_components : 110
## Best Performance
- Train
- Accuracy: 0.9978
- F1-Score: 0.9968
- Validation
- Accuracy: 0.9004
- F1-Score: 0.8545

# Pipeline_3
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: LightGBM
## Hyperparameters
- num_leaves : 50
- n_estimators : 50
- max_depth : -1
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9145
- F1-Score: 0.8775

# Pipeline_4
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 310
- num_leaves : 50
- n_estimators : 200
- max_depth : 10
- learning_rate : 0.3
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.7933
- F1-Score: 0.6403

# Pipeline_5
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 200
- max_depth : 3
- learning_rate : 0.1
- colsample_bytree : 0.8
- n_components : 60
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9156
- F1-Score: 0.8801

# Pipeline_6
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: LightGBM
## Hyperparameters
- num_leaves : 31
- n_estimators : 100
- max_depth : -1
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9156
- F1-Score: 0.8775

# Pipeline_7
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 50
- min_samples_split : 5
- min_samples_leaf : 1
- max_depth : 10
## Best Performance
- Train
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.9113
- F1-Score: 0.8625

# Pipeline_8
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 200
- max_depth : 7
- learning_rate : 0.1
- colsample_bytree : 0.8
- n_components : 210
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.7868
- F1-Score: 0.6253

# Pipeline_9
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 1.0
- n_estimators : 200
- max_depth : 5
- learning_rate : 0.1
- colsample_bytree : 1.0
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9199
- F1-Score: 0.8842

# Pipeline_10
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 10
- num_leaves : 100
- n_estimators : 50
- max_depth : -1
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.9134
- F1-Score: 0.8752
