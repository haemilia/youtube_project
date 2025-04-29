# Model Selection Report - dataset_0
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_2
- NoScaler - NoReducer - RandomForest
- Validation Accuracy: 0.9232
- Validation F1-Score: 0.8893

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_2
- NoScaler - NoReducer - RandomForest
- Validation Accuracy: 0.9232
- Validation F1-Score: 0.8893

# LightGBM Performance
- Pipelines: 1, 4, 5, 7
- Mean Validation Accuracy: 0.8880
- Mean Validation F1-Score: 0.8283

# RandomForest Performance
- Pipelines: 2, 9
- Mean Validation Accuracy: 0.9215
- Mean Validation F1-Score: 0.8877

# XGBoost Performance
- Pipelines: 3, 6, 8, 10
- Mean Validation Accuracy: 0.8907
- Mean Validation F1-Score: 0.8314

---

# Pipeline_1
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 10
- num_leaves : 50
- n_estimators : 50
- max_depth : -1
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.9210
- F1-Score: 0.8862

# Pipeline_2
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 50
- min_samples_split : 5
- min_samples_leaf : 3
- max_depth : None
## Best Performance
- Train
- Accuracy: 0.9708
- F1-Score: 0.9573
- Validation
- Accuracy: 0.9232
- F1-Score: 0.8893

# Pipeline_3
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 50
- max_depth : 7
- learning_rate : 0.1
- colsample_bytree : 0.8
- n_components : 10
## Best Performance
- Train
- Accuracy: 0.9502
- F1-Score: 0.9218
- Validation
- Accuracy: 0.8074
- F1-Score: 0.6834

# Pipeline_4
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: LightGBM
## Hyperparameters
- num_leaves : 31
- n_estimators : 100
- max_depth : 5
- learning_rate : 0.01
## Best Performance
- Train
- Accuracy: 0.9405
- F1-Score: 0.9120
- Validation
- Accuracy: 0.9167
- F1-Score: 0.8747

# Pipeline_5
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: LightGBM
## Hyperparameters
- num_leaves : 50
- n_estimators : 200
- max_depth : 5
- learning_rate : 0.3
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9167
- F1-Score: 0.8774

# Pipeline_6
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 1.0
- n_estimators : 100
- max_depth : 5
- learning_rate : 0.3
- colsample_bytree : 0.8
- n_components : 10
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9177
- F1-Score: 0.8787

# Pipeline_7
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 10
- num_leaves : 50
- n_estimators : 50
- max_depth : -1
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 0.9719
- F1-Score: 0.9568
- Validation
- Accuracy: 0.7976
- F1-Score: 0.6749

# Pipeline_8
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 200
- max_depth : 5
- learning_rate : 0.01
- colsample_bytree : 0.8
## Best Performance
- Train
- Accuracy: 0.9665
- F1-Score: 0.9509
- Validation
- Accuracy: 0.9188
- F1-Score: 0.8818

# Pipeline_9
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- min_samples_split : 2
- min_samples_leaf : 1
- max_depth : 10
- n_components : 10
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9199
- F1-Score: 0.8861

# Pipeline_10
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 200
- max_depth : 5
- learning_rate : 0.01
- colsample_bytree : 0.8
## Best Performance
- Train
- Accuracy: 0.9665
- F1-Score: 0.9509
- Validation
- Accuracy: 0.9188
- F1-Score: 0.8818
