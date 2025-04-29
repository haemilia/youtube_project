# Model Selection Report - dataset_123
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_9
- MinMaxScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8536
- Validation F1-Score: 0.7581

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_9
- MinMaxScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8536
- Validation F1-Score: 0.7581

# RandomForest Performance
- Pipelines: 1, 6
- Mean Validation Accuracy: 0.7993
- Mean Validation F1-Score: 0.5985

# LightGBM Performance
- Pipelines: 2, 7, 8, 9
- Mean Validation Accuracy: 0.8438
- Mean Validation F1-Score: 0.7454

# XGBoost Performance
- Pipelines: 3, 4, 5, 10
- Mean Validation Accuracy: 0.8438
- Mean Validation F1-Score: 0.7400

---

# Pipeline_1
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
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.7983
- F1-Score: 0.5881

# Pipeline_2
## Pipeline Composition
- Scaler: NoScaler
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
- Accuracy: 0.8341
- F1-Score: 0.7203

# Pipeline_3
## Pipeline Composition
- Scaler: MinMaxScaler
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
- Accuracy: 0.8492
- F1-Score: 0.7563

# Pipeline_4
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 50
- max_depth : 7
- learning_rate : 0.1
- colsample_bytree : 0.8
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8438
- F1-Score: 0.7377

# Pipeline_5
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 50
- max_depth : 7
- learning_rate : 0.1
- colsample_bytree : 0.8
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8438
- F1-Score: 0.7377

# Pipeline_6
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- min_samples_split : 2
- min_samples_leaf : 3
- max_depth : None
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8004
- F1-Score: 0.6088

# Pipeline_7
## Pipeline Composition
- Scaler: MinMaxScaler
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
- Accuracy: 0.9967
- F1-Score: 0.9951
- Validation
- Accuracy: 0.8395
- F1-Score: 0.7537

# Pipeline_8
## Pipeline Composition
- Scaler: NoScaler
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
- Accuracy: 0.8482
- F1-Score: 0.7494

# Pipeline_9
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
- Accuracy: 0.8536
- F1-Score: 0.7581

# Pipeline_10
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
- Accuracy: 0.8384
- F1-Score: 0.7285
