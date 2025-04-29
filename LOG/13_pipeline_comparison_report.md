# Model Selection Report - dataset_13
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_6
- MinMaxScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8481
- Validation F1-Score: 0.7548

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_6
- MinMaxScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8481
- Validation F1-Score: 0.7548

# XGBoost Performance
- Pipelines: 1, 5, 7, 10
- Mean Validation Accuracy: 0.8416
- Mean Validation F1-Score: 0.7350

# RandomForest Performance
- Pipelines: 2, 4
- Mean Validation Accuracy: 0.8096
- Mean Validation F1-Score: 0.6334

# LightGBM Performance
- Pipelines: 3, 6, 8, 9
- Mean Validation Accuracy: 0.8387
- Mean Validation F1-Score: 0.7347

---

# Pipeline_1
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
- Accuracy: 0.9816
- F1-Score: 0.9715
- Validation
- Accuracy: 0.8471
- F1-Score: 0.7448

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
- F1-Score: 0.9967
- Validation
- Accuracy: 0.8026
- F1-Score: 0.6072

# Pipeline_3
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
- Accuracy: 0.8275
- F1-Score: 0.7071

# Pipeline_4
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
- Accuracy: 0.9978
- F1-Score: 0.9967
- Validation
- Accuracy: 0.8167
- F1-Score: 0.6595

# Pipeline_5
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
- Accuracy: 0.8351
- F1-Score: 0.7194

# Pipeline_6
## Pipeline Composition
- Scaler: MinMaxScaler
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
- Accuracy: 0.8481
- F1-Score: 0.7548

# Pipeline_7
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
- Accuracy: 0.9816
- F1-Score: 0.9715
- Validation
- Accuracy: 0.8471
- F1-Score: 0.7448

# Pipeline_8
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
- Accuracy: 0.9978
- F1-Score: 0.9967
- Validation
- Accuracy: 0.8330
- F1-Score: 0.7314

# Pipeline_9
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
- Accuracy: 0.8460
- F1-Score: 0.7457

# Pipeline_10
## Pipeline Composition
- Scaler: NoScaler
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
- Accuracy: 0.8373
- F1-Score: 0.7311
