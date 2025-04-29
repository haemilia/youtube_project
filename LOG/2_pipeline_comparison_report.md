# Model Selection Report - dataset_2
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_6
- MinMaxScaler - PCA - XGBoost
- Validation Accuracy: 0.7413
- Validation F1-Score: 0.5533

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_6
- MinMaxScaler - PCA - XGBoost
- Validation Accuracy: 0.7413
- Validation F1-Score: 0.5533

# RandomForest Performance
- Pipelines: 1, 4
- Mean Validation Accuracy: 0.7148
- Mean Validation F1-Score: 0.3475

# LightGBM Performance
- Pipelines: 2, 3, 7, 8
- Mean Validation Accuracy: 0.7370
- Mean Validation F1-Score: 0.5371

# XGBoost Performance
- Pipelines: 5, 6, 9, 10
- Mean Validation Accuracy: 0.7362
- Mean Validation F1-Score: 0.5275

---

# Pipeline_1
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- min_samples_split : 5
- min_samples_leaf : 1
- max_depth : None
## Best Performance
- Train
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7229
- F1-Score: 0.4071

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
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7370
- F1-Score: 0.5204

# Pipeline_3
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 310
- num_leaves : 100
- n_estimators : 100
- max_depth : -1
- learning_rate : 0.3
## Best Performance
- Train
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7392
- F1-Score: 0.5435

# Pipeline_4
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
- Accuracy: 0.9957
- F1-Score: 0.9935
- Validation
- Accuracy: 0.7067
- F1-Score: 0.2878

# Pipeline_5
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 1.0
- n_estimators : 50
- max_depth : 3
- learning_rate : 0.1
- colsample_bytree : 0.8
- n_components : 160
## Best Performance
- Train
- Accuracy: 0.9167
- F1-Score: 0.8613
- Validation
- Accuracy: 0.7316
- F1-Score: 0.5030

# Pipeline_6
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 100
- max_depth : 7
- learning_rate : 0.3
- colsample_bytree : 1.0
- n_components : 360
## Best Performance
- Train
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7413
- F1-Score: 0.5533

# Pipeline_7
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
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7381
- F1-Score: 0.5466

# Pipeline_8
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
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7338
- F1-Score: 0.5377

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
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7359
- F1-Score: 0.5270

# Pipeline_10
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
- Accuracy: 0.9989
- F1-Score: 0.9984
- Validation
- Accuracy: 0.7359
- F1-Score: 0.5270
