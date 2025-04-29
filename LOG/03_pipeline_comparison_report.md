# Model Selection Report - dataset_03
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_10
- NoScaler - PCA - LightGBM
- Validation Accuracy: 0.9284
- Validation F1-Score: 0.8969

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_10
- NoScaler - PCA - LightGBM
- Validation Accuracy: 0.9284
- Validation F1-Score: 0.8969

# LightGBM Performance
- Pipelines: 1, 4, 8, 10
- Mean Validation Accuracy: 0.8994
- Mean Validation F1-Score: 0.8453

# XGBoost Performance
- Pipelines: 2, 3, 6, 7
- Mean Validation Accuracy: 0.8943
- Mean Validation F1-Score: 0.8341

# RandomForest Performance
- Pipelines: 5, 9
- Mean Validation Accuracy: 0.9084
- Mean Validation F1-Score: 0.8592

---

# Pipeline_1
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
- Accuracy: 0.9165
- F1-Score: 0.8813

# Pipeline_2
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 1.0
- n_estimators : 100
- max_depth : 7
- learning_rate : 0.1
- colsample_bytree : 1.0
- n_components : 10
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9165
- F1-Score: 0.8803

# Pipeline_3
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 1.0
- n_estimators : 200
- max_depth : 3
- learning_rate : 0.3
- colsample_bytree : 1.0
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9165
- F1-Score: 0.8767

# Pipeline_4
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
- Accuracy: 0.9165
- F1-Score: 0.8790

# Pipeline_5
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
- F1-Score: 0.9934
- Validation
- Accuracy: 0.9035
- F1-Score: 0.8537

# Pipeline_6
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
- Accuracy: 0.8275
- F1-Score: 0.7026

# Pipeline_7
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- subsample : 1.0
- n_estimators : 200
- max_depth : 3
- learning_rate : 0.3
- colsample_bytree : 1.0
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9165
- F1-Score: 0.8767

# Pipeline_8
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
- Accuracy: 0.8362
- F1-Score: 0.7238

# Pipeline_9
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
- Accuracy: 0.9132
- F1-Score: 0.8648

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
- Accuracy: 0.9978
- F1-Score: 0.9967
- Validation
- Accuracy: 0.9284
- F1-Score: 0.8969
