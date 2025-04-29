# Model Selection Report - dataset_23
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_9
- MinMaxScaler - PCA - XGBoost
- Validation Accuracy: 0.8254
- Validation F1-Score: 0.7187

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_9
- MinMaxScaler - PCA - XGBoost
- Validation Accuracy: 0.8254
- Validation F1-Score: 0.7187

# XGBoost Performance
- Pipelines: 1, 5, 9, 10
- Mean Validation Accuracy: 0.8124
- Mean Validation F1-Score: 0.6826

# RandomForest Performance
- Pipelines: 2, 4
- Mean Validation Accuracy: 0.7587
- Mean Validation F1-Score: 0.4521

# LightGBM Performance
- Pipelines: 3, 6, 7, 8
- Mean Validation Accuracy: 0.8059
- Mean Validation F1-Score: 0.6734

---

# Pipeline_1
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
- Accuracy: 0.9360
- F1-Score: 0.8959
- Validation
- Accuracy: 0.8102
- F1-Score: 0.6637

# Pipeline_2
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
- Accuracy: 0.7744
- F1-Score: 0.5158

# Pipeline_3
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
- Accuracy: 0.8015
- F1-Score: 0.6591

# Pipeline_4
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 50
- min_samples_split : 2
- min_samples_leaf : 3
- max_depth : 10
- n_components : 110
## Best Performance
- Train
- Accuracy: 0.9848
- F1-Score: 0.9766
- Validation
- Accuracy: 0.7430
- F1-Score: 0.3883

# Pipeline_5
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
- Accuracy: 0.8070
- F1-Score: 0.6740

# Pipeline_6
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
- Accuracy: 0.8026
- F1-Score: 0.6590

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
- Accuracy: 0.9978
- F1-Score: 0.9967
- Validation
- Accuracy: 0.8124
- F1-Score: 0.7062

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
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8070
- F1-Score: 0.6693

# Pipeline_9
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
- Accuracy: 0.8254
- F1-Score: 0.7187

# Pipeline_10
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
- Accuracy: 0.8070
- F1-Score: 0.6740
