# Model Selection Report - dataset_023
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_5
- NoScaler - PCA - XGBoost
- Validation Accuracy: 0.9198
- Validation F1-Score: 0.8842

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_6
- MinMaxScaler - NoReducer - LightGBM
- Validation Accuracy: 0.9187
- Validation F1-Score: 0.8855

# LightGBM Performance
- Pipelines: 1, 4, 6, 7
- Mean Validation Accuracy: 0.8981
- Mean Validation F1-Score: 0.8451

# RandomForest Performance
- Pipelines: 2, 9
- Mean Validation Accuracy: 0.8981
- Mean Validation F1-Score: 0.8404

# XGBoost Performance
- Pipelines: 3, 5, 8, 10
- Mean Validation Accuracy: 0.9021
- Mean Validation F1-Score: 0.8527

---

# Pipeline_1
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
- Accuracy: 0.9165
- F1-Score: 0.8787

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
- Accuracy: 0.9967
- F1-Score: 0.9951
- Validation
- Accuracy: 0.9013
- F1-Score: 0.8532

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
- Accuracy: 0.8514
- F1-Score: 0.7622

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
- Accuracy: 0.8416
- F1-Score: 0.7394

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
- Accuracy: 0.9198
- F1-Score: 0.8842

# Pipeline_6
## Pipeline Composition
- Scaler: MinMaxScaler
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
- Accuracy: 0.9187
- F1-Score: 0.8855

# Pipeline_7
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 10
- num_leaves : 50
- n_estimators : 100
- max_depth : 5
- learning_rate : 0.3
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.9154
- F1-Score: 0.8771

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
- Accuracy: 0.9924
- F1-Score: 0.9886
- Validation
- Accuracy: 0.9187
- F1-Score: 0.8821

# Pipeline_9
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
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8948
- F1-Score: 0.8275

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
- Accuracy: 0.9924
- F1-Score: 0.9886
- Validation
- Accuracy: 0.9187
- F1-Score: 0.8821
