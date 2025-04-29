# Model Selection Report - dataset_3
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_5
- NoScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8145
- Validation F1-Score: 0.6885

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_5
- NoScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8145
- Validation F1-Score: 0.6885

# LightGBM Performance
- Pipelines: 1, 5, 6, 7
- Mean Validation Accuracy: 0.8018
- Mean Validation F1-Score: 0.6577

# RandomForest Performance
- Pipelines: 2, 10
- Mean Validation Accuracy: 0.7652
- Mean Validation F1-Score: 0.4942

# XGBoost Performance
- Pipelines: 3, 4, 8, 9
- Mean Validation Accuracy: 0.8075
- Mean Validation F1-Score: 0.6708

---

# Pipeline_1
## Pipeline Composition
- Scaler: NoScaler
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
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.7863
- F1-Score: 0.6168

# Pipeline_2
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- min_samples_split : 2
- min_samples_leaf : 1
- max_depth : 10
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.7755
- F1-Score: 0.5303

# Pipeline_3
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
- Accuracy: 0.8026
- F1-Score: 0.6552

# Pipeline_4
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
- Accuracy: 0.9967
- F1-Score: 0.9951
- Validation
- Accuracy: 0.8048
- F1-Score: 0.6604

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
- Accuracy: 0.8145
- F1-Score: 0.6885

# Pipeline_6
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
- Accuracy: 0.7994
- F1-Score: 0.6500

# Pipeline_7
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
- Accuracy: 0.8069
- F1-Score: 0.6756

# Pipeline_8
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
- Accuracy: 0.8113
- F1-Score: 0.6839

# Pipeline_9
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
- Accuracy: 0.8113
- F1-Score: 0.6839

# Pipeline_10
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
- Accuracy: 0.7549
- F1-Score: 0.4582
