# Model Selection Report - dataset_12
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_10
- NoScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8604
- Validation F1-Score: 0.7797

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_10
- NoScaler - NoReducer - LightGBM
- Validation Accuracy: 0.8604
- Validation F1-Score: 0.7797

# XGBoost Performance
- Pipelines: 1, 3, 6, 9
- Mean Validation Accuracy: 0.8369
- Mean Validation F1-Score: 0.7301

# LightGBM Performance
- Pipelines: 2, 4, 5, 10
- Mean Validation Accuracy: 0.8396
- Mean Validation F1-Score: 0.7381

# RandomForest Performance
- Pipelines: 7, 8
- Mean Validation Accuracy: 0.7927
- Mean Validation F1-Score: 0.5910

---

# Pipeline_1
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
- Accuracy: 0.8009
- F1-Score: 0.6416

# Pipeline_2
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
- Accuracy: 0.8409
- F1-Score: 0.7404

# Pipeline_3
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
- Accuracy: 0.8539
- F1-Score: 0.7692

# Pipeline_4
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
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8052
- F1-Score: 0.6685

# Pipeline_5
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
- Accuracy: 0.8517
- F1-Score: 0.7636

# Pipeline_6
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
- Accuracy: 0.8539
- F1-Score: 0.7692

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
- Accuracy: 0.9957
- F1-Score: 0.9935
- Validation
- Accuracy: 0.7900
- F1-Score: 0.5835

# Pipeline_8
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
- Accuracy: 0.9935
- F1-Score: 0.9902
- Validation
- Accuracy: 0.7955
- F1-Score: 0.5986

# Pipeline_9
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
- Accuracy: 0.9978
- F1-Score: 0.9968
- Validation
- Accuracy: 0.8387
- F1-Score: 0.7404

# Pipeline_10
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
- Accuracy: 0.8604
- F1-Score: 0.7797
