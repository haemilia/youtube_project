# Model Selection Report - dataset_1
# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_4
- NoScaler - NoReducer - RandomForest
- Validation Accuracy: 0.8669
- Validation F1-Score: 0.7884

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_4
- NoScaler - NoReducer - RandomForest
- Validation Accuracy: 0.8669
- Validation F1-Score: 0.7884

# LightGBM Performance
- Pipelines: 1, 3, 7, 8
- Mean Validation Accuracy: 0.8385
- Mean Validation F1-Score: 0.7443

# RandomForest Performance
- Pipelines: 2, 4
- Mean Validation Accuracy: 0.8663
- Mean Validation F1-Score: 0.7869

# XGBoost Performance
- Pipelines: 5, 6, 9, 10
- Mean Validation Accuracy: 0.8428
- Mean Validation F1-Score: 0.7429

---

# Pipeline_1
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
- Accuracy: 0.9675
- F1-Score: 0.9510
- Validation
- Accuracy: 0.8539
- F1-Score: 0.7699

# Pipeline_2
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- min_samples_split : 2
- min_samples_leaf : 3
- max_depth : None
- n_components : 10
## Best Performance
- Train
- Accuracy: 0.9491
- F1-Score: 0.9215
- Validation
- Accuracy: 0.8658
- F1-Score: 0.7853

# Pipeline_3
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 10
- num_leaves : 50
- n_estimators : 100
- max_depth : -1
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8019
- F1-Score: 0.6865

# Pipeline_4
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
- Accuracy: 0.9708
- F1-Score: 0.9557
- Validation
- Accuracy: 0.8669
- F1-Score: 0.7884

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
- Accuracy: 0.9470
- F1-Score: 0.9185
- Validation
- Accuracy: 0.8561
- F1-Score: 0.7663

# Pipeline_6
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
- Accuracy: 0.9426
- F1-Score: 0.9094
- Validation
- Accuracy: 0.8063
- F1-Score: 0.6806

# Pipeline_7
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: LightGBM
## Hyperparameters
- n_components : 10
- num_leaves : 50
- n_estimators : 100
- max_depth : -1
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 1.0000
- F1-Score: 1.0000
- Validation
- Accuracy: 0.8442
- F1-Score: 0.7512

# Pipeline_8
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
- Accuracy: 0.9675
- F1-Score: 0.9510
- Validation
- Accuracy: 0.8539
- F1-Score: 0.7698

# Pipeline_9
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- subsample : 0.8
- n_estimators : 200
- max_depth : 5
- learning_rate : 0.01
- colsample_bytree : 0.8
- n_components : 10
## Best Performance
- Train
- Accuracy: 0.9069
- F1-Score: 0.8512
- Validation
- Accuracy: 0.8528
- F1-Score: 0.7583

# Pipeline_10
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
- Accuracy: 0.9470
- F1-Score: 0.9185
- Validation
- Accuracy: 0.8561
- F1-Score: 0.7663
