# Model Selection Report - dataset_123# Best Performance by Validation Accuracy
- Pipeline Name: Pipeline_8
- MinMaxScaler - PCA - XGBoost
- Validation Accuracy: 0.9759
- Validation F1-Score: 0.0128

# Best Performance by Validation F1-Score
- Pipeline Name: Pipeline_6
- MinMaxScaler - NoReducer - XGBoost
- Validation Accuracy: 0.9759
- Validation F1-Score: 0.0494

# RandomForest Performance
- Pipelines: 1, 3, 5, 7
- Mean Validation Accuracy: 0.9758
- Mean Validation F1-Score: 0.0107

# XGBoost Performance
- Pipelines: 2, 4, 6, 8
- Mean Validation Accuracy: 0.9758
- Mean Validation F1-Score: 0.0156

---

# Pipeline_1
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 50
- max_depth : None
## Best Performance
- Train
- Accuracy: 0.9998
- F1-Score: $0.9967$
- Validation
- Accuracy: 0.9758
- F1-Score: $0.0000$

# Pipeline_2
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- n_estimators : 200
- max_depth : 7
- learning_rate : 0.01
## Best Performance
- Train
- Accuracy: 0.9907
- F1-Score: $0.7611$
- Validation
- Accuracy: 0.9758
- F1-Score: $0.0000$

# Pipeline_3
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- max_depth : 20
- n_components : 8
## Best Performance
- Train
- Accuracy: 0.9998
- F1-Score: $0.9951$
- Validation
- Accuracy: 0.9757
- F1-Score: $0.0429$

# Pipeline_4
## Pipeline Composition
- Scaler: NoScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- n_estimators : 50
- max_depth : 7
- learning_rate : 0.3
- n_components : 1
## Best Performance
- Train
- Accuracy: 0.9758
- F1-Score: $0.0000$
- Validation
- Accuracy: 0.9758
- F1-Score: $0.0000$

# Pipeline_5
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- max_depth : None
## Best Performance
- Train
- Accuracy: 0.9999
- F1-Score: $0.9984$
- Validation
- Accuracy: 0.9758
- F1-Score: $0.0000$

# Pipeline_6
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: NoReducer
- Classifier: XGBoost
## Hyperparameters
- n_estimators : 50
- max_depth : 5
- learning_rate : 0.1
## Best Performance
- Train
- Accuracy: 0.9964
- F1-Score: $0.9206$
- Validation
- Accuracy: 0.9759
- F1-Score: $0.0494$

# Pipeline_7
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: RandomForest
## Hyperparameters
- n_estimators : 100
- max_depth : 10
- n_components : 3
## Best Performance
- Train
- Accuracy: 0.9763
- F1-Score: $0.0385$
- Validation
- Accuracy: 0.9758
- F1-Score: $0.0000$

# Pipeline_8
## Pipeline Composition
- Scaler: MinMaxScaler
- Dimension Reduction: PCA
- Classifier: XGBoost
## Hyperparameters
- n_estimators : 100
- max_depth : 5
- learning_rate : 0.1
- n_components : 7
## Best Performance
- Train
- Accuracy: 0.9760
- F1-Score: $0.0130$
- Validation
- Accuracy: 0.9759
- F1-Score: $0.0128$
