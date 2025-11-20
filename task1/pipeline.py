#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Pipeline Example
# 
# This notebook demonstrates how to build a robust machine learning pipeline using `scikit-learn`. The key benefits of this approach are:
# 
# 1.  **Preventing Data Leakage**: By splitting the data first and using a pipeline, we ensure that information from the validation/test set doesn't leak into the training process (e.g., when imputing or scaling).
# 2.  **Consistency**: The same preprocessing steps are guaranteed to be applied to the training, validation, and test data.
# 3.  **Simplicity & Readability**: It bundles multiple steps into a single object, making the workflow cleaner and easier to manage.

# ## 1. Imports
# 
# First, we import all the necessary libraries. We'll need `pandas` for data manipulation and various modules from `scikit-learn` and `xgboost` to build our pipeline and model.

# In[34]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE


# ## 2. Load Data
# 
# Next, we load the raw training data from `train.csv`. It's important to start with the original, unprocessed data.

# In[35]:


try:
    # This is the original data, before any preprocessing
    df = pd.read_csv('task1/train.csv')
    print('Train data loaded successfully.')
except FileNotFoundError:
    print("Make sure 'task1/train.csv' is in the same directory as this notebook.")
    exit()

try:
    test_df = pd.read_csv('task1/test.csv')
    print('Test data loaded successfully.')
except FileNotFoundError:
    print("Make sure 'task1/test.csv' is in the same directory as this notebook.")
    exit()


# ## 2.1 Feature Engineering

# In[36]:


def feature_engineer(data):
    epsilon = 1e-6
    data['reports_per_day'] = data['reports_received'] / (data['account_age_days'] + epsilon)
    data['kdr_x_hs'] = data['kill_death_ratio'] * data['headshot_percentage']
    data['kdr_x_accuracy'] = data['kill_death_ratio'] * data['accuracy_score']
    data['cheating_skill_metric'] = data['accuracy_score'] * data['headshot_percentage'] * data['spray_control_score']
    return data
df = feature_engineer(df)
test_df = feature_engineer(test_df)
df = df.dropna(subset=['is_cheater'])


# ## 3. Define Features and Target
# 
# We separate our dataset into features (the input variables, `X`) and the target (the variable we want to predict, `y`). We exclude `player_id` as it is an identifier and not a predictive feature.

# In[37]:


# Assuming all columns except 'is_cheater' and 'player_id' are features.
features = [col for col in df.columns if col not in ['is_cheater', 'player_id', 'id']]
X = df[features]
y = df['is_cheater']

print(f'{len(features)} features selected.')


# ## 4. Split Data into Training and Validation Sets
# 
# This is a critical step. We split the data **before** applying any preprocessing. This prevents data leakage, ensuring that our model's performance on the validation set is a true reflection of its ability to generalize to new, unseen data. We use `stratify=y` to maintain the same proportion of cheaters and non-cheaters in both the training and validation sets.

# In[38]:


# Define the cross-validation strategy
n_splits = 5 # You can change the number of splits
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracies = []
classification_reports = []
best_thresholds = []


# ## 5. Define the Preprocessing Pipeline
# 
# Here, we define the steps to clean and prepare the data. We create a `Pipeline` for our numeric features that first imputes missing values (filling them with the mean of the column) and then scales the data (so all features have a similar range).
# 
# We use a `ColumnTransformer` to apply this pipeline to all our numeric feature columns.

# In[39]:


for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{n_splits} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Re-initialize preprocessor and classifier for each fold to avoid data leakage
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features)
        ],
        remainder='passthrough'
    )

    # Best hyperparameters
    best_params = {
        'colsample_bytree': 0.6399899663272012,
        'gamma': 0.22962444598293358,
        'learning_rate': 0.07674172222780437,
        'max_depth': 5,
        'min_child_weight': 6,
        'n_estimators': 408,
        'subsample': 0.9879639408647978
    }

    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params)
    
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)], voting='soft')
    
    print("Training the model pipeline...")
    preprocessor_step = Pipeline(steps=[('preprocessor', preprocessor)])
    X_train_processed = preprocessor_step.fit_transform(X_train, y_train)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

    eclf1.fit(X_train_smote, y_train_smote)
    print("Training complete.")

    print("Evaluating the model on the validation set...")
    X_val_processed = preprocessor_step.transform(X_val)
    y_pred_proba = eclf1.predict_proba(X_val_processed)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = [f1_score(y_val, y_pred_proba >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    best_thresholds.append(best_threshold)
    
    print(f"Best threshold for fold {fold+1}: {best_threshold}")

    y_pred = (y_pred_proba >= best_threshold).astype(int)

    fold_accuracy = accuracy_score(y_val, y_pred)
    fold_report = classification_report(y_val, y_pred, output_dict=True)

    accuracies.append(fold_accuracy)
    classification_reports.append(fold_report)

    print(f"Fold {fold+1} Accuracy: {fold_accuracy}")
    print(f"Fold {fold+1} Classification Report:\n", classification_report(y_val, y_pred))

print("\n--- Cross-Validation Results ---")
print(f"Average Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")

# Calculate average classification report
avg_report = {
    '0.0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
    '1.0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
    'accuracy': 0,
    'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
    'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
}

for report in classification_reports:
    for key, value in report.items():
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                avg_report[key][metric] += metric_value
        else:
            avg_report['accuracy'] += value

num_reports = len(classification_reports)
for key, value in avg_report.items():
    if isinstance(value, dict):
        for metric, metric_value in value.items():
            avg_report[key][metric] /= num_reports
    else:
        avg_report['accuracy'] /= num_reports

print("Average Classification Report:")
for label, metrics in avg_report.items():
    if isinstance(metrics, dict):
        print(f"{label}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    else:
        print(f"accuracy: {metrics:.4f}")

avg_best_threshold = np.mean(best_thresholds)
print(f"\nAverage Best Threshold: {avg_best_threshold}")

# For final prediction on test set, train on the entire dataset
print("\n--- Training final model on full dataset for test predictions ---")
final_preprocessor_step = Pipeline(steps=[('preprocessor', preprocessor)])
X_processed_full = final_preprocessor_step.fit_transform(X, y)

smote_final = SMOTE(random_state=42)
X_smote_full, y_smote_full = smote_final.fit_resample(X_processed_full, y)

final_classifier_step = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)], voting='soft')
final_classifier_step.fit(X_smote_full, y_smote_full)
print("Final model training complete.")



# ## 9. Make Predictions on the Test Set
# 
# Finally, we use the trained pipeline to make predictions on the official test data and generate a submission file. The process is exactly the same as for the validation set, demonstrating the power and simplicity of the pipeline.

# In[44]:


print("Making predictions on the test set...")
try:
    # test_df is already loaded at the beginning of the script
    # test_df = pd.read_csv('task1/test.csv') # No need to load again
    test_df = feature_engineer(test_df)
    test_features = test_df[features] # Make sure test set has the same feature columns

    # The pipeline automatically applies all the same preprocessing steps
    test_features_processed = final_preprocessor_step.transform(test_features)
    test_predictions_proba = final_classifier_step.predict_proba(test_features_processed)[:, 1]
    test_predictions = (test_predictions_proba >= avg_best_threshold).astype(int)

    # Create submission file
    submission_df = pd.DataFrame({'id': test_df['id'], 'is_cheater': test_predictions})
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully.")

except FileNotFoundError:
    print("Could not find 'test.csv'. Skipping prediction part.")

