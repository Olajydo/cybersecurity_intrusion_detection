import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from data_preprocessing import load_data, preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_data("cybersecurity_intrusion_data.csv")
preprocessor = preprocess_data(X_train, X_test)

# Define the XGBoost model,pipeline and also defining our weight value
class_0, class_1 = y_train.value_counts()
scale_pos_weight_value = class_0 / class_1
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_value, eval_metric="aucpr")

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# Parameters for our model
xgb_param_grid = {
    'model__n_estimators': list(range(100, 200, 50)),
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': list(range(3, 6, 1)),
    'model__subsample': [0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.8, 0.9, 1.0],
    'model__gamma': [0, 0.1, 0.2],
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline, param_distributions=xgb_param_grid, n_iter=10, 
    scoring="accuracy", cv=5, random_state=42, n_jobs=-1
)

# Fit the model
xgb_search.fit(X_train, y_train)

# Extract the best model and evaluate
best_xgb_model = xgb_search.best_estimator_

# Getting prediction probabilities
y_probs = best_xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# Setting a lower threshold to balance between Attack detection and minimizing false positives
threshold = 0.3
y_preds_adjusted = (y_probs >= threshold).astype(int)

# Evaluate the new threshold
print("New Classification Report:\n", classification_report(y_test, y_preds_adjusted))

# Extract the trained XGBoost model from the pipeline
xgb_model = best_xgb_model.named_steps["model"]

# Get feature importance values
feature_importance = xgb_model.feature_importances_

# Retrieve feature names after preprocessing
categorical_features = ["protocol_type", "encryption_used", "browser_type"]
numerical_features = [
    "network_packet_size", "ip_reputation_score", "failed_logins",
    "unusual_time_access", "session_duration_log", "login_attempts_log"
]

one_hot_encoder = best_xgb_model.named_steps["preprocessor"].named_transformers_["cat"]
one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_features)

final_feature_names = np.concatenate([numerical_features, one_hot_feature_names])

# Create a DataFrame for feature importance
feat_imp_df = pd.DataFrame({"Feature": final_feature_names, "Importance": feature_importance})

# Get the most important features
most_important_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(5)
print("Most Important Features:", most_important_features)

# Save model
joblib.dump(best_xgb_model, "../models/best_xgb_model.pkl")
