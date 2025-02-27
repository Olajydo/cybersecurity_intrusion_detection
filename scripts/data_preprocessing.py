import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath): #loading data from the directory
    df = pd.read_csv(filepath)

    # Feature Engineering
    df["session_duration_log"] = np.log1p(df["session_duration"])
    df["login_attempts_log"] = np.log1p(df["login_attempts"])

    # Dropping redundant columns
    X = df.drop(columns=["attack_detected", "session_id", "session_duration", "login_attempts"])
    y = df["attack_detected"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_data(): # function to process the data
    categorical = ["protocol_type", "encryption_used", "browser_type"]
    numerical = ["network_packet_size", "ip_reputation_score", "failed_logins", "unusual_time_access",
                 "session_duration_log", "login_attempts_log"]
    
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numerical),
        ("cat", cat_transformer, categorical)
    ])

    return preprocessor
