import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("../models/best_xgb_model.pkl")

def predict_attack(protocol_type, encryption_used, browser_type, network_packet_size, 
                   ip_reputation_score, failed_logins, unusual_time_access, 
                   session_duration, login_attempts):
    
    # Create a DataFrame from the input values
    data = pd.DataFrame({
        "protocol_type": [protocol_type],
        "encryption_used": [encryption_used],
        "browser_type": [browser_type],
        "network_packet_size": [network_packet_size],
        "ip_reputation_score": [ip_reputation_score],
        "failed_logins": [failed_logins],
        "unusual_time_access": [unusual_time_access],
        "session_duration_log": [np.log1p(session_duration)],
        "login_attempts_log": [np.log1p(login_attempts)]
    })
    
    # Make prediction
    prediction = model.predict(data)
    
    # Return prediction result
    return "Attack Detected" if prediction[0] else "No Attack Detected"

# Example usage:
result = predict_attack("HTTPS", "AES", "Chrome", 450, 20, 1, 0, 300, 2)
print(result)