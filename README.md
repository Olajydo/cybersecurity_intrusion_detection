# Cybersecurity intrusion detection

## Project Overview
This project focuses on detecting cybersecurity intrusions using a machine learning model trained on network activity data. The dataset includes key features related to user behavior, login patterns, and IP reputation scores. The goal is to accurately classify attack attempts as possible as we can.

---

## Project Structure
```
cybersecurity_intrusion_detection/
│-- data/                         # Dataset storage
│   ├── cybersecurity_intrusion_data.csv   # Raw dataset
│-- models/                         # Trained model storage
│   ├── best_xgb_model.pkl           # Saved XGBoost model
│-- notebook/                        # Jupyter notebooks for analysis
│   ├── main.ipynb                   # Exploratory data analysis and visualization
│-- scripts/                          # Python scripts for different stages
│   ├── data_preprocessing.py         # Data cleaning and feature engineering
│   ├── modelling.py                  # Model training script
│   ├── inference.py                  # Script for making predictions
│-- .gitignore                        # Files to be ignored in version control
│-- README.md                         # Project documentation
```

---

## How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed. Install required libraries using:
```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing
Run the preprocessing script to clean and transform the dataset:
```bash
python scripts/data_preprocessing.py
```

### 3. Model Training
Train the model using the following command:
```bash
python scripts/modelling.py
```

### 4. Model Inference
To make predictions using the trained model, run:
```bash
python scripts/inference.py
```

---

## Model Performance Summary
### XGBoost Model Performance

- **Threshold** Setting a **threshold of 0.3** offers the best trade-off between detecting attacks and minimizing false alerts.
- **Precision:** 91% (Only 9% of flagged attack cases are actually normal transactions, reducing false positives).
- **Recall:** 79%  Most attack cases are still detected, ensuring good security coverage.
- **Accuracy:** 87% he model has strong overall performance in classifying sessions correctly.
  
### Key Insights from Exploratory Data Analysis:
- Login Attempts and Attack Detection: Higher login attempts correlate with an increased likelihood of an attack being detected.
- Failed Logins vs. Attack Detection: Users with multiple failed logins are more likely to be flagged as attacks.
- Unusual Time Access and Intrusions: Sessions occurring at unusual times have a significantly higher attack detection rate.
- These insights indicate that login behavior, failed login attempts, and access timing are strong indicators of potential cybersecurity threats.

---

## Author
- **Olajide** - Data Scientist & Machine Learning Engineer

For questions, contact yusufolajideda1@gmail.com.

