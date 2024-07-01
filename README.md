# Bank-Card-Fraud-Detection

Welcome to the Bank Card Fraud Detection project. This application aims to detect fraudulent transactions on bank cards using machine learning models. The project is implemented in Python and deployed via Streamlit, offering an interactive interface to analyze and predict fraudulent activities in financial transactions.

# Project Overview
The Bank Card Fraud Detection project is focused on identifying fraudulent transactions using machine learning. Given the increasing volume and sophistication of financial fraud, this project seeks to provide a reliable solution to detect and prevent fraudulent activities. By analyzing transaction data, the model predicts the likelihood of a transaction being fraudulent, helping to safeguard users' financial information.

You can view the deployed application [here](https://bank-card-fraud-detection.streamlit.app/).

## Features

- **Fraud Prediction**: Classifies transactions as fraudulent or legitimate.
- **Interactive Web Interface**: Allows users to input transaction data and get predictions instantly.
- **Model Interpretability**: Provides insights into model decisions using SHAP values.
- **Data Imbalance Handling**: Techniques to manage the imbalance between fraudulent and legitimate transactions.
- **Detailed Metrics**: Evaluates model performance using precision, recall, F1-score, and ROC-AUC.

## Dataset

### Source

The dataset used in this project is from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

### Features

- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: Principal components obtained with PCA to protect sensitive information.
- **Amount**: Transaction amount.
- **Class**: Target variable (1 for fraudulent, 0 for legitimate).

### Preprocessing

- **Standardization**: Scaling the 'Amount' and 'Time' features.
- **PCA Components**: The V1-V28 features are already transformed PCA components.
- **Imbalance Handling**: Using SMOTE to oversample the minority class (fraudulent transactions).

## Machine Learning Pipeline

### Models Used

- **Logistic Regression**: Baseline model for binary classification.
- **Random Forest**: Ensemble learning method effective for imbalanced data.
- **XGBoost**: Gradient boosting technique optimized for classification tasks.
- **Neural Network**: Deep learning model for capturing complex patterns.

### Evaluation Metrics

- **Precision**: Accuracy of the fraud predictions.
- **Recall**: Ability to detect actual fraudulent transactions.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Measures the model's performance across all thresholds.

# Conclusion

The Bank Card Fraud Detection project successfully demonstrates the application of machine learning to detect fraudulent transactions. With a robust preprocessing pipeline, effective feature engineering, and a high-performing Gradient Boosting model, the system provides accurate and reliable fraud detection. Future work includes integrating additional data sources, enhancing the model with deep learning techniques, and deploying the system in a production environment for real-time fraud monitoring.

# Future Work

- **Incorporate Additional Data**: Integrate more diverse data sources to improve model robustness.
- **Enhance Model**: Explore deep learning models for potentially better performance.
- **Production Deployment**: Scale the system for real-time transaction monitoring in a production environment.
- **Continuous Learning**: Implement mechanisms for the model to learn from new fraud patterns dynamically.

# Bank Fraud Detection Web App  

[streamlit-app-2024-07-01-12-07-79.webm](https://github.com/kunalhchandnani/Bank-Card-Fraud-Detection/assets/88874426/f629ece5-0cc3-4a35-8ce2-c367732c2d8c)
