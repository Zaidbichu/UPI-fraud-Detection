# UPI Fraud Detection

An end-to-end machine learning project for detecting suspicious UPI transactions using a modular training pipeline and a Streamlit deployment app.
Source: Kaggle https://share.google/Yli5gYTPvcQhkdVPt
## Project Overview

This project predicts whether a UPI transaction is normal or potentially fraudulent. It includes:

- data ingestion
- data validation
- data transformation
- model training
- live prediction in Streamlit
- dashboard and analytics views for exploration

The goal is to turn a fraud dataset into a practical fraud-monitoring application with an interactive interface.

## Features

- Modular ML pipeline with separate components for each stage
- Schema-based validation using YAML
- Feature engineering for time and balance differences
- Saved preprocessing object and trained model artifact
- Streamlit app with:
  - Dashboard
  - Live Scoring
  - Analytics

## Dataset

The project uses a UPI fraud dataset with columns such as:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

### Important observations

- The dataset is highly imbalanced
- Fraud is concentrated mostly in `TRANSFER` and `CASH_OUT`
- Balance-related and time-based features are useful

## Project Workflow

1. **Data Ingestion**
   - Load the raw dataset
   - Save raw copies into the artifact folder
   - Split into train and test sets

2. **Data Validation**
   - Check schema
   - Check for missing columns
   - Detect drift between train and test
   - Save valid/invalid files

3. **Data Transformation**
   - Drop unused columns
   - Add engineered features
   - Encode categorical variables
   - Scale numeric variables
   - Save transformed arrays and preprocessing object

4. **Model Training**
   - Train classifiers on transformed data
   - Evaluate using F1, precision, and recall
   - Save final model artifact

5. **Deployment**
   - Load the saved model in Streamlit
   - Predict one transaction at a time
   - Display dashboard and analytics pages

## Streamlit App

The deployment app contains three pages:

- **Dashboard**: summary cards, fraud trend by hour, risk by type, recent fraud rows
- **Live Scoring**: enter one transaction and get a prediction
- **Analytics**: suspicious transactions, feature contribution, and notes

## How Live Scoring Works

The app accepts a single transaction, builds a dataframe with the same feature names used during training, and passes it to the saved model wrapper.

The app then shows:

- fraud score or class
- decision such as `ALLOW`, `REVIEW`, or `BLOCK`
- rule-based explanation notes

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- Streamlit
- PyYAML
- pickle

## Folder Structure

```text
upi fraud analysis/
├── FRAUD_DETECTION/
│   ├── COMPONENTS/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_tranformation.py
│   │   └── model_training.py
│   ├── constants/
│   ├── entity/
│   ├── exception/
│   ├── logging/
│   ├── pipeline/
│   └── utils/
├── artifact/
├── data_schema/
├── notebook/
├── upi_dataset/
├── app.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the training pipeline

```powershell
python -m FRAUD_DETECTION.pipeline.training_pipeline
```

### 4. Run the Streamlit app

```powershell
streamlit run app.py
```

## Model Notes

This project was built for fraud detection, so the key metrics are:

- **Precision**: how many predicted fraud cases were actually fraud
- **Recall**: how many real fraud cases were caught
- **F1 Score**: balance between precision and recall

In fraud problems, accuracy alone is not enough because the dataset is highly imbalanced.

## Current Status

- EDA completed
- Data ingestion completed
- Data validation completed
- Data transformation completed
- Model training completed
- Streamlit deployment completed

## Future Improvements

- Add better threshold tuning for `BLOCK` / `REVIEW` / `ALLOW`
- Add real feature importance or SHAP explanations
- Improve the dashboard styling further
- Store the final model path in a cleaner config
- Add unit tests for pipeline components

## Author

Built as a UPI fraud detection project for learning, deployment practice, and portfolio use.
