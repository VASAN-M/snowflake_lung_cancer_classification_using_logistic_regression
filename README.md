# Lung Cancer Prediction using Logistic Regression (Snowflake Data)

This project builds a Logistic Regression model to predict lung cancer
using structured data retrieved from a Snowflake data warehouse.
The workflow includes data extraction, preprocessing, model training,
evaluation, and model persistence.

---

## Data Source
- Platform: Snowflake
- Database: CLASSIFICATION
- Schema: PUBLIC
- Table: CANCER
- Target column: `LUNG_CANCER`
  - True → 1
  - False → 0

⚠️ **Important:**  
Snowflake credentials must NOT be hardcoded in production or public
repositories. Use environment variables or a secure secrets manager.

---

## Workflow

### 1. Data Extraction
- Connected to Snowflake using `snowflake-connector-python`
- Queried the `CANCER` table and loaded data into a pandas DataFrame

### 2. Data Exploration
- Inspected schema and data types
- Checked unique values per column
- Verified absence of missing values

### 3. Data Preprocessing
- Encoded target variable (`LUNG_CANCER`) into binary format
- Converted categorical feature:
  - `GENDER`: M → 1, F → 0
- Standardized numerical features using `StandardScaler`

### 4. Model Training
- Algorithm: Logistic Regression
- Handling class imbalance using `class_weight="balanced"`
- Train-test split:
  - 75% training
  - 25% testing
- Stratified split to preserve class distribution

### 5. Model Evaluation
- Classification Report (precision, recall, F1-score)
- ROC-AUC score for probabilistic performance evaluation

### 6. Model Persistence
- Saved trained Logistic Regression model using `joblib`
- Saved fitted scaler for future inference

---

## Libraries Used
- pandas
- snowflake-connector-python
- scikit-learn
- joblib

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas scikit-learn snowflake-connector-python joblib
