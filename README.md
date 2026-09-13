# HealthConnect ML Pipeline

## Overview  
Building a predictive pipeline that identifies patients likely to miss their medical appointments.

## Project Structure
- **data/raw/** → Original dataset (`HealthConnect_Appointment_Data.csv`)
- **data/processed/** → Cleaned dataset (`cleaned.csv`)
- **notebooks/**  
  - `EDA.ipynb` → Exploratory Data Analysis (distributions, correlations, imbalance)  
  - `pipeline_dev.ipynb` → Preprocessing, model training, evaluation, and artefact saving  
- **models/** → Trained models (`log_reg.pkl`, `random_forest.pkl`, `xgboost.pkl`)
- **tests/** → Validation scripts for pipeline and models


## Deliverables
- Completed EDA notebook with visualizations and insights  
- Implemented pipeline notebook with preprocessing and ML workflows  
- Saved trained models and processed dataset  
- Documented assumptions, limitations, risks, and dependencies  

## Next Steps to be taken
- Handle class imbalance (SMOTE, class weighting)  
- Feature engineering for improved predictors  
- Hyperparameter tuning and advanced evaluation metrics


## UPDATES  

### Project Structure Changes  
- **data/processed/** → Updated to use `Cleaned_Data.csv` (engineered dataset from Data Science track).  
- **models/** → Now includes `gradient_boosting` alongside `log_reg`, `random_forest`, and `xgboost`.  
- **tests/** → `test_model.py` updated to validate Gradient Boosting integration; pytest passed successfully.  

### Deliverables Added  
- Integrated pipeline notebook (`pipeline_dev.ipynb`) centered on Gradient Boosting candidate model.  
- Validation checks for engineered features and numeric scaling.  
 
  

### Next Steps  
- Implement class imbalance handling (SMOTE, class weighting).  
- Integrate experiment tracking (MLflow or similar) for reproducibility.  
- Conduct deeper model comparison and hyperparameter tuning.  
- Document reproducibility and prepare for deployment readiness.  
