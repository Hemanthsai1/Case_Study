# Beer Rating Prediction

This project aims to predict the overall rating of beers based on various review and beer characteristics using machine learning models. The task involves data preprocessing, feature engineering, and model training to create a prediction system. The models used in this project are **LightGBM** and **CatBoost**, which are gradient boosting models.

## Project Structure

# Repository Structure
Beer-Rating-Prediction

├── Beer_dataset_README.pdf    # About the Task
|
├── Data_Science_Case_Study.ipynb    # Jupyter Notebook containing analysis, modeling, and evaluation
|
├── README.md    # Project documentation
|
├── requirements.txt    # Required libraries & dependencies
|
└── train.csv    # Dataset

### Technologies Used

- Programming Language: Python
- Machine Learning Models: LightGBM, CatBoost
- Libraries & Tools: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, CatBoost, LightGBM

### **1. data_cleaning_and_preprocessing.py**

This script handles the initial data cleaning and preprocessing steps. It reads the raw dataset, removes irrelevant columns, handles missing values, and ensures that the dataset is in a suitable format for machine learning models. It also handles the removal of non-numeric columns and ensures that all the data types are correct.

#### Key Steps:

- Remove irrelevant columns (e.g., `beer/name`).
- Handle missing values if necessary.
- Convert categorical variables into numerical formats using one-hot encoding or label encoding.
- Ensure data types are numeric for model compatibility.

### **2. feature_engineering.py**

In this script, we create new features that can help improve the model's performance. This includes transforming continuous variables into categorical variables (e.g., creating alcohol content categories from the `beer/ABV` feature) and performing one-hot encoding on categorical variables.

#### Key Steps:

- Create **ABV categories** from the `beer/ABV` feature.
- Perform one-hot encoding on categorical features (e.g., `beer/style`, `user/gender`).
- Prepare the features (`X`) and target variable (`y`) for model training.

### **3. modeling_and_evaluation.py**

This script focuses on building the machine learning models and evaluating their performance. We train **LightGBM** and **CatBoost** models and evaluate them using three common regression metrics: **RMSE (Root Mean Squared Error)**, **MAE (Mean Absolute Error)**, and **R² (Coefficient of Determination)**. The models are trained on the processed data, and their performance is compared.

#### Key Steps:

- Train LightGBM and CatBoost models.
- Evaluate the models using **RMSE**, **MAE**, and **R²** metrics.
- Output the model performance metrics and compare both models.

## Requirements

Before running the scripts, you need to install the necessary libraries. You can install the required dependencies using the `requirements.txt` file (which you may create separately, if you wish). Below are the common libraries used in this project:

- In bash run
  pip install lightgbm catboost scikit-learn pandas numpy matplotlib

## Future Enhancements

- Hyperparameter Tuning for better optimization.
- Ensemble Models to further improve accuracy.
