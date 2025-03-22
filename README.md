# 📊 Health Insurance Data Analysis & Insurance Premium Prediction

## 🔍 Project Overview
This project analyzes **health insurance claims data** to understand the key factors affecting insurance charges.
This dataset contains information on 1338 insurance policyholders, including demographic data (age, sex), health metrics (BMI, number of children, smoking status), location (region), and associated insurance charges.
It also builds **predictive models** to estimate policyholder charges using **machine learning algorithms**. 

## 📂 Dataset Description
- The dataset includes information about policyholders such as:
  - **Age**
  - **Sex**
  - **BMI (Body Mass Index)**
  - **Number of Dependents (Children)**
  - **Smoker Status**
  - **Region**
  - **Insurance Charges**
- The target variable is **`charges`**, representing the total insurance cost for each individual.

## 📊 Data Analysis & Key Findings
- **Smokers pay significantly higher insurance fees** compared to non-smokers.
- **Age has a slight positive correlation** with insurance charges.
- **BMI alone does not strongly affect charges**, but high BMI combined with smoking leads to **higher costs**.

## 🤖 Machine Learning Models Used
1. **Linear Regression** – Baseline model
2. **XGBoost Regressor** – Improves accuracy using gradient boosting

## 🚀 How to Run the Project
### 1️⃣ Install Several Libraries 
```bash
1. pandas
2. numpy
3. matplotlib
4. seaborn
5. scikit-learn (known as sklearn)
6. xgboost
```

### 2️⃣ Run the Analysis & Model Training
```bash
python insurance_analysis.py
```

## 🎯 Results & Model Performance
| Model | R² Score | Mean Absolute Error |
|--------|---------|----------------------|
| Linear Regression | ~0.75 | High |
| XGBoost | **Best (~0.88)** | **Lowest** |


## 📎 Repository Contents
- **`insurance_analysis.py`** → Data analysis, visualization, model training, model prediction
- **`README.md`** → Documentation (this file)
- **`insurance.csv`** → Raw data

## 🏆 Next Steps
- Add more **features** (e.g., income, medical history)
- Create a bigger dataset using the best model possible

---
🌟 **If you found this project helpful, give it a ⭐ on GitHub!** 🚀
