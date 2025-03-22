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
### 1️⃣ Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tk
```

### 2️⃣ Run the Analysis & Model Training
```bash
python insurance_analysis.py
```

### 3️⃣ Launch the Prediction App (GUI)
```bash
python insurance_gui.py
```

## 🎯 Results & Model Performance
| Model | R² Score | Mean Absolute Error |
|--------|---------|----------------------|
| Linear Regression | ~0.75 | High |
| Random Forest | ~0.85 | Lower |
| XGBoost | **Best (~0.88)** | **Lowest** |

## 📌 Features of the GUI Application
✔ User inputs **age, sex, and smoker status**
✔ Model **predicts insurance charges** in real-time
✔ Uses **Tkinter** for a simple and interactive interface

## 📎 Repository Contents
- **`insurance_analysis.py`** → Data analysis, visualization, model training
- **`insurance_gui.py`** → Interactive prediction app
- **`README.md`** → Documentation (this file)

## 🏆 Next Steps
- Add more **features** (e.g., income, medical history)
- Implement **hyperparameter tuning** for better predictions
- Deploy as a **web app** using Flask or Streamlit

## 📬 Contact
📧 **Your Email** | 🔗 [Your LinkedIn](#) | 🖥️ [Your GitHub](#)

---
🌟 **If you found this project helpful, give it a ⭐ on GitHub!** 🚀
