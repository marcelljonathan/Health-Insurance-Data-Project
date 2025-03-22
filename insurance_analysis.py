### Import several libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### Data Inspection

# Load the dataset
df = pd.read_csv("insurance.csv")

# Display basic information
print(df.info())

### Data summary

# Display summary statistics for numerical columns
print(df.describe())

### Plot the distribution of insurance charges


### Analysis using plots

# Set a style for the plots
sns.set_style("whitegrid")

# Create a histogram to vizualize the distribution of insurance charges

plt.figure(figsize = (8,5))
sns.histplot(df["charges"], bins = 30, kde = True, color = "blue")
plt.title("Distribution of Insurance Charges") # Add lables and titles
plt.xlabel("Charges ($)")
plt.ylabel("Frequency")
plt.show()

# Check how smoking effects insurance charges

plt.figure(figsize=(8,5))
sns.boxplot(x = "smoker", y ="charges", data = df, palette = {"yes":"red","no":"green"})
plt.title("Insurance Charges by Smoking Status")
plt.xlabel("Smoker")
plt.ylabel("Charges ($)")
plt.show()

# See correlation between age and insurance charges

plt.figure(figsize=(8,5))
sns.scatterplot(x=df["age"],y=df["charges"], alpha = 0.5, color = "blue")
plt.title("Age vs Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Charges ($)")
plt.show()

# See correlation between bmi and insurance charges

plt.figure(figsize=(8,5))
sns.scatterplot(x=df["bmi"],y=df["charges"], alpha = 0.75, color = "blue")
plt.title("Body Mass Index vs Insurance Charges")
plt.xlabel("Body Mass Index")
plt.ylabel("Charges ($)")
plt.show()

# Scatter plot: BMI vs. Charges, colored by Smoking Status

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["bmi"], y=df["charges"], hue=df["smoker"], alpha=0.5)
plt.title("BMI vs. Insurance Charges (Smokers vs. Non-Smokers)")
plt.xlabel("BMI")
plt.ylabel("Charges ($)")
plt.show()


### Building a prediction model with linear regression (only use variable age, bmi, sex and smoker status)

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

variables = ["age","bmi","sex","smoker"]
df_selected = df[variables + ["charges"]]

# Convert categorical variables into numerical format
df_selected = pd.get_dummies(df_selected, columns =["sex","smoker"], drop_first = True)

# Define variables (X) and responses (Y)
x = df_selected.drop(columns = ["charges"])
y = df_selected["charges"]

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

# Initialize the model 
model = LinearRegression()

# Train (fit) the model on training data
model.fit(x_train,y_train)

# Check model accuracy
print(f"Model Accuracy (R Squared score): {model.score(x_test,y_test):.4f} ")

# Make a function for prediction
def predict_charges(age, bmi, sex, smoker):
    sex_value = 1 if sex.lower() == "male" else 0 #Convert sex and smoker input into numerical
    smoker_value = 1 if smoker.lower() == "yes" else 0

    #Create Dataframe for prediction
    input_data = pd.DataFrame([[age,bmi,sex_value,smoker_value]], columns = x.columns)

    #Predict charges
    predicted_charge = model.predict(input_data)[0]
    return round(predicted_charge, 0)

# User Input
age = int(input("Enter Age: "))
bmi = float(input("Enter Body Mass Index: "))
sex = input("Enter Sex (Male/Female): ")
smoker = input("Smoker? (Yes/No): ")

predicted_charge =predict_charges(age,bmi,sex,smoker)
print(f"Predicted charges: ${predicted_charge}")


### Building a prediction model using XGBoost Model (variables uses are age, bmi, sex and smoker)

import sklearn
import xgboost 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

variables = ["age","bmi","sex","smoker"]
df_selected = df[variables + ["charges"]]

# Convert categorical variables into numerical format
df_selected = pd.get_dummies(df_selected, columns =["sex","smoker"], drop_first = True)

# Define variables (X) and responses (Y)
x = df_selected.drop(columns = ["charges"])
y = df_selected["charges"]

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

# Train an XGBoost Model
xgb_model = XGBRegressor(n_estimators = 100, learning_rate = 0.1, random_state = 42)
xgb_model.fit(x_train,y_train)

# Prediction on test set
y_pred = xgb_model.predict(x_test)

# Check model accuracy
mae = mean_absolute_error(y_test,y_pred)
r2_score = xgb_model.score(x_test,y_test)

print(f"The XGBoost Mean Absolute Error: {mae:.2f} and R Squared Score : {r2_score:.4f}")

# Make a function for prediction
def predict_charges(age, bmi, sex, smoker):
    sex_value = 1 if sex.lower() == "male" else 0 #Convert sex and smoker input into numerical
    smoker_value = 1 if smoker.lower() == "yes" else 0

    #Create Dataframe for prediction
    input_data = pd.DataFrame([[age,bmi,sex_value,smoker_value]], columns = x.columns)

    #Predict charges
    predicted_charge = xgb_model.predict(input_data)[0]
    return round(predicted_charge, 0)

# User Input
age = int(input("Enter Age: "))
bmi = float(input("Enter Body Mass Index: "))
sex = input("Enter Sex (Male/Female): ")
smoker = input("Smoker? (Yes/No): ")

predicted_charge =predict_charges(age,bmi,sex,smoker)
print(f"Predicted charges: ${predicted_charge}")
