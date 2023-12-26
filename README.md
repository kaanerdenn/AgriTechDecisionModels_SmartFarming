## AgriTech Decision Models & SmartFarming

# Overview

This project focuses on analyzing various factors that influence crop cultivation and employs machine learning to predict suitable crops based on environmental conditions. The goal is to assist farmers and agricultural stakeholders in making informed decisions using data-driven insights.

# Dataset

The dataset, crop.csv, contains various features such as soil composition (Nitrogen, Phosphorus, Potassium), environmental factors (temperature, humidity, pH, rainfall), and the corresponding crop type.

# Features

Data Preprocessing: Cleaning and formatting the dataset for analysis, including handling missing values and converting data types.
Exploratory Data Analysis (EDA): Utilizing Seaborn and Matplotlib to visualize and analyze the dataset.
Statistical Analysis: Computing mean values for various soil and environmental conditions.
Feature Engineering: Implementing OneHotEncoder for categorical variables and MinMaxScaler for numerical features.
Correlation Analysis: Analyzing feature correlations using heatmaps.
Machine Learning: Using Logistic Regression to predict the most suitable crop based on given conditions.
Model Evaluation: Assessing the model's performance with confusion matrices and classification reports.
Cluster Analysis: Employing K-Means clustering to group similar crops based on environmental factors.

# Requirements

This project requires the following Python libraries:

Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
