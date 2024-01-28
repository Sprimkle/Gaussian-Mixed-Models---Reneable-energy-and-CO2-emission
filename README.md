Analysis of Renewable Energy Impact on CO2 Emissions Using Statistical and Machine Learning Methods

This repository contains a comprehensive Python script that integrates statistical methods and machine learning algorithms to analyze the impact of renewable energy production on CO2 emissions. The script is structured to provide a clear understanding of the data and the underlying relationships using various analytical techniques.

Key Features:
Data Extraction and Preprocessing: The script begins by importing data from the "Statistical Review of World Energy Data.xlsx", focusing on oil production and CO2 emissions. It involves data cleaning and preparation, ensuring the data is ready for analysis.
Exploratory Data Analysis (EDA): Visualizations are created to explore trends in oil production and CO2 emissions over time, providing an initial understanding of the data.
Fitting Probability Distributions: The script applies Maximum Likelihood Estimation (MLE) to estimate parameters for normal distributions fitting CO2 and oil data, laying the groundwork for more complex modeling.
Gaussian Mixture Model (GMM) Implementation: A GMM is fitted to the data to explore the underlying distribution. The model's ability to generate synthetic data points based on learned distributions is demonstrated, filling gaps in the existing dataset.
Renewable Energy vs. CO2 Emissions Analysis: An extensive analysis is conducted to understand the relationship between renewable energy generation and CO2 emissions, including a scatter plot visualization and the calculation of a 5-year moving average rate of change in CO2 emissions.
Machine Learning Models: Linear Regression and Gradient Boosting Regressor models are implemented to predict the impact of renewable energy generation on CO2 emission rates. The script includes model fitting, predictions, and evaluation using log-likelihood as a measure of model fit quality.

Key Outcomes:
How to merge and manipulate datasets for analytical purposes.
The application of statistical models like GMM in understanding data distributions.
The use of machine learning models for predictive analysis and their evaluation using statistical methods.

Note: To run this script, ensure you have the necessary data files and Python libraries (pandas, numpy, scipy, matplotlib, sklearn) installed.
