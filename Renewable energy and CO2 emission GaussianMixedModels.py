import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

path = "C:/Users/44731/OneDrive/Desktop/Homework/"

#Extract the CO2 emission and Oil Production page
df_oil = pd.read_excel(path + "Statistical Review of World Energy Data.xlsx", sheet_name="Oil Production - Tonnes", skiprows=2)
df_co2 = pd.read_excel(path + "Statistical Review of World Energy Data.xlsx", sheet_name="CO2e Emissions", skiprows=2)

#Extract only the rows that show the world data
df_oil_total = df_oil.loc[71]

#Extract only the rows that show the world data
df_co2_total = df_co2.loc[97]


#Remove the first row and the last three rows of both dataframes as they are both anomalous
df_oil_total = df_oil_total.iloc[1:-3]
df_co2_total = df_co2_total.iloc[1:-3]


#Merge both dataframes together (Oil and CO2 Emission)
df = pd.merge(df_oil_total, df_co2_total, left_index=True, right_index=True)

# Optionally, you can rename the columns for clarity
df.columns = ['Oil_Total', 'CO2_Total']


#Change the columns to numerical data type
for i in ['Oil_Total','CO2_Total']:
    df[i] = df[i].astype('float')




#Exploratory Data Analysis
# Plotting two lines with markers for each point
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Oil_Total'], label='Oil_Total', color='blue', marker='o')
plt.plot(df.index, df['CO2_Total'], label='CO2_Total', color='green', marker='x')

# Adding titles and labels
plt.title('Oil and CO2 Totals from 1990 to 2022 with Markers')
plt.xlabel('Year')
plt.ylabel('Total')
plt.legend()
plt.grid(True)
plt.show()




# 1. Fit Probability Distributions using MLE (Normal Distribution)
# Estimating parameters for CO2 Total
mu_co2, std_co2 = norm.fit(df['CO2_Total'])
print("Normal Distribution parameters for CO2: Mu and STD")
print(mu_co2, std_co2)

# Estimating parameters for Oil Total
mu_oil, std_oil = norm.fit(df['Oil_Total'])
print("Normal Distribution parameters for Oil: Mu and STD")
print(mu_oil, std_oil)

# 2. Implement Gaussian Mixture Models
# Fitting a GMM to the data
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(df)

# 3. Sample a New Data Point and Visualize
# Sampling a new data point
sampled_point, _ = gmm.sample(1)

# Visualization
plt.figure(figsize=(12, 6))

# Plotting the original data
plt.scatter(df['Oil_Total'], df['CO2_Total'], label='Original Data')

# Plotting the sampled data point
plt.scatter(sampled_point[:, 0], sampled_point[:, 1], color='red', label='Sampled Data Point')

# Adding legends and labels
plt.title('Gaussian Mixture Model and Sampled Data Point')
plt.xlabel('Oil_Total')
plt.ylabel('CO2_Total')
plt.legend()

plt.show()


print('Synthetic point: Predicting the amount of CO" produced given the Oil produced is:')
print(sampled_point[0][0])
print("Predicted CO2:")
print(sampled_point[0][1])











#--------------------------------------------------------
#Second Task - Renewable Energy vs Rate of increase of CO2 Emission

#We look at the total amount of renewable energy produced per year
df_renew = pd.read_excel(path + "Statistical Review of World Energy Data.xlsx", sheet_name="Renewable power - TWh", skiprows=2)
df_renew_total = df_renew.loc[107]
df_renew_total = df_renew_total.iloc[1:-3]

#Recall that the amount of CO2 generated per year was loaded previously
df_co2_total

#Merge the two datasets with overlapping years together
df_renew_co2 = pd.merge(df_renew_total, df_co2_total, left_index=True, right_index=True)
df_renew_co2.columns = ['Renewable Energy Generation','CO2 Emission']



#Do a plot to see how renewable energy generation changes with CO2 Emission
plt.figure(figsize=(12, 6))
plt.scatter(df_renew_co2['Renewable Energy Generation'],df_renew_co2['CO2 Emission'], label='Renewable Energy Consumption vs CO2 Emission')
plt.xlabel('Renewable Energy Generation')
plt.ylabel('CO2 Emission')
plt.title('Renewable Energy Generation vs CO2 Emission')
plt.show()
'''
From the plot, we can clearly see that although CO2 emission is still increasing...
Producing more renewable energy clearly reduces the RATE of increase of CO2 emission
'''





#Based on that finding, we try to evaluate the production of renewable energy versus RATE of increase of CO2
#Firstly, calculate the year-to-year difference in CO2 Emissions 
df_renew_co2['Rate of change of CO2 Emission'] = df_renew_co2['CO2 Emission'].diff()

#Then, calculate the 5-year moving average (The average of the most recent 5 years)
df_renew_co2['5 Year Moving Average Rate of Change of CO2 Emission'] = df_renew_co2['Rate of change of CO2 Emission'].rolling(window=5).mean()

#We then plot the renewable energy generation versus 5 year moving average CO2 production
plt.figure(figsize=(12, 6))
plt.scatter(df_renew_co2['Renewable Energy Generation'], df_renew_co2['5 Year Moving Average Rate of Change of CO2 Emission'], label='Renewable Energy Consumption vs 5-Year Average Rate of Change of CO2 Emission')
plt.xlabel('Renewable Energy Generation')
plt.ylabel('5 Year Moving Average Rate of Change of CO2 Emission')
plt.title('Renewable Energy Generation vs 5 Year Moving Average Rate of Change of CO2 Emission')

plt.show()


'''
Based on this, we now see a trend: At first, when the amount of renewable energy produced was too negligible and small,
the year to year CO2 emission was still increasing at an alarming rate

But once the amount of renewawble energy increased sufficiently (More than 500), we see the CO2 emission increasing slower and slower
i.e. renewable energy is effective in reducing CO2 emission
'''


#As a result of the above's finding, we remove the anomalies at the start where renewable energy generation is less than 500
df_renew_co2 = df_renew_co2[df_renew_co2['Renewable Energy Generation'] >= 500]


#The plot now looks something like this, as expected. Clear downward trend!
plt.figure(figsize=(12, 6))
plt.scatter(df_renew_co2['Renewable Energy Generation'], df_renew_co2['5 Year Moving Average Rate of Change of CO2 Emission'], label='Renewable Energy Consumption vs 5-Year Average Rate of Change of CO2 Emission')
plt.xlabel('Renewable Energy Generation')
plt.ylabel('5 Year Moving Average Rate of Change of CO2 Emission')
plt.title('Renewable Energy Generation vs 5 Year Moving Average Rate of Change of CO2 Emission')
plt.show()







'''
There seems to be gaps in the plot earlier between energy=3000 and energy=3500. Let's generate 2 synthetic points using GMM'
'''
# Extract relevant columns
data = df_renew_co2[['Renewable Energy Generation', '5 Year Moving Average Rate of Change of CO2 Emission']]

# Fit a Gaussian Mixture Model (GMM) to the data
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data)

# Generate new points for Renewable Energy Generation = 3000 and 3500 using GMM
new_points = np.array([[3000, 0], [3500, 0]])  # Assuming the rate of change is set to 0 for the new points

# Use the GMM to predict the component (cluster) assignment for the new points
component_assignments = gmm.predict(new_points)

# Generate synthetic data points for the new points based on the predicted component means and covariances
synthetic_points = [np.random.multivariate_normal(gmm.means_[component], gmm.covariances_[component]) for component in component_assignments]

# Display the results
print("Synthetic Points:")
print(synthetic_points)


# Plot the original data and the synthetic points
plt.figure(figsize=(12, 6))
plt.scatter(data['Renewable Energy Generation'], data['5 Year Moving Average Rate of Change of CO2 Emission'], label='Original Data')
plt.scatter(new_points[:, 0], [point[1] for point in synthetic_points], color='blue', marker='x', label='Synthetic Points')
plt.xlabel('Renewable Energy Generation')
plt.ylabel('5 Year Moving Average Rate of Change of CO2 Emission')
plt.title('Renewable Energy Generation vs 5 Year Moving Average Rate of Change of CO2 Emission')
plt.legend()
plt.show()







'''
Machine Learning

Now that the dataset is ready (i.e. we see a clear trend), we start fitting a machine learning model
Let's fit 2 machine learning models (Linear Regression and Gradient Boosting), and then use MLE to evaluate the model fit quality
'''
from sklearn.linear_model import LinearRegression
# Extract relevant columns
X = df_renew_co2[['Renewable Energy Generation']]
y = df_renew_co2['5 Year Moving Average Rate of Change of CO2 Emission']

# Initialize and fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Display the coefficients
print('Linear Regression Coefficients:')
print(f'Intercept: {model.intercept_}')
print(f'Coefficient for Renewable Energy Generation: {model.coef_[0]}')

# Predictions using the linear regression model
predictions = model.predict(X)

# Calculate the residuals (differences between actual and predicted values)
residuals = y - predictions






'''
Linear Regression has a useful feature which allows us to predict and estimate how much Renewable Energy Generation is needed before the CO2 Emissions start reducing (Falls below 0)
We extrapolate the data to see where the y axis hits 0 (Shown on the red point on the graph)
'''
renewable_energy_for_y_0 = (0 - model.intercept_) / model.coef_[0]


# Plot the original data, synthetic points, and the point for y = 0
plt.figure(figsize=(12, 6))
plt.scatter(data['Renewable Energy Generation'], data['5 Year Moving Average Rate of Change of CO2 Emission'], label='Original Data')
plt.scatter(new_points[:, 0], [point[1] for point in synthetic_points], color='blue', marker='x', label='Synthetic Points')
plt.scatter(renewable_energy_for_y_0, 0, color='red', marker='o', label='Point for y = 0')  # Adding the point for y = 0
plt.xlabel('Renewable Energy Generation')
plt.ylabel('5 Year Moving Average Rate of Change of CO2 Emission')
plt.title('Renewable Energy Generation vs 5 Year Moving Average Rate of Change of CO2 Emission')
plt.legend()
plt.show()













'''
Using MLE (Maximum Likelihood Estimation) to evaluate how good is the model fit:
    Maximising the likelihood is equivalent to maximising the log likelihood.
    Bigger (less negative) likelihood indicates better fit
'''
# Calculate the log-likelihood using the normal distribution assumption
log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(np.mean(residuals**2)) + (residuals ** 2) / np.mean(residuals**2))

print(f'\nLog-Likelihood: {log_likelihood}')





from sklearn.ensemble import GradientBoostingRegressor
# Initialize and fit a Gradient Boosting Regressor model
model_rf = GradientBoostingRegressor(random_state=42)
model_rf.fit(X, y)

# Predictions using the random forest model
predictions_rf = model_rf.predict(X)

# Calculate the residuals (differences between actual and predicted values)
residuals_rf = y - predictions_rf

# Calculate the log-likelihood using the normal distribution assumption
log_likelihood_rf = -0.5 * np.sum(np.log(2 * np.pi) + np.log(np.var(residuals_rf)) + (residuals_rf ** 2) / np.var(residuals_rf))

# Display results
print('Gradient Boosting Regressor Model:')
print(f'Log-Likelihood: {log_likelihood_rf}')


'''
As the log likelihood is larger for gradient boosting regressor, this indicates that the model is a better fit compared to linear regression by MLE principles
'''