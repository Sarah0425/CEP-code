import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_excel('data for the whole period.xlsx', sheet_name='Sheet1')

# Set the Date column as the index
data.set_index('Date', inplace=True)

# Function to perform seasonal decomposition and adjust the data
def decompose_and_adjust(data, column_name, period):
    decomposition = seasonal_decompose(data[column_name], model='additive', period=period)
    trend = decomposition.trend.interpolate()  # Interpolate to fill NaN values in trend
    residuals = decomposition.resid.interpolate()  # Interpolate to fill NaN values in residuals
    return trend, residuals

# List of variables to decompose
variables = ['power limitations', 'Temperature', 'Net Trade', 'Total Generation', 'Nuclear (%)', 'Natural Gas (%)', 'Coal (%)', 'Renewables (%)']

# Perform seasonal decomposition for each variable
for var in variables:
    data[f'{var}_trend'], data[f'{var}_residuals'] = decompose_and_adjust(data, var, 365)

# Drop rows with NaN values that might still exist after interpolation
data.dropna(inplace=True)

# Split the data into pre-war and war-time periods
war_start_date = '2022-02-22'  # Replace with the actual date
pre_war_data = data[data.index <= war_start_date]
war_data = data[data.index > war_start_date]

# Function to prepare the data, add interaction terms, and fit the model
def prepare_and_fit(data):
    control_variables = ['Temperature_residuals', 'Net Trade_residuals', 'Total Generation_residuals']
    independent_variables = ['Nuclear (%)_residuals', 'Natural Gas (%)_residuals', 'Renewables (%)_residuals', 'Demand_difference', 'CII']

    # Adding interaction terms
    data['Nuclear_NaturalGas'] = data['Nuclear (%)_residuals'] * data['Natural Gas (%)_residuals']
    data['Nuclear_Renewables'] = data['Nuclear (%)_residuals'] * data['Renewables (%)_residuals']
    data['NaturalGas_Renewables'] = data['Natural Gas (%)_residuals'] * data['Renewables (%)_residuals']
    data['Nuclear_NaturalGas_Renewables'] = data['Nuclear (%)_residuals'] * data['Natural Gas (%)_residuals'] * data['Renewables (%)_residuals']

    independent_variables += ['Nuclear_NaturalGas', 'Nuclear_Renewables', 'NaturalGas_Renewables', 'Nuclear_NaturalGas_Renewables']

    # Split the dataset into training and testing sets
    train_data = data[data.index.day % 2 != 0]  # Odd days for training
    test_data = data[data.index.day % 2 == 0]   # Even days for testing

    X_train = train_data[independent_variables + control_variables]
    y_train = train_data['power limitations_residuals']
    X_test = test_data[independent_variables + control_variables]
    y_test = test_data['power limitations_residuals']

    # Initialize and train the linear regression model using statsmodels
    X_train_sm = sm.add_constant(X_train)  # Add constant term for statsmodels
    model_sm = sm.OLS(y_train, X_train_sm).fit()

    return model_sm, X_train_sm, y_train, X_test, y_test

# Fit models for pre-war and war-time periods
model_pre_war, X_train_pre_war_sm, y_train_pre_war, X_test_pre_war, y_test_pre_war = prepare_and_fit(pre_war_data)
model_war, X_train_war_sm, y_train_war, X_test_war, y_test_war = prepare_and_fit(war_data)

# Print the summary to evaluate parameter significance
print("Pre-War Model Summary:")
print(model_pre_war.summary())
print("\nWar-Time Model Summary:")
print(model_war.summary())

# Predict on the training and testing data using statsmodels model
train_predictions_pre_war_sm = model_pre_war.predict(X_train_pre_war_sm)
test_predictions_pre_war_sm = model_pre_war.predict(sm.add_constant(X_test_pre_war))

train_predictions_war_sm = model_war.predict(X_train_war_sm)
test_predictions_war_sm = model_war.predict(sm.add_constant(X_test_war))

# Calculate the performance metrics using statsmodels model
train_mse_pre_war_sm = mean_squared_error(y_train_pre_war, train_predictions_pre_war_sm)
test_mse_pre_war_sm = mean_squared_error(y_test_pre_war, test_predictions_pre_war_sm)
train_r2_pre_war_sm = r2_score(y_train_pre_war, train_predictions_pre_war_sm)
test_r2_pre_war_sm = r2_score(y_test_pre_war, test_predictions_pre_war_sm)

train_mse_war_sm = mean_squared_error(y_train_war, train_predictions_war_sm)
test_mse_war_sm = mean_squared_error(y_test_war, test_predictions_war_sm)
train_r2_war_sm = r2_score(y_train_war, train_predictions_war_sm)
test_r2_war_sm = r2_score(y_test_war, test_predictions_war_sm)

# Output performance metrics for statsmodels model
print(f"Pre-War - Training MSE: {train_mse_pre_war_sm}, Test MSE: {test_mse_pre_war_sm}")
print(f"Pre-War - Training R2: {train_r2_pre_war_sm}, Test R2: {test_r2_pre_war_sm}")

print(f"War-Time - Training MSE: {train_mse_war_sm}, Test MSE: {test_mse_war_sm}")
print(f"War-Time - Training R2: {train_r2_war_sm}, Test R2: {test_r2_war_sm}")

# Compare coefficients
coefficients_pre_war = model_pre_war.params
coefficients_war = model_war.params

coef_comparison = pd.DataFrame({
    'Pre-War Coefficients': coefficients_pre_war,
    'War-Time Coefficients': coefficients_war
})

print("\nCoefficient Comparison:")
print(coef_comparison)

# Plot the comparison
coef_comparison.plot(kind='bar', figsize=(14, 7))
plt.title('Coefficient Comparison: Pre-War vs War-Time')
plt.xlabel('Variables')
plt.ylabel('Coefficient Values')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

# Visualize the relationship for selected interaction terms in both periods
interaction_terms = ['Nuclear_NaturalGas', 'Nuclear_Renewables', 'NaturalGas_Renewables', 'Nuclear_NaturalGas_Renewables']
for term in interaction_terms:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pre_war_data[term], y=pre_war_data['power limitations_residuals'], label='Pre-War', color='blue')
    sns.scatterplot(x=war_data[term], y=war_data['power limitations_residuals'], label='War-Time', color='red')
    plt.title(f'Relationship between {term} and Power Limitations Residuals: Pre-War vs War-Time')
    plt.xlabel(term)
    plt.ylabel('Power Limitations Residuals')
    plt.legend()
    plt.show()
