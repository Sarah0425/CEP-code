import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# load Excel data
data = pd.read_excel('test model.xlsx', sheet_name='Sheet1')

data['Date'] = pd.to_datetime(data['Date'])
data.rename(columns={'Date': 'ds', 'Total Demand': 'y'}, inplace=True)


pre_war_data = data.loc[data['ds'] <= '2022-02-22']

post_war_data = data.loc[data['ds'] > '2022-02-22']

# build Prophet
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# add seasonality
model.add_seasonality(name='annual', period=365, fourier_order=10)
model.add_seasonality(name='weekly', period=7, fourier_order=3)

model.fit(pre_war_data)

# forecast for future 630-days
future = model.make_future_dataframe(periods=630)
forecast = model.predict(future)

# Plot the actual value versus the predicted value
fig, ax = plt.subplots(figsize=(14, 7))
model.plot(forecast, ax=ax)
ax.plot(pre_war_data['ds'], pre_war_data['y'], label='Actual', color='black')
ax.legend()
plt.xlabel('Date')
plt.ylabel('Total demand (MW)')
plt.title('Prophet Model - Actual vs Forecast1')
plt.savefig('prophet_combined_prediction.png')
plt.close()

#to show demand_difference
fig, ax = plt.subplots(figsize=(14, 7))
model.plot(forecast, ax=ax)
ax.plot(pre_war_data['ds'], pre_war_data['y'], label='Actual (Pre-War)', color='black')
ax.plot(post_war_data['ds'], post_war_data['y'], label='Actual (Post-War)', color='green')
ax.legend()
plt.xlabel('Date')
plt.ylabel('Total demand (MW)')
plt.title('Prophet Model - Actual vs Forecast2')
plt.savefig('demand_difference.png')
plt.close()

# show the forecast results
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365))
# save
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel('forecast_results.xlsx', index=False)
