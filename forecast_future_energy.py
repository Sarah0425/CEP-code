import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

# Load the data
data = pd.read_excel('prediction.xlsx', sheet_name='Sheet1')

# change to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# new date range
last_known_date = data.index.max()
future_end_date = pd.Timestamp('2023-06-30')  # 预测结束日期
all_dates = pd.date_range(start=data.index.min(), end=future_end_date, freq='D')


data = data.reindex(all_dates)

#initial adjustment by the report information
adjustment_factors = {
'Nuclear': 0.5,
   'Coal': 0.40,
   'Natural Gas': 0.8,
    'Renewables': 0.64
}
for energy, factor in adjustment_factors.items():
    data.loc['2022-10-27':'2023-06-30', energy] *= factor

data.loc['2022-11-17':'2022-12-20', ['Nuclear', 'Coal', 'Natural Gas', 'Renewables']] *= 0.8
data.loc['2022-11-21':'2023-01-24', 'Nuclear'] *= 0.9
data.loc['2022-11-21':'2023-01-24', 'Renewables'] *= 1.10
data.loc['2023-02-01':'2023-03-24', ['Renewables', 'Nuclear']] *= 1.10
data.loc['2023-03-25':'2023-04-27', 'Nuclear'] *= 0.9
data.loc['2023-03-25':'2023-04-27', ['Natural Gas', 'Renewables']] *= 1.1
data.loc['2023-04-25':'2023-05-24', ['Nuclear', 'Coal']] *= 0.9
data.loc['2023-05-25':'2023-06-30', ['Nuclear', 'Renewables']] *= 0.9

# prophet model
def predict_energy_with_cv(energy_name, initial='730 days', period='180 days', horizon='90 days'):
   
    original_data = data[energy_name].copy()

    # Log conversion
   data[energy_name] = np.log1p(data[energy_name])

    energy_df = data.reset_index().rename(columns={'index': 'ds', energy_name: 'y'})
    energy_df.dropna(subset=['y'], inplace=True)  # 确保没有缺失值

  
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(energy_df)

    # Perform cross validation
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv)
    print(f'Performance metrics for {energy_name}:\n', df_p[['horizon', 'mse', 'rmse', 'mae', 'mape', 'coverage']])

    # Create a future data framework
    future = model.make_future_dataframe(periods=246, freq='D')
    forecast = model.predict(future)

    # Convert the prediction back to the original scale
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # plot and save
    fig = model.plot(forecast)
    plt.scatter(energy_df['ds'], original_data.dropna(), color='black', marker='.', label='Actual')
    add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.title(f'Forecast and Cross-Validation Results for {energy_name}')
    plt.legend()
    plt.savefig(f'{energy_name}_forecast_cv.png')
    plt.close(fig)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Predict each energy source and save the results to an Excel file
energy_types = ['Nuclear', 'Coal', 'Natural Gas', 'Renewables']
with pd.ExcelWriter('energy_forecasts.xlsx') as writer:
    for energy in energy_types:
        forecast = predict_energy_with_cv(energy)
        forecast.to_excel(writer, sheet_name=energy)

print("Forecasts and cross-validation results saved to energy_forecasts.xlsx and images saved in the project directory.")



#######an attempt to predict power limitations by prophet only, not used in the thesis
# load data
file_path = 'power limitations.xlsx'
data = pd.read_excel(file_path)


df_prophet = data.rename(columns={'Date': 'ds', 'power limitations': 'y'})

model = Prophet(daily_seasonality=True, yearly_seasonality=True)


model.fit(df_prophet)


future_dates = model.make_future_dataframe(periods=(pd.to_datetime("2023-06-30") - pd.to_datetime("2022-09-23")).days + 1)


forecast = model.predict(future_dates)


df_cv = cross_validation(model, initial='365 days', period='90 days', horizon = '180 days')


df_p = performance_metrics(df_cv)


print(df_p.head())



predicted_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


plt.figure(figsize=(10, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Historical Data')
plt.plot(predicted_data['ds'], predicted_data['yhat'], label='Predicted Data', color='red')
plt.fill_between(predicted_data['ds'], predicted_data['yhat_lower'], predicted_data['yhat_upper'], color='gray', alpha=0.2)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Power Limitations')
plt.title('Forecast vs Historical Data')


plt.savefig('forecast_vs_historical_power_limitations.png')


plt.show()

predicted_data.to_excel('predicted_power_limitations.xlsx', index=False)
