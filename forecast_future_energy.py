import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

# Load the data
data = pd.read_excel('prediction.xlsx', sheet_name='Sheet1')

# 转换日期列为datetime类型，并设为索引
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 创建一个新的日期范围，包括未来的日期
last_known_date = data.index.max()
future_end_date = pd.Timestamp('2023-06-30')  # 预测结束日期
all_dates = pd.date_range(start=data.index.min(), end=future_end_date, freq='D')

# 扩展原始DataFrame到新的时间范围
data = data.reindex(all_dates)

#应用初始调整
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

# 函数：为每种能源类型创建并训练Prophet模型，进行预测和交叉验证
def predict_energy_with_cv(energy_name, initial='730 days', period='180 days', horizon='90 days'):
    # 保留原始观测值用于绘图
    original_data = data[energy_name].copy()

    # 对数据进行对数转换
    data[energy_name] = np.log1p(data[energy_name])

    energy_df = data.reset_index().rename(columns={'index': 'ds', energy_name: 'y'})
    energy_df.dropna(subset=['y'], inplace=True)  # 确保没有缺失值

    # 创建并训练Prophet模型
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(energy_df)

    # 执行交叉验证
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv)
    print(f'Performance metrics for {energy_name}:\n', df_p[['horizon', 'mse', 'rmse', 'mae', 'mape', 'coverage']])

    # 创建未来数据框架
    future = model.make_future_dataframe(periods=246, freq='D')
    forecast = model.predict(future)

    # 将预测结果转换回原始比例
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # 绘制预测结果并保存图像
    fig = model.plot(forecast)
    plt.scatter(energy_df['ds'], original_data.dropna(), color='black', marker='.', label='Actual')
    add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.title(f'Forecast and Cross-Validation Results for {energy_name}')
    plt.legend()
    plt.savefig(f'{energy_name}_forecast_cv.png')
    plt.close(fig)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# 预测每种能源，并保存结果到一个Excel文件
energy_types = ['Nuclear', 'Coal', 'Natural Gas', 'Renewables']
with pd.ExcelWriter('energy_forecasts.xlsx') as writer:
    for energy in energy_types:
        forecast = predict_energy_with_cv(energy)
        forecast.to_excel(writer, sheet_name=energy)

print("Forecasts and cross-validation results saved to energy_forecasts.xlsx and images saved in the project directory.")

#预测因变量
# 加载数据
file_path = 'power limitations.xlsx'
data = pd.read_excel(file_path)

# 准备数据
df_prophet = data.rename(columns={'Date': 'ds', 'power limitations': 'y'})

# 初始化 Prophet 模型，启用日、周、年季节性
model = Prophet(daily_seasonality=True, yearly_seasonality=True)

# 拟合模型
model.fit(df_prophet)

# 创建未来日期的 DataFrame 以进行预测，同时保留历史数据以绘制对比图
future_dates = model.make_future_dataframe(periods=(pd.to_datetime("2023-06-30") - pd.to_datetime("2022-09-23")).days + 1)

# 进行预测
forecast = model.predict(future_dates)

# 进行交叉验证
df_cv = cross_validation(model, initial='365 days', period='90 days', horizon = '180 days')

# 计算性能指标
df_p = performance_metrics(df_cv)

# 显示性能指标
print(df_p.head())


# 显示预测结果
predicted_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# 绘制预测结果和历史数据的对比图
plt.figure(figsize=(10, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Historical Data')
plt.plot(predicted_data['ds'], predicted_data['yhat'], label='Predicted Data', color='red')
plt.fill_between(predicted_data['ds'], predicted_data['yhat_lower'], predicted_data['yhat_upper'], color='gray', alpha=0.2)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Power Limitations')
plt.title('Forecast vs Historical Data')

# 保存图像到当前项目
plt.savefig('forecast_vs_historical_power_limitations.png')

# 显示图
plt.show()

# 保存预测结果到 Excel 文件
predicted_data.to_excel('predicted_power_limitations.xlsx', index=False)
