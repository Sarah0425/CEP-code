import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import shap
from scipy.stats import ttest_ind

# Load the data
data = pd.read_excel('data for the whole period.xlsx', sheet_name='Sheet1')

# 转换日期列
data['Date'] = pd.to_datetime(data['Date'])

# 创建战争标签
war_date = pd.to_datetime('2022-02-22')
data['War Period'] = data['Date'] >= war_date

# 生成时间特征
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear

# 创建组合变量
data['Nuclear_Coal'] = data['Nuclear (%)'] + data['Coal (%)']
data['Nuclear_NaturalGas'] = data['Nuclear (%)'] + data['Natural Gas (%)']
data['Nuclear_Renewables'] = data['Nuclear (%)'] + data['Renewables (%)']
data['Coal_NaturalGas'] = data['Natural Gas (%)'] + data['Coal (%)']
data['Coal_Renewables'] = data['Coal (%)'] + data['Renewables (%)']
data['NaturalGas_Renewables'] = data['Natural Gas (%)'] + data['Renewables (%)']

#regional 交互项
data['Nuclear_regional'] = data['Nuclear (%)'] * (1/(1 + data['Nuclear_Region_CII']))
data['Coal_regional'] = data['Coal (%)'] * (1/(1 + data['Coal_Region_CII']))
data['NaturalGas_regional'] = data['Natural Gas (%)'] * (1/(1 + data['Gas_Region_CII']))
data['Renewables_regional'] = data['Renewables (%)'] * (1/(1 + data['Renewables_Region_CII']))

data['NaturalGas_trade'] = data['Natural Gas (%)'] * data['Net Trade']

# 按奇数日和偶数日划分训练集和测试集
train_data = data[data['DayOfYear'] % 2 != 0]
test_data = data[data['DayOfYear'] % 2 == 0]

# 选择特征和目标变量（包括主要时间特征和分解后的主要成分）
features = ['Nuclear (%)', 'Coal (%)', 'Natural Gas (%)', 'Renewables (%)',
            'Net Trade', 'Temperature',
            'Demand_difference', 'CII',
           # 'Nuclear_Coal', 'Nuclear_Renewables', 'Coal_Renewables', 'NaturalGas_Renewables',
           # 'Nuclear_regional', 'Coal_regional', 'NaturalGas_regional', 'Renewables_regional'
            ]
target = 'power limitations'

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用默认参数训练随机森林模型
model_default = RandomForestRegressor(n_estimators=300, random_state=42)
model_default.fit(X_train_scaled, y_train)

# 在测试数据上进行预测
y_pred_default = model_default.predict(X_test_scaled)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred_default))
r2 = r2_score(y_test, y_pred_default)

# 计算目标变量的平均值
mean_y_test = np.mean(y_test)

# 计算无量纲化的RMSE
dimensionless_rmse = rmse / mean_y_test

print(f'RMSE: {rmse}')
print(f'R²: {r2}')
print(f'non-dimensional RMSE: {dimensionless_rmse}')


# 获取特征重要性
feature_importances = model_default.feature_importances_

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()

# 分别训练两个模型，比较战争前和战争期间的特征重要性
pre_war_data = data[data['War Period'] == False]
during_war_data = data[data['War Period'] == True]

X_pre_war = pre_war_data[features]
y_pre_war = pre_war_data[target]

X_during_war = during_war_data[features]
y_during_war = during_war_data[target]

X_pre_war_scaled = scaler.fit_transform(X_pre_war)
X_during_war_scaled = scaler.transform(X_during_war)

model_pre_war = RandomForestRegressor(n_estimators=300, random_state=42)
model_pre_war.fit(X_pre_war_scaled, y_pre_war)

model_during_war = RandomForestRegressor(n_estimators=300, random_state=42)
model_during_war.fit(X_during_war_scaled, y_during_war)

feature_importances_pre_war = model_pre_war.feature_importances_
feature_importances_during_war = model_during_war.feature_importances_

# 比较特征重要性
importance_comparison = pd.DataFrame({
    'Feature': features,
    'Importance (Pre-War)': feature_importances_pre_war,
    'Importance (During-War)': feature_importances_during_war
})

print('\nImportance Comparison:')
print(importance_comparison)

# 创建数据框用于可视化
importance_comparison = pd.DataFrame({
    'Feature': features,
    'Importance (Pre-War)': feature_importances_pre_war,
    'Importance (During-War)': feature_importances_during_war
})

# 设置图形的大小
plt.figure(figsize=(12, 8))

# 为战前和战时的特征重要性绘制条形图
width = 0.35  # 条形的宽度
n = len(features)  # 特征的数量
index = np.arange(n)  # 组成一个适合条形图的索引数组

# 绘制条形图
plt.barh(index, importance_comparison['Importance (During-War)'], height=width, color='purple', label='During-War')
plt.barh(index + width, importance_comparison['Importance (Pre-War)'], height=width, color='pink', label='Pre-War')

# 添加图表标签和标题
plt.xlabel('Feature Importance')
plt.title('Comparison of Feature Importance Pre-War and During-War')
plt.yticks(index + width / 2, importance_comparison['Feature'])  # 设置y轴刻度标签
plt.legend()  # 添加图例

# 显示图形
plt.show()

# 计算和打印 SHAP 值，并绘制图形
explainer_pre_war = shap.TreeExplainer(model_pre_war)
shap_values_pre_war = explainer_pre_war.shap_values(X_pre_war_scaled)

explainer_during_war = shap.TreeExplainer(model_during_war)
shap_values_during_war = explainer_during_war.shap_values(X_during_war_scaled)

# 确保SHAP值正确反映特征对目标的正负向影响
# 打印SHAP值
print("SHAP Values (Pre-War):")
print(shap_values_pre_war)
print("SHAP Values (During-War):")
print(shap_values_during_war)

# 可视化SHAP值
shap.summary_plot(shap_values_pre_war, X_pre_war, plot_type="bar", show=False)
plt.title('SHAP Summary Plot (Pre-War)')
plt.show()

shap.summary_plot(shap_values_during_war, X_during_war, plot_type="bar", show=False)
plt.title('SHAP Summary Plot (During-War)')
plt.show()

shap.summary_plot(shap_values_pre_war, X_pre_war, show=False)
plt.title('SHAP Summary Plot (Pre-War)')
plt.show()

shap.summary_plot(shap_values_during_war, X_during_war, show=False)
plt.title('SHAP Summary Plot (During-War)')
plt.show()

# 计算每个特征高低值的平均SHAP值
def calculate_shap_mean_high_low(shap_values, feature_values):
    high_shap_mean = []
    low_shap_mean = []
    threshold = np.median(feature_values, axis=0)

    for i in range(feature_values.shape[1]):
        high_shap_mean.append(np.mean(shap_values[feature_values[:, i] > threshold[i], i]))
        low_shap_mean.append(np.mean(shap_values[feature_values[:, i] <= threshold[i], i]))

    return np.array(high_shap_mean), np.array(low_shap_mean)


# 计算战争前和战争期间每个特征高低值的平均SHAP值
high_shap_pre_war, low_shap_pre_war = calculate_shap_mean_high_low(shap_values_pre_war, X_pre_war_scaled)
high_shap_during_war, low_shap_during_war = calculate_shap_mean_high_low(shap_values_during_war, X_during_war_scaled)

# 计算变化
shap_difference_high = high_shap_during_war - high_shap_pre_war
shap_difference_low = low_shap_during_war - low_shap_pre_war

# 统计显著性测试
p_values_high = []
p_values_low = []
for i in range(len(features)):
    t_stat_high, p_val_high = ttest_ind(
        shap_values_pre_war[:, i][X_pre_war_scaled[:, i] > np.median(X_pre_war_scaled[:, i])],
        shap_values_during_war[:, i][X_during_war_scaled[:, i] > np.median(X_during_war_scaled[:, i])],
        equal_var=False)
    p_values_high.append(p_val_high)

    t_stat_low, p_val_low = ttest_ind(
        shap_values_pre_war[:, i][X_pre_war_scaled[:, i] <= np.median(X_pre_war_scaled[:, i])],
        shap_values_during_war[:, i][X_during_war_scaled[:, i] <= np.median(X_during_war_scaled[:, i])],
        equal_var=False)
    p_values_low.append(p_val_low)

# 创建结果表格
result_high = pd.DataFrame({
    'Feature': features,
    'Mean SHAP (Pre-War High)': high_shap_pre_war,
    'Mean SHAP (During-War High)': high_shap_during_war,
    'SHAP Difference (High)': shap_difference_high,
    'p-value (High)': p_values_high
})

result_low = pd.DataFrame({
    'Feature': features,
    'Mean SHAP (Pre-War Low)': low_shap_pre_war,
    'Mean SHAP (During-War Low)': low_shap_during_war,
    'SHAP Difference (Low)': shap_difference_low,
    'p-value (Low)': p_values_low
})

# 判断显著性
significance_level = 0.05
result_high['Significant (High)'] = result_high['p-value (High)'] < significance_level
result_low['Significant (Low)'] = result_low['p-value (Low)'] < significance_level

# 打印结果
print("High Feature Value SHAP Analysis:")
print(result_high)

print("\nLow Feature Value SHAP Analysis:")
print(result_low)

# 保存结果到Excel文件
with pd.ExcelWriter('SHAP_value_analysis_high_low.xlsx') as writer:
    result_high.to_excel(writer, sheet_name='High Feature Values', index=False)
    result_low.to_excel(writer, sheet_name='Low Feature Values', index=False)


#使用战争时训练的随机森林模型来预测未来的 power limitations
future_data = pd.read_excel('predicted data.xlsx', sheet_name='Sheet1')

# 转换日期列
future_data['Date'] = pd.to_datetime(future_data['Date'])

# 生成时间特征
future_data['Month'] = future_data['Date'].dt.month
future_data['DayOfYear'] = future_data['Date'].dt.dayofyear

# 创建组合变量
future_data['Nuclear_Coal'] = future_data['Nuclear (%)'] + future_data['Coal (%)']
future_data['Nuclear_NaturalGas'] = future_data['Nuclear (%)'] + future_data['Natural Gas (%)']
future_data['Nuclear_Renewables'] = future_data['Nuclear (%)'] + future_data['Renewables (%)']
future_data['Coal_NaturalGas'] = future_data['Natural Gas (%)'] + future_data['Coal (%)']
future_data['Coal_Renewables'] = future_data['Coal (%)'] + future_data['Renewables (%)']
future_data['NaturalGas_Renewables'] = future_data['Natural Gas (%)'] + future_data['Renewables (%)']

# 创建区域交互项
future_data['Nuclear_regional'] = future_data['Nuclear (%)'] * (1/(1+future_data['Nuclear_Region_CII']))
future_data['Coal_regional'] = future_data['Coal (%)'] * (1/(1+future_data['Coal_Region_CII']))
future_data['NaturalGas_regional'] = future_data['Natural Gas (%)'] * (1/(1+future_data['Gas_Region_CII']))
future_data['Renewables_regional'] = future_data['Renewables (%)'] * (1/(1+future_data['Renewables_Region_CII']))

future_data['NaturalGas_trade'] = future_data['Natural Gas (%)'] * future_data['Net Trade']

# 选择相同的特征
features = ['Nuclear (%)', 'Coal (%)', 'Natural Gas (%)', 'Renewables (%)',
            'Net Trade', 'Temperature',
            'Demand_difference', 'CII',
            # 'Nuclear_Coal', 'Nuclear_Renewables', 'Coal_Renewables', 'NaturalGas_Renewables',
            # 'Nuclear_regional', 'Coal_regional', 'NaturalGas_regional', 'Renewables_regional'
            ]

# 标准化处理
X_future = future_data[features]
scaler = StandardScaler()
X_future_scaled = scaler.fit_transform(X_future)

# 使用战争时期训练的模型进行预测
y_future_pred = model_default.predict(X_future_scaled)

# 保存预测结果到Excel
predictions = pd.DataFrame({
    'Date': future_data['Date'],
    'Predicted Power Limitations': y_future_pred
})

predictions.to_excel('future_predictions_power_limitations.xlsx', index=False)

import matplotlib.pyplot as plt

# 绘制预测结果图像
plt.figure(figsize=(12, 6))
plt.plot(predictions['Date'], predictions['Predicted Power Limitations'], marker='o', linestyle='-', color='purple', label='Predicted Power Limitations')
plt.xlabel('Date')
plt.ylabel('Power Limitations (MW)')
plt.title('Predicted Power Limitations Over Time')
plt.grid(True)
plt.xticks(rotation=45)  # 旋转日期标签以提高可读性
plt.tight_layout()  # 调整布局以确保没有内容被裁剪
plt.legend()
plt.show()
