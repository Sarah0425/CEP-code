import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 读取Excel数据
df = pd.read_excel('data for the whole period.xlsx', sheet_name='Sheet1')

# Drop 'Date', 'Demand_difference', and 'CII' columns for descriptive statistics
columns_to_describe = df.drop(columns=['Date', 'Demand_difference (MW)', 'CII'])

# Calculate descriptive statistics
descriptive_stats = columns_to_describe.describe()

# Save the descriptive statistics to a new Excel file
output_excel_path = 'descriptive_statistics.xlsx'
descriptive_stats.to_excel(output_excel_path)

# Display descriptive statistics
print(descriptive_stats)

# Set the Date column as the index
df.set_index('Date', inplace=True)
# Generating individual plots for each column
columns_to_plot = columns_to_describe.columns
for column in columns_to_plot:
    plt.figure(figsize=(12, 6))
    # 绘制整个时间序列
    sns.lineplot(data=df[column], label=f'{column}')
    # 高亮12月和1月的数据点
    highlighted_data = df[column][(df.index.month == 12) | (df.index.month == 1)]
    plt.scatter(highlighted_data.index, highlighted_data, color='purple', label='Dec-Jan Data', zorder=3)
    # 添加战争开始的红色垂直线
    plt.axvline(pd.Timestamp('2022-02-22'), color='red', linestyle='--', linewidth=2, label='Start of Conflict')
    plt.title(f'Time Series Plot for {column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{column}_timeseries_plot.png')
    plt.close()