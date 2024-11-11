import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

file_path = r'C:\Users\ZainHafizMuhammad\Downloads\coding_challenge\analysis_task_data.xlsx'
df = pd.read_excel(file_path, parse_dates=['time']).set_index('time')

# TASK 2.1: Calculate total forecast in MWh for each category
columns_to_process = [
    'Wind Day Ahead Forecast [in MW]',
    'Wind Intraday Forecast [in MW]',
    'PV Day Ahead Forecast [in MW]',
    'PV Intraday Forecast [in MW]'
]
total_forecast_mwh = df.resample('H')[columns_to_process].sum().sum()  # Sum by hour then totalize
for column, total in total_forecast_mwh.items():
    print(f"Total {column} in MWh for 2021: {total}")

# TASK 2.2: average hourly production over a day
hourly_avg = df.groupby(df.index.hour)[columns_to_process].mean()

plt.figure(figsize=(10, 5))
for col, color in zip(columns_to_process, ['blue', 'cyan', 'red', 'yellow']):
    plt.plot(hourly_avg.index, hourly_avg[col], label=col.replace(' Forecast [in MW]', ''), color=color)

plt.title('Average Wind and Solar Production Over a 24-Hour Period')
plt.xlabel('Hour of Day')
plt.ylabel('Average Production (MW)')
plt.xticks(range(0, 24))
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
# TASK 2.3:  average value for wind and PV using day-ahead forecasts and day-ahead prices

price_column = 'Day Ahead Price hourly [in EUR/MWh]'
wind_forecast_column = 'Wind Day Ahead Forecast [in MW]'
pv_forecast_column = 'PV Day Ahead Forecast [in MW]'

df['Wind Revenue'] = df[wind_forecast_column] * df[price_column]
df['PV Revenue'] = df[pv_forecast_column] * df[price_column]

total_wind_revenue = df['Wind Revenue'].sum()
total_pv_revenue = df['PV Revenue'].sum()
total_wind_generation = df[wind_forecast_column].sum()
total_pv_generation = df[pv_forecast_column].sum()

average_value_wind = total_wind_revenue / total_wind_generation
average_value_pv = total_pv_revenue / total_pv_generation
average_da_price = df[price_column].mean()
print(f"Average value for Wind in 2021: {average_value_wind:.2f} EUR/MWh")
print(f"Average value for PV in 2021: {average_value_pv:.2f} EUR/MWh")
print(f"Average day-ahead price in 2021: {average_da_price:.2f} EUR/MWh")

if average_value_wind > average_da_price:
    wind_comparison = "higher"
else:
    wind_comparison = "lower"

if average_value_pv > average_da_price:
    pv_comparison = "higher"
else:
    pv_comparison = "lower"
print(f"The average value of Wind is {wind_comparison} than the average day-ahead price.")
print(f"The average value of PV is {pv_comparison} than the average day-ahead price.")

#TASK 2.4
# total daily renewable production (wind + PV) and average day-ahead price
df['Total Renewable Production'] = df['Wind Day Ahead Forecast [in MW]'] + df['PV Day Ahead Forecast [in MW]']
daily_data = df.resample('D').agg({
    'Total Renewable Production': 'sum',
    'Day Ahead Price hourly [in EUR/MWh]': 'mean'
})

highest_day = daily_data['Total Renewable Production'].idxmax()
lowest_day = daily_data['Total Renewable Production'].idxmin()

highest_day_price = daily_data.loc[highest_day, 'Day Ahead Price hourly [in EUR/MWh]']
lowest_day_price = daily_data.loc[lowest_day, 'Day Ahead Price hourly [in EUR/MWh]']
print(f"Highest renewable production day: {highest_day.date()} with an average DA price of {highest_day_price:.2f} EUR/MWh")
print(f"Lowest renewable production day: {lowest_day.date()} with an average DA price of {lowest_day_price:.2f} EUR/MWh")

explanation = ("Higher renewable production often leads to lower prices due to increased supply of low-cost energy, "
               "whereas lower renewable production requires more expensive generation, raising prices.")
print("Explanation of price differences:", explanation)
#TASK 2.5
df['is_weekend'] = df.index.weekday >= 5 
weekday_avg_price = df.loc[~df['is_weekend'], 'Day Ahead Price hourly [in EUR/MWh]'].mean()
weekend_avg_price = df.loc[df['is_weekend'], 'Day Ahead Price hourly [in EUR/MWh]'].mean()
print(f"Average hourly day-ahead price during weekdays: {weekday_avg_price:.2f} EUR/MWh")
print(f"Average hourly day-ahead price during weekends: {weekend_avg_price:.2f} EUR/MWh")

#TASK 2.6
daily_min_price = df['Day Ahead Price hourly [in EUR/MWh]'].resample('D').min()
daily_max_price = df['Day Ahead Price hourly [in EUR/MWh]'].resample('D').max()
daily_revenue = daily_max_price - daily_min_price
total_revenue = daily_revenue.sum()
print(f"Total revenue for the battery in 2021: {total_revenue:.2f} EUR")
#TASK 2.7
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Wind_DA_Forecast'] = df['Wind Day Ahead Forecast [in MW]']
df['PV_DA_Forecast'] = df['PV Day Ahead Forecast [in MW]']
df['Wind_ID_Forecast'] = df['Wind Intraday Forecast [in MW]']
df['PV_ID_Forecast'] = df['PV Intraday Forecast [in MW]']
df['Total_Renewable_Forecast'] = df['Wind_DA_Forecast'] + df['PV_DA_Forecast']
df['Imbalance_Price'] = df['Imbalance Price Quarter Hourly  [in EUR/MWh]']
df['ID_15min_Price'] = df['Intraday Price Price Quarter Hourly  [in EUR/MWh]']
df['ID_1hr_Price'] = df['Intraday Price Hourly  [in EUR/MWh]']
df['DA_ID_1hr_Price_Diff'] = df['ID_1hr_Price'] - df['Day Ahead Price hourly [in EUR/MWh]']
df['ID_1hr_15min_Price_Diff'] = df['ID_1hr_Price'] - df['ID_15min_Price']
df['Prev_DA_Price'] = df['Day Ahead Price hourly [in EUR/MWh]'].shift(1)
df['Prev_ID_1hr_Price'] = df['ID_1hr_Price'].shift(1)
df['Prev_Imbalance_Price'] = df['Imbalance_Price'].shift(1)
df['DA_Price_Rolling_Avg'] = df['Day Ahead Price hourly [in EUR/MWh]'].rolling(window=3).mean()
df['ID_1hr_Price_Rolling_Avg'] = df['ID_1hr_Price'].rolling(window=3).mean()
df['Imbalance_Price_Rolling_Avg'] = df['Imbalance_Price'].rolling(window=3).mean()
df['DA_Price_Rolling_Std'] = df['Day Ahead Price hourly [in EUR/MWh]'].rolling(window=3).std()

df.dropna(inplace=True)

features = [
    'Hour', 'DayOfWeek', 'Wind_DA_Forecast', 'PV_DA_Forecast', 'Wind_ID_Forecast', 'PV_ID_Forecast',
    'Total_Renewable_Forecast', 'Imbalance_Price', 'DA_ID_1hr_Price_Diff', 'ID_1hr_15min_Price_Diff',
    'Prev_DA_Price', 'Prev_ID_1hr_Price', 'Prev_Imbalance_Price',
    'DA_Price_Rolling_Avg', 'ID_1hr_Price_Rolling_Avg', 'Imbalance_Price_Rolling_Avg', 'DA_Price_Rolling_Std'
]
X = df[features]
y = df['DA_ID_1hr_Price_Diff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


df['Predicted_Price_Diff'] = model.predict(X)
transaction_cost = 1.5  # EUR/MWh

df['Position'] = np.where(df['Predicted_Price_Diff'] > 2 + transaction_cost, 100,  
                          np.where(df['Predicted_Price_Diff'] < -2 - transaction_cost, -100, 0))  

df['Daily_Profit'] = df['Position'] * (df['DA_ID_1hr_Price_Diff'] - transaction_cost * np.sign(df['Position']))
df['Cumulative_Profit'] = df['Daily_Profit'].cumsum()

plt.figure(figsize=(10, 5))
plt.plot(df['Cumulative_Profit'], label='Cumulative Profit')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit (EUR)')
plt.title('Cumulative Performance of the Trading Strategy (100 MW Position)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

total_profit = df['Cumulative_Profit'].iloc[-1]
print(f"Total cumulative profit for the advanced strategy in 2021: {total_profit:.2f} EUR")

# Additional Analysis: Compare Actual vs Predicted Price Differences
df['Prediction_Error'] = df['DA_ID_1hr_Price_Diff'] - df['Predicted_Price_Diff']

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['DA_ID_1hr_Price_Diff'], label='Actual Price Difference', color='blue')
plt.plot(df.index, df['Predicted_Price_Diff'], label='Predicted Price Difference', color='orange')
plt.xlabel('Date')
plt.ylabel('Price Difference (EUR/MWh)')
plt.title('Actual vs Predicted Price Difference (Day-Ahead - Intraday Hourly)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Prediction_Error'], label='Prediction Error', color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
plt.xlabel('Date')
plt.ylabel('Prediction Error (EUR/MWh)')
plt.title('Prediction Error Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()