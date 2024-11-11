# Coding-challenge
```python
import requests
import sqlite3
import pandas as pd


GITHUB_URL = 'https://github.com/FlexPwr/QuantChallenge/raw/main/trades.sqlite'
DB_PATH = 'trades.sqlite'  
TABLE_NAME = 'epex_12_20_12_13' 

def download_db():
    response = requests.get(GITHUB_URL)
    response.raise_for_status()  # Check if the download was successful
    with open(DB_PATH, 'wb') as file:
        file.write(response.content)
    print("Database downloaded successfully.")

def load_data():
    # Connect to the SQLite database and load the table into a DataFrame
    with sqlite3.connect(DB_PATH) as conn:
        data = pd.read_sql_query(f'SELECT * FROM "{TABLE_NAME}"', conn)
    return data

download_db()
data = load_data()
print("Data loaded from the database:")
print(data)
```
# Notes 1.1
In this code, I defined two functions, compute_total_buy_volume and compute_total_sell_volume, to calculate the total volumes for buy and sell trades by filtering the dataset based on the 'side' column. Then, I applied these functions to the data and printed the total buy and sell volumes.
# TASK 1.1
# Defining functions to compute total buy and sell volumes
```python
def compute_total_buy_volume(data) -> float:
    return data[data['side'] == 'buy']['quantity'].sum()

def compute_total_sell_volume(data) -> float:
    return data[data['side'] == 'sell']['quantity'].sum()

#buy and sell volumes
total_buy_volume = compute_total_buy_volume(data)
total_sell_volume = compute_total_sell_volume(data)

print(f"Total Buy Volume: {total_buy_volume}")
print(f"Total Sell Volume: {total_sell_volume}")
```
# TASK 1.2
This code calculates profit and loss (PnL) for each strategy by adding up the income from sales and subtracting the costs of purchases. It goes through each strategy in the data, making it easy to add more strategies later.  The code loops through each strategy.
# PnL (Profit and Loss) for each strategy
```python
def compute_pnl(strategy_id: str, data) -> float:
    # Filter data 
    strategy_data = data[data['strategy'] == strategy_id]
    
    # Positive income for 'sell', negative for 'buy'
    pnl = strategy_data.apply(
        lambda row: row['quantity'] * row['price'] if row['side'] == 'sell' else -row['quantity'] * row['price'], 
        axis=1
    ).sum()
    
    return pnl

strategy_ids = data['strategy'].unique()
for strategy_id in strategy_ids:
    pnl = compute_pnl(strategy_id, data)
    print(f"PnL for {strategy_id}: {pnl} euros")
```
# TASK 1.3
I did not do task 1.3 as i was not feeling comfortable doing it. I have done api before but once or twice. I surely am able to do that but i need some time. 
# TASK 2
I downloaded csv and loaded into df as i am more comfortable reading data first. 
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

file_path = r'C:\Users\ZainHafizMuhammad\Downloads\coding_challenge\analysis_task_data.xlsx'
df = pd.read_excel(file_path, parse_dates=['time']).set_index('time')
```
# TASK 2.1: Total forecast in MWh for each category
Normally production in MW is multiplied by hour to get MWh, but as here i resampled it to hourly, i just needed to sum all values for each required column. 
```python
columns_to_process = [
    'Wind Day Ahead Forecast [in MW]',
    'Wind Intraday Forecast [in MW]',
    'PV Day Ahead Forecast [in MW]',
    'PV Intraday Forecast [in MW]'
]
total_forecast_mwh = df.resample('H')[columns_to_process].sum().sum()  # Sum by hour then totalize
for column, total in total_forecast_mwh.items():
    print(f"Total {column} in MWh for 2021: {total}")
```
# TASK 2.2: average hourly production over a day for whole year
As addition to previous task, i just computed hourly avg for every hour of the day for whole year. For example hour 0 would avg all 365 values for the year and with index.hour will make it over 24 hour time. and just plotting using matplotlib by using for loop for all 4 variables resulting in 4 different lines.
```python
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
```
# TASK 2.3: average value for wind and PV using day-ahead forecasts and day-ahead prices
The avg hourly revenue is simply multiplication of wind or solar production with the hourly price. So i took 3 columns, multiplied wind forecast with da hourly and pv forecast and made new columns named Wind and PV revenue.I just summed these 2 new columns and got avg revenue for solar n wing in EUR/MWh. similar with total generation in MW. For wind and PV, the average value was calculated by dividing the total revenue by the total generation and avg da price by simply taking mean of price column. 
To Determine if the average value for Wind and PV is higher or lower than the average day-ahead price, it used if else loop to comapre avg value wind, great or less than hourly da price and similar with pv. Turns out avg value of both wind and solar is less that avg day ahead price as there is variability in renewable production as always due to weather temperature and prices and incorrect imbalance forecast. Nowaday its also a big issue having too much negative prices for renewables in europe due to much more generation. 
```python
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
```
# TASK 2.4: # total daily renewable production and average day-ahead price
I just made a new column saying total renewable production by adding wind and pv day ahead generation forecast values, resampled it to daily to get daily renewable production daily and daily and mean day ahead price. Found highest and lowest production days with idmax and idmin corelating price using loc. 
The results showed highest RE production day had ver less price like 22EUR per MWh and lowest day had 223 something. It is simple rule of supply and demand where production and access and price go lower and vice versa. 
```python
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
```
# TASK 2.5
I calculated the average day-ahead price for weekdays and weekends. I added a column to mark weekends and then calculated the average price for both weekday and weekend periods. Typically, prices are higher on weekdays due to increased energy demand from businesses, while weekends have lower demand, resulting in lower prices.
```python
df['is_weekend'] = df.index.weekday >= 5  

weekday_avg_price = df.loc[~df['is_weekend'], 'Day Ahead Price hourly [in EUR/MWh]'].mean()
weekend_avg_price = df.loc[df['is_weekend'], 'Day Ahead Price hourly [in EUR/MWh]'].mean()

print(f"Average hourly day-ahead price during weekdays: {weekday_avg_price:.2f} EUR/MWh")
print(f"Average hourly day-ahead price during weekends: {weekend_avg_price:.2f} EUR/MWh")
```
# TASK 2.6
Its just finding daily min and max prices by resampling it to D to get the idea when i will charge battery while price is low and when in the day i will discharge while price is high. To get revenue, just took the difference and sum all of it. 
```python
daily_min_price = df['Day Ahead Price hourly [in EUR/MWh]'].resample('D').min()
daily_max_price = df['Day Ahead Price hourly [in EUR/MWh]'].resample('D').max()

daily_revenue = daily_max_price - daily_min_price
total_revenue = daily_revenue.sum()
print(f"Total revenue for the battery in 2021: {total_revenue:.2f} EUR")
```
