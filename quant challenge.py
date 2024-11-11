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

#TASK 1.1
# Defining functions to compute total buy and sell volumes
def compute_total_buy_volume(data) -> float:
    return data[data['side'] == 'buy']['quantity'].sum()

def compute_total_sell_volume(data) -> float:
    return data[data['side'] == 'sell']['quantity'].sum()

#buy and sell volumes
total_buy_volume = compute_total_buy_volume(data)
total_sell_volume = compute_total_sell_volume(data)

print(f"Total Buy Volume: {total_buy_volume}")
print(f"Total Sell Volume: {total_sell_volume}")

#TASK 1.2
#PnL (Profit and Loss) for each strategy
def compute_pnl(strategy_id: str, data) -> float:
    # Filter data for the specified strategy
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

