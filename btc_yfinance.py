# note: yfinance runs unstably!

import yfinance as yf
import config
import pandas as pd
# from datetime import datetime

start_date = '2021-02-05'
end_date = '2023-03-06'

BTC_Ticker = yf.Ticker("BTC-USD")
BTC_Data = BTC_Ticker.history(start=start_date, end=end_date, interval='1h')
# print(BTC_Data.dtypes)
BTC_Data = BTC_Data.reset_index()
print(BTC_Data.dtypes)

BTC_Data['Datetime'] = pd.to_datetime(BTC_Data['Datetime'], errors='coerce')
BTC_Data['Datetime'] = BTC_Data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
BTC_Data['Datetime'] = pd.to_datetime(BTC_Data['Datetime'])

BTC_Data = BTC_Data[['Datetime', 'Open', 'Close', 'Volume']]
# print(BTC_Data.dtypes)

with open(config.BTC_FILE, 'w', encoding='utf-8-sig') as f:
    BTC_Data.to_csv(f, index=False)
