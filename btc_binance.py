# 1. Use binance API to get historical price
# when data is farther than the last 7 days and interval-time<1 hour
# (API key is required).
# 2. Use yfinance with other cases.

import config
import csv
from binance.client import Client # pip install python-binance

client = Client(config.API_KEY, config.API_SECRET)

csvfile = open(config.BTC_FILE, 'w', newline='')
csvfile.write('Open_time,Open,High,Low,Close,Volume,Close_time,\
Quote_asset_volume,Number_of_trades,\
Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore\n')

candlestick_writer = csv.writer(csvfile, delimiter=',')

candlesticks = client.get_historical_klines(
    'BTCUSDT', Client.KLINE_INTERVAL_30MINUTE, config.START_TIME_HISTORY, config.END_TIME_HISTORY
)
for candlestick in candlesticks:
    # candlestick[0] = candlestick[0]/1000 # convert time returned in miliseconds to seconds
    candlestick_writer.writerow(candlestick)

csvfile.close()

