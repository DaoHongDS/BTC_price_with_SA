# BTC_price_with_SA

Do you own bitcoin (BTC)? Do you want to know the trend of BTC price in near future? I do.

And in this project, I have implemented the best model in the paper [Predicting the Price of Bitcoin Using Sentiment-Enriched Time Series Forecasting](https://www.mdpi.com/2504-2289/7/3/137)

# Data Processing

![image](https://drive.google.com/uc?export=view&id=1ekxoaxDdCWZqLqvlO43RfhvZ0u5ptbdA "BTC price data pipeline and how to feed it to forecasting model")

## Twitter dataset

Data files of tweets are downloaded from [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets).
If you want to get more updated and "beautiful" tweets data from X, you can visit [X Developer Portal](https://developer.x.com/en/docs/x-api/getting-started/about-x-api) and pay for its API.

## Bitcoin historical price

In this project, BTC historical price is get from Binance.com by API keys. You can get [API keys](https://www.binance.com/en/binance-api) for free if you have an Binance account.

There is an easier way to get BTC price without account and API keys. That is using yfinance library to get price from Yahoo Finance. But yfinance only provides price with minimal interval is 1 hour. And in my experience, yfinance runs unstably!

## Text preprocessing

## Sentiment Analysis

## Data Aggregation

## Data merging and splitting


# Model

# Source code

1. Clone the project
2. Download data files of tweets from [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)
3. Put data files of tweets in the data folder
```bash
├── data
│   ├── Bitcoin_tweets.csv
│   ├── Bitcoin_tweets_dataset_2.csv
├── models
├── btc_binance.py
├── btc_yfinance.py
├── config.py
├── model.py
├── text_process.py
├── tweet.py
```
5. Log in to Binance.com to create [API keys](https://www.binance.com/en/binance-api) (If you have not got an Binance account yet, you should sign up to have one).
6. Insert your API keys into config.py file.
7. Run btc_binance.py file to get historical price (in USD) of BTC 
8.  Run tweet.py file to process the tweet data (clean text and calculate sentiment score) and to merge the price data set with the sentiment score by time steps. We have a single dataset to feed into predict models.
9.  Run model.py file to fit model.
