# BTC_price_with_SA

Do you own bitcoin (BTC)? Do you want to know the trend of BTC price in the near future? 

In this project, I have implemented the best model in the paper [Predicting the Price of Bitcoin Using Sentiment-Enriched Time Series Forecasting](https://www.mdpi.com/2504-2289/7/3/137)

# Data Processing

![image](https://drive.google.com/uc?export=view&id=1ekxoaxDdCWZqLqvlO43RfhvZ0u5ptbdA "BTC price data pipeline and how to feed it to forecasting model")

## Twitter dataset

Data files of tweets are downloaded from [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets). This dataset spans from 2021-02-05 to 2023-03-05 with some gaps.

If you want to get more updated and more "beautiful" tweets data from X, you can visit [X Developer Portal](https://developer.x.com/en/docs/x-api/getting-started/about-x-api) and pay for its API.

## Bitcoin historical price

In this project, BTC historical price is get from Binance.com by API keys. You can get [API keys](https://www.binance.com/en/binance-api) for free if you have an Binance account.

There is an easier way to get BTC price without an account and any API keys. That is using yfinance library to get price from Yahoo Finance. But yfinance only provides price with minimal interval is 1 hour. And in my experience, yfinance runs unstably!

## Text preprocessing

There are some steps in text preprocessing:

1. Replace the abbreviations
2. Remove http/https links, tagged accounts, extra spaces and special characters
3. Lower case all the tweets
4. Lemmatize and stemming
5. Remove stop words

After text preprocessing steps, cleaned tweets dataset is input for Sentiment Analysis.

## Sentiment Analysis

VADER Sentiment Analysis (VADER SA) is used to obtain sentiment score of each tweet.  
VADER SA is a rule based method which is builded from sentiment of each word in document and some heuristics (Punctuation, Capitalization, Degree modifiers...).
For each piece of text, VADER provides 4 sentiment scores: 

- <em>possitive, negative, neutral</em> - valence scores, corresponding to sentiment polarity with intensity.
- <em>compound</em> score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive)

## Data Aggregation

The next step is aggregating the sentiment of all tweets from each interval (30-min) into a single sentiment score corresponding to that time period.
The overall sentiment score of each interval is weighted by number of followers as below:

$$
  \frac{\sum_{i=1}^{n}s(x_i)(\ln(w_i+1)+1)}{\sum_{i=1}^{n}\ln(w_i+1)+1}  
$$

where $s(x_i)$ is the sentiment score of a tweet $x_i$ in the current interval, $w_i$ is the number of followers the author of the tweet had at the time of creation.

## Data merging and splitting

Finally, the 2 tweet and historical price datasets are merged into a single dataset by matching their respective time steps.

The merged dataset is splitted into train, validation and test dataset with approximate percent are 80%, 10% and 10% (avoid splitting during a period with data gaps).

# Model

The experiments in the paper prove that simpler models, particularly <em>Linear Regression</em> models have the best performance, while the more complex models are overfitting. Therefore, Linear Regression models are implemented here.

16 past BTC price values and sentiment scores are fed into the forcasting models. They will generate forecasts price for the future 8 time steps.

The models are evaluated by comparing BTC predicted values with the actual values in terms of MAE and RMSE metrics.

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
