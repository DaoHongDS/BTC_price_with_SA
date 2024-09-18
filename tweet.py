import os.path
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import config
from text_process import replace_abbreviations, clean_text, remove_stopwords

pd.set_option('display.max_columns', None)

def read_tw_from_file(file_name):
    df = pd.read_csv(file_name, engine='python')

    # drop unuse columns
    columns_to_drop = ['user_name', 'user_location', 'user_description', \
                       'user_created', 'user_favourites', 'user_verified', \
                       'source' , 'hashtags', 'is_retweet']
    df.drop(columns=columns_to_drop, inplace=True)

    # 'date' column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['date'] = pd.to_datetime(df['date'])

    # 'text' column
    df['processed_text'] = df['text'].apply(replace_abbreviations)
    df['processed_text'] = clean_text(df['processed_text'])
    df['processed_text'] = df['processed_text'].apply(remove_stopwords)

    # number column
    df['user_followers'] = pd.to_numeric(df['user_followers'], errors='coerce')
    df['user_friends'] = pd.to_numeric(df['user_friends'], errors='coerce')

    return df

def VaderSA(df):
    analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(text):
        return analyzer.polarity_scores(text)

    df['VaderSA'] = df['processed_text'].apply(analyze_sentiment)
    # df['VaderNeg'] = df['VaderSA'].apply(lambda x: x['neg'])
    # df['VaderNeu'] = df['VaderSA'].apply(lambda x: x['neu'])
    # df['VaderPos'] = df['VaderSA'].apply(lambda x: x['pos'])
    df['VaderCmp'] = df['VaderSA'].apply(lambda x: x['compound'])

    return df

def weightByInfluence_up(saScore, influence, friend=0):
    return saScore * (np.log(influence + friend + 1) + 1)

def weightByInfluence_below(influence, friend=0):
    return np.log(influence + friend + 1) + 1

def weightByInfluence(df):
    # df['weighted_SA_up'] = df.apply(lambda x: weightByInfluence_up(x['VaderCmp'], x['user_followers'], x['user_friends']), axis=1)
    # df['weighted_SA_below'] = df.apply(lambda x: weightByInfluence_below(x['user_followers'], x['user_friends']), axis=1)

    df['weighted_SA_up'] = df.apply(lambda x: weightByInfluence_up(x['VaderCmp'], x['user_followers']), axis=1)
    df['weighted_SA_below'] = df.apply(lambda x: weightByInfluence_below(x['user_followers']), axis=1)

    sum_tweet_df = df.groupby(pd.Grouper(key='date', freq='30min'))\
        .agg({'weighted_SA_up': 'sum', 'weighted_SA_below': 'sum'})

    # sum_tweet_df = df.groupby(pd.Grouper(key='date', freq='1h'))\
    #     .agg({'weighted_SA_up': 'sum', 'weighted_SA_below': 'sum'})

    def cal_weight(up, down):
        if down==0:
            return 0
        else:
            return up/down
    sum_tweet_df['weighted_SA'] = sum_tweet_df.apply(lambda x: cal_weight(x['weighted_SA_up'], x['weighted_SA_below']), axis=1)

    return sum_tweet_df

if os.path.exists(config.SA_FILE):
    df_SA = pd.read_csv(config.SA_FILE, engine='python')
    df_SA['date'] = pd.to_datetime(df_SA['date'], errors='coerce')
    df_SA['weighted_SA'] = pd.to_numeric(df_SA['weighted_SA'], errors='coerce')
else:
    file_name_1 = 'data/Bitcoin_tweets.csv'
    file_name_2 = 'data/Bitcoin_tweets_dataset_2.csv'

    df = read_tw_from_file(file_name_2)
    df1 = read_tw_from_file(file_name_1)

    df = pd.concat([df, df1], ignore_index=True)
    df.reset_index(inplace=True)
    del df["index"]

    df = VaderSA(df)
    df_SA = weightByInfluence(df)
    df_SA.reset_index(inplace=True)
    df_SA.to_csv(config.SA_FILE, index=False)

df_BTC = pd.read_csv(config.BTC_FILE, engine='python')
df_BTC.rename(columns={'Open_time': 'date'}, inplace=True)
df_BTC['date'] = pd.to_datetime(df_BTC['date'], errors='coerce', unit='ms')
# df_BTC['date'] = df_BTC['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
# df_BTC['date'] = pd.to_datetime(df_BTC['date'])

df_BTC['Open'] = pd.to_numeric(df_BTC['Open'], errors='coerce')
df_BTC['Close'] = pd.to_numeric(df_BTC['Close'], errors='coerce')
df_BTC['Volume'] = pd.to_numeric(df_BTC['Volume'], errors='coerce')

######### combined dataframe ####################

df = df_SA.set_index('date').join(df_BTC.set_index('date'))
df.reset_index(inplace=True)
df = df[['date', 'Open', 'Close', 'Volume', 'weighted_SA']]
df.to_csv(config.BTC_SA_FILE, index=False)

print('combined df: \n', df.iloc[:5])

########### plot ###############

scaler = MinMaxScaler(feature_range=(-1,1))
x = df['Open'].values.reshape(-1,1)
x_scaled = scaler.fit_transform(x)
df['Open'] = x_scaled

scaler2 = MinMaxScaler(feature_range=(0,1))
x1 = df['weighted_SA'].values.reshape(-1,1)
x1_scaled = scaler2.fit_transform(x1)
df['weighted_SA'] = x1_scaled

plt.plot(df['date'], df['Open'], label='Open price')
plt.plot(df['date'], df['weighted_SA'], label='Weighted Vader compound SA')
plt.xticks(rotation = 90)

plt.show()

