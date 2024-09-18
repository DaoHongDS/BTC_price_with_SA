import config
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Normalization, Dense, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

df = pd.read_csv(config.BTC_SA_FILE, engine='python')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

df_train = df[(df['date']<=config.END_TIME_TRAIN) & \
              (df['weighted_SA']!=0.0) & \
              (df['Close'].notna())]
df_val = df[(df['date']>=config.START_TIME_VAL) & (df['date']<=config.END_TIME_VAL) & \
            (df['weighted_SA']!=0.0) & \
            (df['Close'].notna())]
df_test = df[(df['date']>=config.START_TIME_TEST)& \
             (df['weighted_SA']!=0.0) & \
             (df['Close'].notna())]

np_train = np.array(df_train[['Close', 'weighted_SA']])
np_val = np.array(df_val[['Close', 'weighted_SA']])
np_test = np.array(df_test[['Close', 'weighted_SA']])

# scaler = StandardScaler()
# scaler.fit(np_train)
# train_data = scaler.transform(np_train)
# val_data = scaler.transform(np_val)
# test_data = scaler.transform(np_test)

def windowed_dataset(series, input_series_size, n_next_interval):
    X, y = [], []
    for i in range(len(series) - input_series_size):
        X.append(series[i:i+input_series_size])
        y.append(series[i+input_series_size+n_next_interval,0]) # take only close price
    return np.array(X), np.array(y)

# def windowed_dataset(series, input_series_size, n_next_interval, shuffle_buffer, batch_size):
#     # series = tf.expand_dims(series, axis=-1)
#     ds = tf.data.Dataset.from_tensor_slices(series)
#
#     tot_size = input_series_size+n_next_interval+1
#     ds = ds.window(tot_size, shift=1, drop_remainder=True)
#     ds = ds.flat_map(lambda w: w.batch(tot_size))
#     # ds = ds.shuffle(shuffle_buffer)
#
#     ds = ds.map(lambda w: (w[:input_series_size], w[input_series_size+n_next_interval]))
#     ds = ds.batch(batch_size).prefetch(1)
#
#     return ds

# train_data = windowed_dataset(np_train, 16, 0, 1000, 32)
# val_data = windowed_dataset(np_val, 16, 0, 1000, 32)
# # print('first sample to train \n', list(train_data.take(1)))


X_train, y_train = windowed_dataset(np_train, 16, 0)
X_val, y_val = windowed_dataset(np_val, 16, 0)
X_test, y_test = windowed_dataset(np_test, 16, 0)
# print('X_train shape',X_train.shape)
# print('y_train shape', y_train.shape)
# print('X_val shape',X_val.shape)
# print('y_val shape', y_val.shape)

normalizer = Normalization(axis=-1)
normalizer.adapt(X_train)

model = Sequential([
    Input(shape=(16, 2)),
    normalizer,
    Flatten(),
    Dense(1),
])

# model = Sequential([
#     Input(shape=(16, 2)),
#     normalizer,
#     LSTM(32, input_shape=(16,2)),
#     Dense(1)
# ])


model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.5),
              loss = tf.keras.losses.MeanAbsoluteError(),
              metrics=['mae', 'root_mean_squared_error'])

ckpt = ModelCheckpoint('./models/model.keras',
                       monitor='val_mae',
                       save_best_only=True
                       )

early_stop = EarlyStopping(monitor='val_mae',
                           patience=5,
                           )

history = model.fit(X_train, y_train, batch_size=32, epochs=300, verbose=1,
                    callbacks=[ckpt, early_stop],
                    validation_data=(X_val, y_val))

# inverse transform
# X = scaler2.inverse_transform(X_train)

def plot_loss(h):
  plt.plot(h.history['loss'], label='loss')
  plt.plot(h.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Close]')
  plt.legend()
  plt.grid(True)
  plt.show()



print('history', history.history.keys())
plot_loss(history)

results = model.evaluate(X_test, y_test, batch_size=32)
print("test loss, test acc:", results)