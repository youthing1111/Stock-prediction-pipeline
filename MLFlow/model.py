from airflow.hooks.base import BaseHook
from sqlalchemy import create_engine
import pandas as pd
import logging
#import sys
import warnings
import boto3
#from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

conn = BaseHook.get_connection('postgres')
engine = create_engine(f'postgresql://airflow:airflow@localhost:5434/stock')

with engine.connect() as conn:
    df = pd.read_sql(
        sql="SELECT * FROM acb_stock",
        con=conn.connection
    )

import numpy as np

sma = df['close'].rolling(9).mean()
df = df.join(sma, rsuffix='_sma')
weights = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
wma = df['close'].rolling(9).apply(lambda x: sum((weights*x)) / sum(weights), raw=True)
df = df.join(wma, rsuffix='_wma')
ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
df = df.join(ema12, rsuffix='_ema12')
df = df.join(ema26, rsuffix='_ema26')
df['MACD']  = df['close_ema12'] - df['close_ema26']
def get_stochastic_oscillator(df, period=14):
    for i in range(len(df)):
        low = df.iloc[i]['close']
        high = df.iloc[i]['close']
        if i >= period:
            n = 0
            while n < period:
                if df.iloc[i-n]['close'] >= high:
                    high = df.iloc[i-n]['close']
                elif df.iloc[i-n]['close'] < low:
                    low = df.iloc[i-n]['close']
                n += 1
            df.at[i, 'best_low'] = low
            df.at[i, 'best_high'] = high
            df.at[i, 'fast_k'] = 100*((df.iloc[i]['close']-df.iloc[i]['best_low'])/(df.iloc[i]['best_high']-df.iloc[i]['best_low']))

    df['fast_d'] = df['fast_k'].rolling(3).mean().round(2)
    df['slow_k'] = df['fast_d']
    df['slow_d'] = df['slow_k'].rolling(3).mean().round(2)

    return df
get_stochastic_oscillator(df)
df['LW'] = ((df['high']-df['close'])/(df['high']-df['low']))*100
df['close_t+1'] = df['close'].diff(1)
df['close_t+1'] = np.nan
df['close_t+1'].iloc[:-1] = df['close'].iloc[1:]
def f(df):
    if df['close_t+1'] > df['close']:
        val = 1
    elif df['close_t+1'] < df['close']:
        val = 0
    else:
        val = np.nan
    return val
df['Target'] = df.apply(f, axis=1)
def acc_dist(data, trend_periods=9, open_col='OPEN', high_col='high', low_col='low', close_col='close', vol_col='volume'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        df.at[index,'acc_dist'] = ac
    data['acc_dist_ema' + str(trend_periods)] = data['acc_dist'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    return data
acc_dist(df)
def on_balance_volume(data, trend_periods=9, close_col='close', vol_col='volume'):
    for index, row in data.iterrows():
        if index > 0:
            last_obv = data.at[index - 1, 'obv']
            if row[close_col] > data.at[index - 1, close_col]:
                current_obv = last_obv + row[vol_col]
            elif row[close_col] < data.at[index - 1, close_col]:
                current_obv = last_obv - row[vol_col]
            else:
                current_obv = last_obv
        else:
            last_obv = 0
            current_obv = row[vol_col]

        df.at[index,'obv'] = current_obv
    data['obv_ema' + str(trend_periods)] = data['obv'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    return data
on_balance_volume(df)
def price_volume_trend(data, trend_periods=9, close_col='close', vol_col='volume'):
    for index, row in data.iterrows():
        if index > 0:
            last_val = data.at[index - 1, 'pvt']
            last_close = data.at[index - 1, close_col]
            today_close = row[close_col]
            today_vol = row[vol_col]
            current_val = last_val + (today_vol * (today_close - last_close) / last_close)
        else:
            current_val = row[vol_col]

        df.at[index,'pvt'] = current_val
    data['pvt_ema' + str(trend_periods)] = data['pvt'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    return data
price_volume_trend(df)
def typical_price(data, high_col = 'high', low_col = 'low', close_col = 'close'):
    data['typical_price'] = (data[high_col] + data[low_col] + data[close_col]) / 3
    return data
typical_price(df)

df_nonna = df.dropna()
df_nonna = df_nonna.drop(columns=['date'])
from sklearn.model_selection import train_test_split
X = df_nonna.drop(columns=['close_t+1'])
y = df_nonna['close_t+1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from xgboost import XGBRegressor

mlflow.set_tracking_uri("http://localhost:5005")
mlflow.set_experiment("my-test-exp")
# Useful for multiple runs (only doing one run in this sample notebook)
with mlflow.start_run():
    model = XGBRegressor(enable_categorical=True)
    model.fit(X_train, y_train)

    predict_test = model.predict(X_test)
    predict_train = model.predict(X_train)  

    from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
    mape_test = mean_absolute_percentage_error(y_test.values,predict_test)
    
    signature = infer_signature(predict_train, predict_test)

    # Log parameter, metrics, and model to MLflow
    mlflow.log_metric("MAPE", mape_test)
    mlflow.xgboost.log_model(model,artifact_path='testmodel')