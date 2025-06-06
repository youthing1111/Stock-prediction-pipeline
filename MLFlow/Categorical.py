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

#logging.basicConfig(level=logging.WARN)
#logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    return accuracy

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

# Read the file from the URL
excel_url = (
    "/Users/trieuhoanghiep/Documents/Me/Project/VN STOCK/Simplize_ACB_PriceHistory_20240526.xlsx"
)
df = pd.read_excel(excel_url)
# Calculate the 5-day simple moving average
sma = df['CLOSE'].rolling(9).mean()

# Print the DataFrame containing the closing prices and the SMA
df = df.join(sma, rsuffix='_sma')
# Calculate the 3-day weighted moving average
weights = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
wma = df['CLOSE'].rolling(9).apply(lambda x: sum((weights*x)) / sum(weights), raw=True)

# Print the DataFrame containing the sales and the WMA
df = df.join(wma, rsuffix='_wma')
# Calculate the 3-day exponential moving average
ema12 = df['CLOSE'].ewm(span=12, adjust=False).mean()
ema26 = df['CLOSE'].ewm(span=26, adjust=False).mean()
# Print the DataFrame containing the closing prices and the EMA
df = df.join(ema12, rsuffix='_ema12')
df = df.join(ema26, rsuffix='_ema26')
df['MACD']  = df['CLOSE_ema12'] - df['CLOSE_ema26']
def get_stochastic_oscillator(df, period=14):
    for i in range(len(df)):
        low = df.iloc[i]['CLOSE']
        high = df.iloc[i]['CLOSE']
        if i >= period:
            n = 0
            while n < period:
                if df.iloc[i-n]['CLOSE'] >= high:
                    high = df.iloc[i-n]['CLOSE']
                elif df.iloc[i-n]['CLOSE'] < low:
                    low = df.iloc[i-n]['CLOSE']
                n += 1
            df.at[i, 'best_low'] = low
            df.at[i, 'best_high'] = high
            df.at[i, 'fast_k'] = 100*((df.iloc[i]['CLOSE']-df.iloc[i]['best_low'])/(df.iloc[i]['best_high']-df.iloc[i]['best_low']))

    df['fast_d'] = df['fast_k'].rolling(3).mean().round(2)
    df['slow_k'] = df['fast_d']
    df['slow_d'] = df['slow_k'].rolling(3).mean().round(2)

    return df
get_stochastic_oscillator(df)
df['LW'] = ((df['HIGH']-df['CLOSE'])/(df['HIGH']-df['LOW']))*100
df['CLOSE_t+1'] = df['CLOSE'].diff(1)
df['CLOSE_t+1'] = np.nan
df['CLOSE_t+1'].iloc[:-1] = df['CLOSE'].iloc[1:]
def f(df):
    if df['CLOSE_t+1'] > df['CLOSE']:
        val = 1
    elif df['CLOSE_t+1'] < df['CLOSE']:
        val = 0
    else:
        val = np.nan
    return val
df['Target'] = df.apply(f, axis=1)
def acc_dist(data, trend_periods=9, open_col='OPEN', high_col='HIGH', low_col='LOW', close_col='CLOSE', vol_col='VOLUME'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        df.at[index,'acc_dist'] = ac
    data['acc_dist_ema' + str(trend_periods)] = data['acc_dist'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    
    return data
acc_dist(df)
def on_balance_volume(data, trend_periods=9, close_col='CLOSE', vol_col='VOLUME'):
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
def price_volume_trend(data, trend_periods=9, close_col='CLOSE', vol_col='VOLUME'):
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
def typical_price(data, high_col = 'HIGH', low_col = 'LOW', close_col = 'CLOSE'):
    
    data['typical_price'] = (data[high_col] + data[low_col] + data[close_col]) / 3

    return data
typical_price(df)
df_1 = df.dropna()
df_2 = df_1.drop(columns='CLOSE_t+1')
df_2.set_index('DATE',inplace = True)
df_2 = df_2.iloc[-750:]

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(df_2)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["Target"], axis=1)
test_x = test.drop(["Target"], axis=1)
train_y = train[["Target"]]
test_y = test[["Target"]]


max_depth=6
max_features='sqrt'
mlflow.set_tracking_uri("http://0.0.0.0:5005")
mlflow.set_experiment("my-test-exp")
# Useful for multiple runs (only doing one run in this sample notebook)
with mlflow.start_run():
    # Execute ElasticNet
    lr = DecisionTreeClassifier(random_state=42,max_depth=max_depth,max_features=max_features)
    lr.fit(train_x, train_y)

    # Evaluate Metrics
    predicted_qualities = lr.predict(test_x)
    acc = eval_metrics(test_y, predicted_qualities)

    # Print out metrics
    print(f"Decision Tree model:")
    print("  Accuracy: %s" % acc)

    # Infer model signature
    predictions = lr.predict(train_x)
    signature = infer_signature(train_x, predictions)

    # Log parameter, metrics, and model to MLflow
    mlflow.log_metric("Accuracy", acc)
    mlflow.sklearn.log_model(lr,artifact_path='testmodel')