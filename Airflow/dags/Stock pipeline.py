from github import Github
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models.connection import Connection
import pandas as pd
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
import requests
import pandas as pd 
import numpy as np
import boto3
from airflow.hooks.base import BaseHook
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import io 
from xgboost import XGBRegressor
from io import StringIO

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': '2025-06-07',
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}

c = Connection(
    conn_id='postgres',
    conn_type='Postgres',
    login='airflow',
    password='airflow'
)

def update_data():
    hook = PostgresHook(postgres_conn_id="postgres")

    sql_ts = """ SELECT * FROM public."acb_stock" """
    df_old = hook.get_pandas_df(sql_ts)

    date  = datetime.today().strftime('%Y-%m-%d')

    close = []
    open = []
    high = []
    low = []
    volume = []
    html = requests.get(f'http://en.stockbiz.vn/Stocks/ACB/LookupQuote.aspx?date={date}').text
    soup = BeautifulSoup(html)
    table = soup.find('table')
    div1 = table.find('div',{'id':'mainwrap'})
    div2 = div1.find('div',{'id':'container'})
    table1 = div1.find('table',{'id':'ctl00_PlaceHolderContentArea_CenterZone'})
    tr1 = table1.find_all('tr',{'class':'rowcolor2'})
    tr2 = table1.find_all('tr',{'class':'rowcolor1'})
    
    for row in tr1:
        stock_price = row.find_all('td',attrs={'align':'right'})[4].text
        close.append(stock_price.strip())
    for row in tr2:
        stock_price = row.find_all('td',attrs={'align':'right'})[4].text
        close.append(stock_price.strip())
    close.append(stock_price.strip())

    for row in tr1:
        stock_price = row.find_all('td',attrs={'align':'right'})[1].text
        open.append(stock_price.strip())
    for row in tr2:
        stock_price = row.find_all('td',attrs={'align':'right'})[1].text
        open.append(stock_price.strip())
    open.append(stock_price.strip())

    for row in tr1:
        stock_price = row.find_all('td',attrs={'align':'right'})[2].text
        high.append(stock_price.strip())
    for row in tr2:
        stock_price = row.find_all('td',attrs={'align':'right'})[2].text
        high.append(stock_price.strip())
    high.append(stock_price.strip())

    for row in tr1:
        stock_price = row.find_all('td',attrs={'align':'right'})[3].text
        low.append(stock_price.strip())
    for row in tr2:
        stock_price = row.find_all('td',attrs={'align':'right'})[3].text
        low.append(stock_price.strip())
    low.append(stock_price.strip())

    for row in tr1:
        stock_price = row.find_all('td',attrs={'align':'right'})[7].text
        volume.append(stock_price.strip())
    for row in tr2:
        stock_price = row.find_all('td',attrs={'align':'right'})[7].text
        volume.append(stock_price.strip())
    volume.append(stock_price.strip())
    volume[0] = volume[0].replace(',', '')
    df_latest = pd.DataFrame({'OPEN':[open[0]],'HIGH':high[0],'LOW':low[0],'close':[close[0]],'VOLUME':volume[0]})
    df_latest['OPEN'] = df_latest['OPEN'].astype(float).apply(lambda x: x*1000)
    df_latest['HIGH'] = df_latest['HIGH'].astype(float).apply(lambda x: x*1000)
    df_latest['LOW'] = df_latest['LOW'].astype(float).apply(lambda x: x*1000)
    df_latest['close'] = df_latest['close'].astype(float).apply(lambda x: x*1000)
    df_latest['VOLUME'] = df_latest['VOLUME'].astype(float)

    conn = BaseHook.get_connection('postgres')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')

    #df_old = df_old.drop(columns=['change'])
    df_old = df_old.sort_values(by='date')
    df_old.set_index("date",inplace=True)

    df_latest.rename(index={0:date},inplace=True)
    df_latest = df_latest.rename(columns={'HIGH': 'high', 'LOW': 'low','close':'close','OPEN':'open','VOLUME':'volume'})
    df_latest['volume'] = df_latest['volume']/1000000

    df_full = pd.concat([df_old, df_latest])
    df_full.reset_index(names="date",inplace=True)

    if set(df_full[['close','open','high','low','volume']].iloc[-2]) == set(df_full[['close','open','high','low','volume']].iloc[-1]):
        print('asdeqfwe')
    else:
        df_full.to_sql(f'acb_stock', engine, if_exists='replace', index=False)

        csv_buffer = StringIO()
        df_full.to_csv(csv_buffer)

        s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='4U247Rhn8cRGtTgiiJUg',
                        aws_secret_access_key='58SHIao9tx0bQNjaS2MyU8rNZdOwUhROsv4yiNyP')
        s3.Object('mlflow-artifacts', 'data_copy.csv').put(Body=csv_buffer.getvalue())

def prediction():
    hook = PostgresHook(postgres_conn_id="postgres")

    sql_ts = """ SELECT * FROM public."acb_stock" """
    df = hook.get_pandas_df(sql_ts)

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
    df_all = df
    df = df.iloc[[-1]]
    input = df.drop(columns=['close_t+1'])

    s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='4U247Rhn8cRGtTgiiJUg',
                        aws_secret_access_key='58SHIao9tx0bQNjaS2MyU8rNZdOwUhROsv4yiNyP'
    )

    temp_model_location = './temp_model.pkl'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(s3.Bucket("mlflow-artifacts").Object("1/232bdcf2cc164c76bade410782fd9a48/artifacts/testmodel/model.xgb").get()['Body'].read())
    temp_model_file.close()
    model = XGBRegressor(enable_categorical=True)
    model.load_model(temp_model_location)

    input = input.drop(columns=['date'])
    pred = model.predict(input)

    df_close = df_all[['date','close']]
    df_close['date'] = pd.to_datetime(df_close['date'],format= '%Y-%m-%d')
    print(df_close)

    date=datetime.today().strftime('%Y-%m-%d')
    date_after = pd.to_datetime(date) + timedelta(days=1)

    df_pred = pd.DataFrame([[date_after,round(pred[0],0)]], columns=['date','close'])
    df_pred['date'] = pd.to_datetime(df_pred['date'],format= '%Y-%m-%d')
    print(df_pred)
    df_concat = pd.concat([df_close.iloc[[-1]], df_pred], ignore_index=True)

    b = df_close
    c = df_concat

    fig, ax = plt.subplots()
    ax.plot(b['date'].iloc[-3:],b['close'].iloc[-3:],color="gray", label = 'History price')
    ax = plt.subplot(111)
    ax.plot(c['date'],c['close'],color="red", label = 'Predicted price')
    plt.title('ACB stock prediction', fontdict={'fontsize':20})
    ax.set_xlabel('date')
    ax.set_ylabel('Stock price (VND)')
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    x = c['date']
    y = c['close']
    values  =c['close'].to_list()
    for i, txt in enumerate(values):
        ax.annotate(txt, (x[i], y[i]))
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg")

    temp_model_location = './plot.jpg'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(buf.getvalue())
    temp_model_file.close()
    buf.close()
    plt.show()

    s3.meta.client.upload_file(temp_model_location, 'mlflow-artifacts', 'plot.jpg')

    content = s3.Bucket("mlflow-artifacts").Object("/plot.jpg").get()['Body'].read()

    g = Github("")

    repo = g.get_user().get_repo('StockPredict.github.io')
    all_files = []
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file = file_content
            all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))

    # Upload to github
    git_prefix = 'images/thumbs/'
    git_file = git_prefix + '01.jpg'
    if git_file in all_files:
        contents = repo.get_contents(git_file)
        repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
        print(git_file + ' UPDATED')
    else:
        repo.create_file(git_file, "committing files", content, branch="main")
        print(git_file + ' CREATED')

    all_files = []
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file = file_content
            all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))

    # Upload to github
    git_prefix = 'images/fulls/'
    git_file = git_prefix + '01.jpg'
    if git_file in all_files:
        contents = repo.get_contents(git_file)
        repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
        print(git_file + ' UPDATED')
    else:
        repo.create_file(git_file, "committing files", content, branch="main")
        print(git_file + ' CREATED')

with DAG(
    dag_id="Stock_prediction",
    default_args=default_args,
    schedule='30 13 * * *'
) as dag:
    task_1 =PythonOperator(
        task_id = "update_data",
        python_callable=update_data
    )
    task_2 =PythonOperator(
        task_id = "prediction",
        python_callable=prediction
    )
    task_1>>task_2