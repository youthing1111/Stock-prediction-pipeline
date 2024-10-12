from github import Github
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models.connection import Connection
from airflow.hooks.base_hook import BaseHook
import pandas as pd
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
import requests
import pandas as pd 
import numpy as np
import mlflow
import boto3
from airflow.hooks.base import BaseHook
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import io 
from xgboost import XGBClassifier, XGBRegressor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}

mlflow.set_tracking_uri("http://0.0.0.0:5005")

c = Connection(
    conn_id='postgres_server_2',
    conn_type='Postgres',
    login='airflow',
    password='airflow'
)

#extract tasks
def get_src_tables(**context):
    hook = PostgresHook(postgres_conn_id="postgres_server_2")
    #sql = """ SELECT * FROM stock_categorical """
    #df = hook.get_pandas_df(sql)

    sql_ts = """ SELECT * FROM public."stock_time_series" """
    df_ts = hook.get_pandas_df(sql_ts)
    context['ti'].xcom_push(key="old data ts", value=df_ts)
    

def scrap_stock(**context):
    date  = datetime.today().strftime('%m/%d/%Y')
    context['ti'].xcom_push(key="current_date", value=date)
    close = []
    open = []
    high = []
    low = []
    volume = []
    html = requests.get(f'http://en.stockbiz.vn/Stocks/ACB/LookupQuote.aspx?Date={date}').text
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
    df_latest = pd.DataFrame({'OPEN':[open[0]],'HIGH':high[0],'LOW':low[0],'CLOSE':[close[0]],'VOLUME':volume[0]})
    df_latest['OPEN'] = df_latest['OPEN'].astype(float).apply(lambda x: x*1000)
    df_latest['HIGH'] = df_latest['HIGH'].astype(float).apply(lambda x: x*1000)
    df_latest['LOW'] = df_latest['LOW'].astype(float).apply(lambda x: x*1000)
    df_latest['CLOSE'] = df_latest['CLOSE'].astype(float).apply(lambda x: x*1000)
    df_latest['VOLUME'] = df_latest['VOLUME'].astype(float)
    context['ti'].xcom_push(key="new data", value=df_latest)

def concat_old_data(**context):
    conn = BaseHook.get_connection('postgres_server_2')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')
    df_old = pd.read_sql_query('SELECT * FROM stock_categorical ', engine)
    df_old.set_index("DATE",inplace=True)
    print(df_old)

    date=context['ti'].xcom_pull(key="current_date")
    df_new = context['ti'].xcom_pull(key="new data")
    df_new.rename(index={0:date},inplace=True)
    df_full = pd.concat([df_old, df_new])
    df_full.reset_index(names="DATE",inplace=True)
    df_full.drop_duplicates(inplace=True,keep='last')
    df_full.to_sql(f'stock_categorical', engine, if_exists='replace', index=False)
    df_full.drop(columns='DATE',inplace=True)
    context['ti'].xcom_push(key="full data", value=df_full)

def transform_data(**context):
    df = context['ti'].xcom_pull(key="full data")
    sma = df['CLOSE'].rolling(9).mean()
    df = df.join(sma, rsuffix='_sma')
    weights = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    wma = df['CLOSE'].rolling(9).apply(lambda x: sum((weights*x)) / sum(weights), raw=True)
    df = df.join(wma, rsuffix='_wma')
    ema12 = df['CLOSE'].ewm(span=12, adjust=False).mean()
    ema26 = df['CLOSE'].ewm(span=26, adjust=False).mean()
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
    input = df.iloc[[-1]]
    context['ti'].xcom_push(key="input_monitoring", value=input)

    input.drop(columns=["Target","CLOSE_t+1"],inplace=True)
    context['ti'].xcom_push(key="input", value=input)

    input['CLOSE_diff1'] = input['CLOSE'].diff(1)
    context['ti'].xcom_push(key="input_ts", value=input)

def pred_Categorical(**context):
    s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='',
                        aws_secret_access_key=''
    )

    temp_model_location = './temp_model.xgb'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(s3.Bucket("mlflow-artifacts").Object("7/2ddf3e31f2304446a02d124067d40814/artifacts/XGBoost/model.xgb").get()['Body'].read())
    temp_model_file.close()
    model = XGBClassifier()
    model.load_model(temp_model_location)

    input = context['ti'].xcom_pull(key="input")
    pred = model.predict(input).tolist()
    context['ti'].xcom_push(key="pred_categorical", value=pred)

def pred_time_series(**context):
    df_old_ts = context['ti'].xcom_pull(key="old data ts")
    #df_old_ts['CLOSE']= df_old_ts['CLOSE'].apply(lambda x: x.replace(',','.'))
    df_old_ts['CLOSE'] = df_old_ts['CLOSE'].astype(float)

    s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='',
                        aws_secret_access_key=''
    )
    df_new = context['ti'].xcom_pull(key="new data")

    df_new_ts = df_new[['CLOSE']]
    date=context['ti'].xcom_pull(key="current_date")
    df_new_ts.rename(index={0:date},inplace=True)

    df_old_ts.set_index("DATE",inplace=True)
    df_full_ts = pd.concat([df_old_ts, df_new_ts])
    print(df_full_ts.index)
    df_full_ts.index = pd.to_datetime(df_full_ts.index,format='mixed').date

    conn = BaseHook.get_connection('postgres_server_2')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')
    df_full_ts.drop_duplicates(inplace=True,keep='last')
    df_full_ts.reset_index(names='DATE',inplace=True)
    df_full_ts.to_sql(f'stock_time_series', engine, if_exists='replace', index=False)

    context['ti'].xcom_push(key="df_full_ts", value=df_full_ts)
    
    temp_model_location_2 = './temp_model_2.xgb'
    temp_model_file = open(temp_model_location_2, 'wb')
    temp_model_file.write(s3.Bucket("mlflow-artifacts").Object("6/dea4644da47c4c329304885e4e477aac/artifacts/Time-series/model.xgb").get()['Body'].read())
    temp_model_file.close()
    model = XGBRegressor()
    model.load_model(temp_model_location_2)

    input = context['ti'].xcom_pull(key="input_ts")
    pred_ts = model.predict(input)

    yhat = pd.DataFrame(pred_ts)
    yhat.columns = ['predicted_index']
    print(yhat)
    context['ti'].xcom_push(key="pred_time_series", value=yhat)

def plot(**context):
    df_full_ts = context['ti'].xcom_pull(key="df_full_ts")
    print(df_full_ts)
    last_value = df_full_ts['CLOSE'].iloc[-1]
    new_df = pd.DataFrame({"pred_index":[last_value]}) 

    yhat = context['ti'].xcom_pull(key="pred_time_series")
    trans_value = last_value+yhat
    trans_value.rename(columns={"predicted_index":"pred_index"},inplace=True)
    trans_value['pred_index'] = trans_value['pred_index'].astype(int)
    df_ts_pred = pd.concat([new_df, trans_value], ignore_index=True)


    date=datetime.today().strftime('%Y-%m-%d')
    print(date)
    df_ts_pred.rename(index={0:date},inplace=True)

    date_after = pd.to_datetime(date) + timedelta(days=1)
    print(date_after)
    df_ts_pred.rename(index={1:date_after},inplace=True)
    print(df_ts_pred)
    df_ts_pred.index = pd.to_datetime(df_ts_pred.index).date
    print(df_ts_pred)

    #b = df_full_ts.reset_index(names="date")
    b = df_full_ts[['CLOSE',"DATE"]]
    print(b)

    c = df_ts_pred.reset_index(names="date")
    c = c[['pred_index',"date"]]
    print(c)

    fig, ax = plt.subplots()
    ax.plot(b['DATE'].iloc[-5:],b['CLOSE'].iloc[-5:],color="gray", label = 'History price')
    ax = plt.subplot(111)
    ax.plot(c['date'],c['pred_index'],color="red", label = 'Predicted price')
    plt.title('ACB stock prediction', fontdict={'fontsize':20})
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock price (VND)')
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    x = c['date']
    y = c['pred_index']
    values  =c['pred_index'].to_list()
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

    s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='',
                        aws_secret_access_key=''
    )
    s3.meta.client.upload_file(temp_model_location, 'mlflow-artifacts', 'plot.jpg')

def plot_github(**context):
    s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='',
                        aws_secret_access_key=''
    )
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

def monitoring(**context):
    conn = BaseHook.get_connection('postgres_server_2')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')

    df_old_clf = pd.read_sql_query('SELECT * FROM monitoring_clf ', engine)
    prediction_clf = context['ti'].xcom_pull(key="pred_categorical")
    date=datetime.today().strftime('%m-%d-%Y')

    input_clf = context['ti'].xcom_pull(key="input_monitoring")
    if date != df_old_clf.iloc[-1:].index.values[0]:
        df_old_clf['Target'].iloc[-1] = input_clf['Target']

    new_row = {"Date": date, "Target": np.nan,'Prediction': prediction_clf}
    new_input = pd.DataFrame.from_dict(new_row)
    df_concat_clf = pd.concat([df_old_clf,new_input])
    df_concat_clf.drop_duplicates(keep='last',inplace=True)
    df_concat_clf.to_sql(f'monitoring_clf', engine, if_exists='replace', index=False)

    df_old_ts = pd.read_sql_query('SELECT * FROM monitoring_time_series ', engine)
    prediction_ts = context['ti'].xcom_pull(key="pred_time_series")
    input_ts = context['ti'].xcom_pull(key="input_ts")
    if date != df_old_clf.iloc[-1:].index.values[0]:
        df_old_ts['CLOSE_diff_t+1'].iloc[-1] = input_ts[-1]
    print(prediction_ts)
    new_row = {"Date": date, "CLOSE_diff_t+1": np.nan,'New_Prediction_ts': prediction_ts.iloc[0].values}
    new_input_ts = pd.DataFrame.from_dict(new_row)
    print(new_input_ts)
    df_concat_clf = pd.concat([df_old_ts,new_input_ts])
    print(df_concat_clf)
    df_concat_clf.drop_duplicates(keep='last',inplace=True)
    df_concat_clf.to_sql(f'monitoring_time_series', engine, if_exists='replace', index=False)
    
with DAG(
    dag_id="dag_with_postgres_hooks",
    default_args=default_args,
    schedule_interval='30 13 * * *'
) as dag:
    task_1 =PythonOperator(
        task_id = "scrap_new_data",
        python_callable=scrap_stock
    )
    task_2 =PythonOperator(
        task_id = "concat_data",
        python_callable=concat_old_data
    )
    task_3 =PythonOperator(
        task_id = "featur_engineer",
        python_callable=transform_data
    )
    task_4 =PythonOperator(
        task_id = "pred_Categorical",
        python_callable=pred_Categorical
    )
    task_5 =PythonOperator(
        task_id = "pred_time_series",
        python_callable=pred_time_series
    )
    task_6 =PythonOperator(
        task_id = "get_src_tables",
        python_callable=get_src_tables
    )
    task_7 =PythonOperator(
        task_id = "plot",
        python_callable=plot
    )
    task_8 =PythonOperator(
        task_id = "plot_github",
        python_callable=plot_github
    )
    task_9 =PythonOperator(
        task_id = "monitoring",
        python_callable=monitoring
    )
    task_6>> task_1 >> task_2 >> task_3>> task_4>> task_5>> task_7>> task_8>> task_9




