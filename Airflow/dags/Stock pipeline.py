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
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': '2025-06-07',
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

c = Connection(
    conn_id='postgres',
    conn_type='Postgres',
    login='airflow',
    password='airflow'
)

def update_data():
    #get old data from postgres
    hook = PostgresHook(postgres_conn_id="postgres")
    sql_ts = """ SELECT * FROM public."acb_stock" """
    df_old = hook.get_pandas_df(sql_ts)

    #new data from source
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

    #concat data
    conn = BaseHook.get_connection('postgres')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')
    df_old = df_old.sort_values(by='date')
    df_old.set_index("date",inplace=True)
    df_latest.rename(index={0:date},inplace=True)
    df_latest = df_latest.rename(columns={'HIGH': 'high', 'LOW': 'low','close':'close','OPEN':'open','VOLUME':'volume'})
    df_latest['volume'] = df_latest['volume']/1000000
    df_full = pd.concat([df_old, df_latest])
    df_full.reset_index(names="date",inplace=True)

    #store in postgres and minio
    if set(df_full[['close','open','high','low','volume']].iloc[-2]) == set(df_full[['close','open','high','low','volume']].iloc[-1]):
        print('asdeqfwe')
    else:
        df_full.to_sql(f'acb_stock', engine, if_exists='replace', index=False)
        df_35 = df_full.iloc[-26:]
        csv_buffer = StringIO()
        df_35.to_csv(csv_buffer)
        s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='',
                        aws_secret_access_key='')
        s3.Object('mlflow-artifacts', 'data_copy.csv').put(Body=csv_buffer.getvalue())

def prediction_2():
    #get transformed data for prediction
    s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='',
                        aws_secret_access_key=''
    )
    object = s3.Bucket("mlflow-artifacts").Object("data_transform.csv").get()['Body'].read()
    s = str(object,'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    df_reorder = df[['close', 'open', 'high', 'low', 'volume', 'change', 'close_sma', 'close_wma', 'close_ema12', 'close_ema26', 'MACD', 'best_low', 'best_high', 'fast_k', 'fast_d', 'slow_k', 'slow_d', 'LW', 'Target', 'acc_dist', 'acc_dist_ema9', 'obv', 'obv_ema9', 'pvt', 'pvt_ema9', 'typical_price']]
    input = df_reorder.iloc[[-1]]

    #get model from minio
    temp_model_location = './temp_model.pkl'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(s3.Bucket("mlflow-artifacts").Object("1/232bdcf2cc164c76bade410782fd9a48/artifacts/testmodel/model.xgb").get()['Body'].read())
    temp_model_file.close()
    model = XGBRegressor(enable_categorical=True)
    model.load_model(temp_model_location)
    pred = model.predict(input)

    #visualise
    df_close = df[['date','close']]
    df_close['date'] = pd.to_datetime(df_close['date'],format= '%Y-%m-%d')
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

    #upload to github website
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
    #submit spark for feature engineering
    task_2 = SparkSubmitOperator(
		application = "/opt/airflow/dags/spark_task.py",
        conn_id= 'spark',
		task_id='spark_submit_task',
        env_vars={
            'YARN_CONF_DIR': '/opt/bitnami/spark/yarn'  
        }
	)
    task_4 =PythonOperator(
        task_id = "prediction_2",
        python_callable=prediction_2
    )
    task_1>>task_2>>task_4