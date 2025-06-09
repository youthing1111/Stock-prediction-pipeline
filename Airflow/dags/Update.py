from github import Github
from datetime import datetime, timedelta
#from airflow.utils.dates import days_ago
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
import mlflow
import boto3
from airflow.hooks.base import BaseHook
import pickle
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import io 
from xgboost import XGBClassifier, XGBRegressor

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

def update_data(**context):
    hook = PostgresHook(postgres_conn_id="postgres")

    sql_ts = """ SELECT * FROM public."acb_stock" """
    df_old = hook.get_pandas_df(sql_ts)

    date  = datetime.today().strftime('%Y-%m-%d')

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

    conn = BaseHook.get_connection('postgres')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')

    #df_old = df_old.drop(columns=['change'])
    df_old = df_old.sort_values(by='date')
    df_old.set_index("date",inplace=True)

    df_latest.rename(index={0:date},inplace=True)
    df_latest = df_latest.rename(columns={'HIGH': 'high', 'LOW': 'low','CLOSE':'close','OPEN':'open','VOLUME':'volume'})
    df_latest['volume'] = df_latest['volume']/1000000

    df_full = pd.concat([df_old, df_latest])
    df_full.reset_index(names="date",inplace=True)
    df_full.drop_duplicates(inplace=True,keep='last')

    df_full.to_sql(f'acb_stock', engine, if_exists='replace', index=False)

with DAG(
    dag_id="Stock_prediction_update",
    default_args=default_args,
    #schedule='30 13 * * *'
) as dag:
    task_1 =PythonOperator(
        task_id = "update_data",
        python_callable=update_data
    )
    task_1