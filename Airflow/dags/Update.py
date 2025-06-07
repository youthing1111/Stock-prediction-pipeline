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

c = Connection(
    conn_id='postgres',
    conn_type='Postgres',
    login='airflow',
    password='airflow',
    host='host.docker.internal'
)

def get_src_tables(**context):
    hook = PostgresHook(postgres_conn_id="postgres")

    sql_ts = """ SELECT * FROM public."acb_stock" """
    df_ts = hook.get_pandas_df(sql_ts)
    context['ti'].xcom_push(key="old data ts", value=df_ts)

def scrap_stock(**context):
    date  = datetime.today().strftime('%Y-%m-%d')
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

with DAG(
    dag_id="Stock_prediction_udpate",
    default_args=default_args,
    schedule_interval='30 13 * * *'
) as dag:
    task_1 =PythonOperator(
        task_id = "get_src_tables",
        python_callable=get_src_tables
    )
    task_2 =PythonOperator(
        task_id = "scrap_new_data",
        python_callable=scrap_stock
    )
    task_1>> task_2