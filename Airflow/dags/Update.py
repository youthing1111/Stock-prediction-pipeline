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

with DAG(
    dag_id="Stock_prediction_udpate",
    default_args=default_args,
    schedule_interval='30 13 * * *'
) as dag:
    task_1 =PythonOperator(
        task_id = "get_src_tables",
        python_callable=get_src_tables
    )
    task_1