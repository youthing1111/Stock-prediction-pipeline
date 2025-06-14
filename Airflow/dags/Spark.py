import airflow
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow import DAG

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': '2025-06-12',
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

with DAG(
    dag_id="Test_spark",
    default_args=default_args,
    #schedule='30 13 * * *'
) as dag:
    task_1 = SparkSubmitOperator(
		application = "/opt/airflow/dags/spark_task.py",
        conn_id= 'spark',
		task_id='spark_submit_task',
        env_vars={
            'YARN_CONF_DIR': '/opt/bitnami/spark/yarn'  # Set this if using Yarn
        }
		)
    task_1




