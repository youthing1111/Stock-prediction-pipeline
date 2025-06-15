from pyspark.sql import SparkSession
from pyspark import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import boto3
import pandas as pd
from io import StringIO
from io import BytesIO

spark = SparkSession \
    .builder \
    .appName("Test") \
    .getOrCreate() 

s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='4U247Rhn8cRGtTgiiJUg',
                        aws_secret_access_key='58SHIao9tx0bQNjaS2MyU8rNZdOwUhROsv4yiNyP'
    )

object = s3.Bucket("mlflow-artifacts").Object("acb_stock_202506151415.csv").get()['Body'].read()


s = str(object,'utf-8')
data = StringIO(s)
df = pd.read_csv(data)

df_sp = spark.createDataFrame(df)

w = (Window().partitionBy("date").rowsBetween(-9, 0))
df_sp = df_sp.withColumn('close_sma', F.avg("close").over(w))


print(df_sp.limit(5).show() )

csv_buffer = StringIO()
df.to_csv(csv_buffer)

s3.Object('mlflow-artifacts', 'data_copy.csv').put(Body=csv_buffer.getvalue())


