from pyspark.sql import SparkSession
from pyspark import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.functions import desc
from pyspark.sql.window import Window
import boto3
import pandas as pd
from io import StringIO
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.types import DoubleType, StructField
from pyspark.sql import pandas as ps

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

w = (Window().orderBy(desc("date")).rowsBetween(0, 9))
df_sp = df_sp.withColumn('close_sma', F.avg("close").over(w))

@pandas_udf(DoubleType())
def ewm_udf_12(s: pd.Series) -> pd.Series:
    return s.ewm(span=12).mean()  # Adjust the span as needed

@pandas_udf(DoubleType())
def ewm_udf_26(s: pd.Series) -> pd.Series:
    return s.ewm(span=26).mean()  # Adjust the span as needed

df_sp = df_sp.withColumn('close_ema12', ewm_udf_12(F.col("close")))
df_sp = df_sp.withColumn('close_ema26', ewm_udf_26(F.col("close")))
df_sp = df_sp.withColumn('MACD', df_sp.close_ema12 - df_sp.close_ema26 )


print(df_sp.collect()[:5])

csv_buffer = StringIO()
df.to_csv(csv_buffer)

s3.Object('mlflow-artifacts', 'data_copy.csv').put(Body=csv_buffer.getvalue())


