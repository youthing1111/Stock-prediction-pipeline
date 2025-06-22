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

w = (Window().orderBy(desc("date")).rowsBetween(0, 8))
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

window_spec = (Window().orderBy(desc("date")).rowsBetween(0,14 - 1))
# Calculate the best low and high
df_sp = df_sp.withColumn("best_low", F.min("close").over(window_spec))
df_sp = df_sp.withColumn("best_high", F.max("close").over(window_spec))
df_sp = df_sp.withColumn(
    "fast_k",
    100 * (F.col("close") - F.col("best_low")) / (F.col("best_high") - F.col("best_low")+0.0001)
)
fast_k_window = Window.orderBy(desc("date")).rowsBetween(0, 2)
# Calculate fast %D
df_sp = df_sp.withColumn("fast_d", F.round(F.avg("fast_k").over(fast_k_window), 2))
# Calculate slow %K and slow %D
df_sp = df_sp.withColumn("slow_k", F.col("fast_d"))
df_sp = df_sp.withColumn("slow_d", F.round(F.avg("slow_k").over(fast_k_window), 2))

df_sp = df_sp.withColumn("LW", 100 * (F.col("high") - F.col("close")) / (F.col("high") - F.col("low")+0.0001))
diff_window = (Window().orderBy(desc("date")))
df_sp = df_sp.withColumn('close_t+1', F.lag('close').over(diff_window))
df_sp = df_sp.withColumn('Target', F.when(F.col('close_t+1')>F.col('close'),1).otherwise(0))
df_sp = df_sp.withColumn('acc_dist',F.when(F.col('high') != F.col('low'),((F.col("close") - F.col("low") - F.col("high") + F.col("close")) / (F.col("high") - F.col("low")))*F.col("volume")).otherwise(0))
@pandas_udf(DoubleType())
def acc_dist_ema_9(s: pd.Series) -> pd.Series:
    return s.ewm(com=9).mean()  # Adjust the span as needed
df_sp = df_sp.withColumn('acc_dist_ema_9', acc_dist_ema_9(F.col("acc_dist")))

df_sp = df_sp.withColumn("obv", F.lit(0))
window_spec = Window.orderBy("date")
# Calculate the OBV
df_sp = df_sp.withColumn("prev_close", F.lag('close').over(window_spec))
df_sp = df_sp.withColumn("prev_obv", F.lag("volume").over(window_spec))
df_sp = df_sp.withColumn(
    "obv",
    F.when(F.col('close') > F.col("prev_close"), F.col('volume') + F.col("prev_obv"))
        .when(F.col('close') < F.col("prev_close"), F.col("prev_obv") - F.col('volume'))
        .otherwise(F.col("prev_obv"))
)

print(df_sp.collect()[:5])

csv_buffer = StringIO()
df.to_csv(csv_buffer)

s3.Object('mlflow-artifacts', 'data_copy.csv').put(Body=csv_buffer.getvalue())


