from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import desc
from pyspark.sql.window import Window
import boto3
import pandas as pd
from io import StringIO
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit

spark = SparkSession \
    .builder \
    .config(map={"spark.dynamicAllocation.enabled":"false"}) \
    .appName("Feature engineering") \
    .getOrCreate() 

#get data
s3 = boto3.resource('s3',
                        endpoint_url='http://host.docker.internal:9005',
                        aws_access_key_id='',
                        aws_secret_access_key=''
    )
object = s3.Bucket("mlflow-artifacts").Object("data_copy.csv").get()['Body'].read()
s = str(object,'utf-8')
data = StringIO(s)
df = pd.read_csv(data)

#feature engineering
df_sp = spark.createDataFrame(df)
w = (Window().orderBy(desc("date")).rowsBetween(0, 8))
df_sp = df_sp.withColumn('close_sma', F.avg("close").over(w))
df_sp = df_sp.withColumn('close_wma', lit(0))
@pandas_udf(DoubleType())
def ewm_udf_12(s: pd.Series) -> pd.Series:
    return s.ewm(span=12).mean()  
@pandas_udf(DoubleType())
def ewm_udf_26(s: pd.Series) -> pd.Series:
    return s.ewm(span=26).mean()  
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
# Calculate fast_d
df_sp = df_sp.withColumn("fast_d", F.round(F.avg("fast_k").over(fast_k_window), 2))
# Calculate slow_k and slow_d
df_sp = df_sp.withColumn("slow_k", F.col("fast_d"))
df_sp = df_sp.withColumn("slow_d", F.round(F.avg("slow_k").over(fast_k_window), 2))
df_sp = df_sp.withColumn("LW", 100 * (F.col("high") - F.col("close")) / (F.col("high") - F.col("low")+0.0001))
diff_window = (Window().orderBy("date"))
df_sp = df_sp.withColumn('close_t+1', F.lag('close').over(diff_window))
df_sp = df_sp.withColumn('Target', F.when(F.col('close_t+1')>F.col('close'),1).otherwise(0))
df_sp = df_sp.withColumn('acc_dist',F.when(F.col('high') != F.col('low'),((F.col("close") - F.col("low") - F.col("high") + F.col("close")) / (F.col("high") - F.col("low")))*F.col("volume")).otherwise(0))
@pandas_udf(DoubleType())
def acc_dist_ema9(s: pd.Series) -> pd.Series:
    return s.ewm(com=9).mean()  # Adjust the span as needed
df_sp = df_sp.withColumn('acc_dist_ema9', acc_dist_ema9(F.col("acc_dist")))
window_spec = Window.orderBy("date")

#doing the rest of feature engineering not using spark because too weird
df = df_sp.toPandas()
df = df.sort_values(by='date', ascending=True)
df['obv'] = 0
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
df = df.drop(columns=['close_t+1'])
df_1 = df.tail()

#upload transformed data to minio
csv_buffer = StringIO()
df_1.to_csv(csv_buffer)
s3.Object('mlflow-artifacts', 'data_transform.csv').put(Body=csv_buffer.getvalue())


