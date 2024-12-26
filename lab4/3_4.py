from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, to_date, dayofweek, month, year
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("Predict Purchase and Redeem") \
    .getOrCreate()
############################3

user_profile = spark.read.option("header", "true").csv("user_profile_table.csv")
user_balance = spark.read.option("header", "true").csv("user_balance_table.csv")
mfd_day_share_interest = spark.read.option("header", "true").csv("mfd_day_share_interest.csv")
mfd_bank_shibor = spark.read.option("header", "true").csv("mfd_bank_shibor.csv")

#################
# 转换日期格式
user_balance = user_balance.withColumn("report_date", to_date(col("report_date"), "yyyyMMdd"))
mfd_day_share_interest = mfd_day_share_interest.withColumn("mfd_date", to_date(col("mfd_date"), "yyyyMMdd"))
mfd_bank_shibor = mfd_bank_shibor.withColumn("mfd_date", to_date(col("mfd_date"), "yyyyMMdd"))

# 提取日期特征
user_balance = user_balance.withColumn("year", year(col("report_date"))) \
    .withColumn("month", month(col("report_date"))) \
    .withColumn("day", dayofweek(col("report_date")))

# 合并数据
# 确保使用正确的列名进行join
mergedData = user_profile.join(user_balance, "user_id") \
    .join(mfd_day_share_interest, user_balance["report_date"] == mfd_day_share_interest["mfd_date"]) \
    .join(mfd_bank_shibor, user_balance["report_date"] == mfd_bank_shibor["mfd_date"])
#####################




# 特征选择和转换
assembler = VectorAssembler(inputCols=["total_purchase_amt", "total_redeem_amt", "mfd_daily_yield", "Interest_O_N", "Interest_1_W", "Interest_2_W", "Interest_1_M", "Interest_3_M", "Interest_6_M", "Interest_9_M", "Interest_1_Y"], outputCol="features")
featureData = assembler.transform(mergedData)

train_data, test_data = featureData.randomSplit([0.7, 0.3], seed=42)

rf = RandomForestRegressor(featuresCol="features", labelCol="total_purchase_amt")
pipeline = Pipeline(stages=[rf])

model = pipeline.fit(train_data)

predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="total_purchase_amt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# 生成2014年9月的数据
from datetime import datetime, timedelta

def generate_dates(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

start_date = datetime(2014, 9, 1)
end_date = datetime(2014, 9, 30)
dates = [date.strftime("%Y%m%d") for date in generate_dates(start_date, end_date)]

september_data = spark.createDataFrame(dates, StringType()).withColumnRenamed("value", "report_date")
september_data = september_data.withColumn("report_date", to_date(col("report_date"), "yyyyMMdd"))

# 预测
september_predictions = model.transform(september_data)

september_predictions.select("report_date", "prediction").withColumn("purchase", col("prediction")).withColumn("redeem", col("prediction")) \
    .write.option("header", "true").csv("tc_comp_predict_table.csv")