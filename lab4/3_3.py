from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, dayofweek
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, VectorSizeHint
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType

# 初始化Spark Session
spark = SparkSession.builder \
    .appName("Predict Purchase and Redeem") \
    .getOrCreate()

# 加载数据
user_profile = spark.read.option("header", "true").csv("user_profile_table.csv")
user_balance = spark.read.option("header", "true").csv("user_balance_table.csv")
mfd_day_share_interest = spark.read.option("header", "true").csv("mfd_day_share_interest.csv")
mfd_bank_shibor = spark.read.option("header", "true").csv("mfd_bank_shibor.csv")

# 数据预处理
user_balance = user_balance.withColumn("report_date", to_date(col("report_date"), "yyyyMMdd"))
mfd_day_share_interest = mfd_day_share_interest.withColumn("mfd_date", to_date(col("mfd_date"), "yyyyMMdd"))
mfd_bank_shibor = mfd_bank_shibor.withColumn("mfd_date", to_date(col("mfd_date"), "yyyyMMdd"))

user_balance = user_balance.withColumn("year", year(col("report_date"))) \
    .withColumn("month", month(col("report_date"))) \
    .withColumn("dayofweek", dayofweek(col("report_date")))

# 合并数据
mergedData = user_profile.join(user_balance, "user_id") \
    .join(mfd_day_share_interest, user_balance["report_date"] == mfd_day_share_interest["mfd_date"]) \
    .join(mfd_bank_shibor, user_balance["report_date"] == mfd_bank_shibor["mfd_date"])

# 特征工程
categoricalColumns = ['sex', 'constellation']
numericColumns = ['year', 'month', 'dayofweek', 'mfd_daily_yield', 'Interest_O_N', 'Interest_1_W', 'Interest_2_W', 'Interest_1_M', 'Interest_3_M', 'Interest_6_M', 'Interest_9_M', 'Interest_1_Y']

# 对分类特征进行索引和独热编码
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(mergedData) for column in categoricalColumns]
encoder = OneHotEncoder(inputCols=[column+"_index" for column in categoricalColumns], outputCols=[column+"_vec" for column in categoricalColumns])

assembler = VectorAssembler(inputCols=numericColumns + [column+"_vec" for column in categoricalColumns], outputCol="features", handleInvalid="skip")

# 模型训练
rf = RandomForestRegressor(featuresCol="features", labelCol="total_purchase_amt")
pipeline = Pipeline(stages=indexers + [encoder, assembler, rf])

model = pipeline.fit(mergedData)

# 模型评估
predictions = model.transform(mergedData)
evaluator = RegressionEvaluator(labelCol="total_purchase_amt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# 预测2014年9月数据
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

# 存储结果
september_predictions.select("report_date", "prediction").withColumnRenamed("prediction", "purchase") \
    .withColumn("redeem", col("prediction")) \
    .write.option("header", "true").csv("tc_comp_predict_table.csv")