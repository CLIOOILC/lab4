from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, lag, mean
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# 初始化 SparkSession
spark = SparkSession.builder.appName("BalancePrediction").getOrCreate()

# 加载数据
user_balance = spark.read.option("header", "true").option("inferSchema", "true").csv("user_balance_table.csv")
day_share_interest = spark.read.option("header", "true").option("inferSchema", "true").csv("mfd_day_share_interest.csv")
bank_shibor = spark.read.option("header", "true").option("inferSchema", "true").csv("mfd_bank_shibor.csv")

# Step 1: 数据预处理
# 提取 user_balance_table 的关键字段
user_balance_daily = user_balance.groupBy("report_date").agg(
    sum("total_purchase_amt").alias("daily_purchase"),
    sum("total_redeem_amt").alias("daily_redeem")
)

# 将收益率表加入每日数据
combined_data = user_balance_daily.join(day_share_interest, user_balance_daily.report_date == day_share_interest.mfd_date, "left")

# 将银行拆借利率表加入每日数据
combined_data = combined_data.join(bank_shibor, combined_data.report_date == bank_shibor.mfd_date, "left")

# 删除冗余列
combined_data = combined_data.drop("mfd_date")

# Step 2: 特征工程
# 定义窗口
window_spec = Window.orderBy("report_date")

# 生成滞后特征
combined_data = combined_data.withColumn("lag_1_purchase", lag("daily_purchase", 1).over(window_spec))
combined_data = combined_data.withColumn("lag_1_redeem", lag("daily_redeem", 1).over(window_spec))
combined_data = combined_data.withColumn("lag_7_purchase", lag("daily_purchase", 7).over(window_spec))
combined_data = combined_data.withColumn("lag_7_redeem", lag("daily_redeem", 7).over(window_spec))

# 填补缺失值
mean_values = combined_data.select(mean("daily_purchase").alias("mean_purchase"), mean("daily_redeem").alias("mean_redeem")).collect()[0]
combined_data = combined_data.na.fill({"lag_1_purchase": mean_values["mean_purchase"], "lag_1_redeem": mean_values["mean_redeem"]})
combined_data = combined_data.na.fill({"lag_7_purchase": mean_values["mean_purchase"], "lag_7_redeem": mean_values["mean_redeem"]})

# Step 3: 数据集划分
train_data = combined_data.filter((col("report_date") >= "20140701") & (col("report_date") <= "20140831"))
predict_data = combined_data.filter((col("report_date") >= "20140901") & (col("report_date") <= "20140930"))

# 特征向量
assembler = VectorAssembler(inputCols=["lag_1_purchase", "lag_7_purchase", "mfd_daily_yield", "Interest_O_N"], outputCol="features")
train_data = assembler.transform(train_data).select("features", "daily_purchase", "daily_redeem")
predict_data = assembler.transform(predict_data)

# Step 4: 训练模型
# 预测 daily_purchase
lr_purchase = LinearRegression(featuresCol="features", labelCol="daily_purchase")
model_purchase = lr_purchase.fit(train_data)
predicted_purchase = model_purchase.transform(predict_data)

# 预测 daily_redeem
lr_redeem = LinearRegression(featuresCol="features", labelCol="daily_redeem")
model_redeem = lr_redeem.fit(train_data)
predicted_redeem = model_redeem.transform(predict_data)

# Step 5: 保存结果
# 整理预测结果
final_result = predicted_purchase.select(col("report_date"), col("prediction").alias("purchase"))
final_result = final_result.join(predicted_redeem.select(col("report_date").alias("redeem_date"), col("prediction").alias("redeem")),
                                 final_result.report_date == predicted_redeem.redeem_date, "inner").drop("redeem_date")

# 格式化结果并保存为 CSV
final_result.select(col("report_date"), col("purchase").cast("long"), col("redeem").cast("long")).write \
    .option("header", "true") \
    .csv("tc_comp_predict_table.csv")

print("预测结果已保存到 tc_comp_predict_table.csv")