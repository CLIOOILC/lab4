from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, lag, mean
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas

# 初始化 SparkSession
spark = SparkSession.builder.appName("PurchaseRedeemPrediction").getOrCreate()

# 加载数据
user_balance = spark.read.option("header", "true").option("inferSchema", "true").csv("user_balance_table.csv")
day_share_interest = spark.read.option("header", "true").option("inferSchema", "true").csv("mfd_day_share_interest.csv")
bank_shibor = spark.read.option("header", "true").option("inferSchema", "true").csv("mfd_bank_shibor.csv")

# Step 1: 数据预处理
# 汇总申购与赎回数据
user_balance_daily = user_balance.groupBy("report_date").agg(
    sum("total_purchase_amt").alias("daily_purchase"),
    sum("total_redeem_amt").alias("daily_redeem")
)

# 合并收益率表
combined_data = user_balance_daily.join(
    day_share_interest,
    user_balance_daily.report_date == day_share_interest.mfd_date,
    "left"
)

# 合并银行拆借利率表
combined_data = combined_data.join(
    bank_shibor,
    combined_data.report_date == bank_shibor.mfd_date,
    "left"
)

# 删除冗余列
combined_data = combined_data.drop("mfd_date")

# Step 2: 特征工程
# 定义窗口函数以生成滞后特征
window_spec = Window.orderBy("report_date")
combined_data = combined_data.withColumn("lag_1_purchase", lag("daily_purchase", 1).over(window_spec))
combined_data = combined_data.withColumn("lag_7_purchase", lag("daily_purchase", 7).over(window_spec))
combined_data = combined_data.withColumn("lag_1_redeem", lag("daily_redeem", 1).over(window_spec))
combined_data = combined_data.withColumn("lag_7_redeem", lag("daily_redeem", 7).over(window_spec))

# 填充缺失值
mean_values = combined_data.select(
    mean("daily_purchase").alias("mean_purchase"),
    mean("daily_redeem").alias("mean_redeem")
).collect()[0]
combined_data = combined_data.na.fill({
    "lag_1_purchase": mean_values["mean_purchase"],
    "lag_7_purchase": mean_values["mean_purchase"],
    "lag_1_redeem": mean_values["mean_redeem"],
    "lag_7_redeem": mean_values["mean_redeem"],
    "mfd_daily_yield": 0.0,  # 填充缺失收益率
    "Interest_O_N": 0.0      # 填充缺失利率
})

# 转换数据类型为 DoubleType
combined_data = combined_data.select(
    col("report_date"),
    col("daily_purchase").cast("double"),
    col("daily_redeem").cast("double"),
    col("mfd_daily_yield").cast("double"),
    col("Interest_O_N").cast("double"),
    col("lag_1_purchase").cast("double"),
    col("lag_7_purchase").cast("double"),
    col("lag_1_redeem").cast("double"),
    col("lag_7_redeem").cast("double")
)

# Step 3: 数据集划分
train_data = combined_data.filter((col("report_date") >= "20140701") & (col("report_date") <= "20140831"))
predict_data = combined_data.filter((col("report_date") >= "20140901") & (col("report_date") <= "20140930"))

# 特征向量
assembler = VectorAssembler(
    inputCols=["lag_1_purchase", "lag_7_purchase", "mfd_daily_yield", "Interest_O_N"],
    outputCol="features"
)
train_data = assembler.transform(train_data).select("features", "daily_purchase", "daily_redeem")
predict_data = assembler.transform(predict_data)

# Step 4: 模型训练和预测
# 预测 daily_purchase
lr_purchase = LinearRegression(featuresCol="features", labelCol="daily_purchase")
model_purchase = lr_purchase.fit(train_data)
predicted_purchase = model_purchase.transform(predict_data).select("report_date", col("prediction").alias("purchase"))

# 预测 daily_redeem
lr_redeem = LinearRegression(featuresCol="features", labelCol="daily_redeem")
model_redeem = lr_redeem.fit(train_data)
predicted_redeem = model_redeem.transform(predict_data).select("report_date", col("prediction").alias("redeem"))

# Step 5: 整理预测结果并保存
final_result = predicted_purchase.join(
    predicted_redeem,
    "report_date"
)

# 收集结果到本地
result_df = final_result.select(
    col("report_date"),
    col("purchase").cast("long"),
    col("redeem").cast("long")
).toPandas()

# 保存为单一 CSV 文件
result_df.to_csv("tc_comp_predict_table.csv", index=False)

print("预测结果已保存到 tc_comp_predict_table.csv")