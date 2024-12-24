'''from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("LoadCSVToSparkSQL") \
    .getOrCreate()

# 加载 CSV 文件到 DataFrame
df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv("/home/hadoop/Documents/lab4/user_balance_table.csv")

# 注册为 Spark SQL 表
df.createOrReplaceTempView("user_balance_table")

# 测试查询
result = spark.sql("SELECT * FROM user_balance_table LIMIT 10")
result.show()'''

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("City Average Balance") \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

# 加载数据
user_balance_path = "/home/hadoop/Documents/lab4/user_balance_table.csv"
user_profile_path = "/home/hadoop/Documents/lab4/user_profile_table.csv"

# 读取CSV文件，自动识别列名（header="true"）
user_balance_df = spark.read.option("header", "true").csv(user_balance_path)
user_profile_df = spark.read.option("header", "true").csv(user_profile_path)

# 注册DataFrame为临时视图
user_balance_df.createOrReplaceTempView("user_balance_table")
user_profile_df.createOrReplaceTempView("user_profile_table")

# 执行Spark SQL查询
result_df = spark.sql("""
    SELECT  
        p.city,
        AVG(CAST(b.tBalance AS DOUBLE)) AS avg_balance
    FROM 
        user_balance_table b
    JOIN 
        user_profile_table p
    ON 
        b.user_id = p.user_id
    WHERE 
        b.report_date = '20140301'
    GROUP BY 
        p.city
    ORDER BY 
        avg_balance DESC
""")

result_df.show()
spark.stop()