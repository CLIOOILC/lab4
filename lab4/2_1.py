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

# 创建Spark会话
session = SparkSession.builder \
    .appName("UrbanMeanBalance") \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

# 数据文件路径
balance_data_file = "/data/user_balance_table.csv"
profile_data_file = "/data/user_profile_table.csv"

# 读取CSV文件，设置header为真以自动读取列名
balance_data_frame = session.read.option("header", "true").csv(balance_data_file)
profile_data_frame = session.read.option("header", "true").csv(profile_data_file)

# 将DataFrame注册为临时视图
balance_data_frame.createOrReplaceTempView("balance_view")
profile_data_frame.createOrReplaceTempView("profile_view")

# 执行SQL查询以计算城市平均余额
query_result = session.sql("""
    SELECT  
        c.city,
        AVG(CAST(ba.balance AS DOUBLE)) AS average_balance
    FROM 
        balance_view ba
    JOIN 
        profile_view c
    ON 
        ba.user_id = c.user_id
    WHERE 
        ba.report_date = '20140301'
    GROUP BY 
        c.city
    ORDER BY 
        average_balance DESC
""")

query_result.show()
session.stop()