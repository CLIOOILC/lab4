from pyspark.sql import SparkSession

# 初始化Spark会话
spark_session = SparkSession.builder \
    .appName("TopUserActivityByCity") \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

# 定义数据文件路径
balance_data_location = "/data/user_balance.csv"
profile_data_location = "/data/user_profile.csv"

# 将CSV文件加载到DataFrame中
balance_frame = spark_session.read.option("header", "true").csv(balance_data_location)
profile_frame = spark_session.read.option("header", "true").csv(profile_data_location)

# 将DataFrame注册为全局临时视图
balance_frame.createOrReplaceTempView("balance_view")
profile_frame.createOrReplaceTempView("profile_view")

# 执行SQL查询以找出每个城市的前三大用户资金流动量
top_users_query = spark_session.sql("""
WITH city_user_flow AS (
    SELECT 
        profile.city AS city_name,
        balance.user_id,
        SUM(balance.total_purchase_amt + balance.total_redeem_amt) AS flow_total
    FROM 
        balance_view balance
    JOIN 
        profile_view profile
    ON 
        balance.user_id = profile.user_id
    WHERE 
        balance.report_date LIKE '201408%'
    GROUP BY 
        profile.city, balance.user_id
),
city_user_ranking AS (
    SELECT 
        city_name,
        user_id,
        flow_total,
        DENSE_RANK() OVER (PARTITION BY city_name ORDER BY flow_total DESC) AS user_rank
    FROM 
        city_user_flow
)
SELECT 
    city_name,
    user_id,
    flow_total
FROM 
    city_user_ranking
WHERE 
    user_rank <= 3 
ORDER BY 
    city_name,
    user_rank;
""")

top_users_query.show()
spark_session.stop()