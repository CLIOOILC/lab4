from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CityTopUsers") \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

user_balance_path = "/home/hadoop/Documents/lab4/user_balance_table.csv"
user_profile_path = "/home/hadoop/Documents/lab4/user_profile_table.csv"

# 读取数据到 DataFrame
user_balance_df = spark.read.option("header", "true").csv(user_balance_path)
user_profile_df = spark.read.option("header", "true").csv(user_profile_path)

# 注册 DataFrame 为临时视图
user_balance_df.createOrReplaceTempView("user_balance")
user_profile_df.createOrReplaceTempView("user_profile")

# 执行 SQL 查询
result = spark.sql("""

WITH user_flow AS (
    SELECT 
        p.city AS city_id,
        b.user_id,
        SUM(b.total_purchase_amt + b.total_redeem_amt) AS total_flow
    FROM 
        user_balance b
    JOIN 
        user_profile p
    ON 
        b.user_id = p.user_id
    WHERE 
        b.report_date LIKE '201408%'
    GROUP BY 
        p.City, b.user_id
),
ranked_users AS (
    SELECT 
        city_id,
        user_id,
        total_flow,
        RANK() OVER (PARTITION BY city_id ORDER BY total_flow DESC) AS rank
    FROM 
        user_flow
)
SELECT 
    city_id,
    user_id,
    total_flow
FROM 
    ranked_users
WHERE 
    rank <= 3 
ORDER BY 
    city_id,
    rank;
""")

result.show()
spark.stop()