from pyspark import SparkContext
import datetime as dt

# 初始化Spark上下文
spark_context = SparkContext("local", "UserActivityAnalysis")

# 读取数据文件
balance_records = spark_context.textFile("balance_records.csv")

# 移除数据文件的首行（标题行）
header_line = balance_records.first()
record_rows = balance_records.filter(lambda record: record != header_line)

# 数据解析函数，提取用户ID和报告日期
def extract_user_and_date(record):
    columns = record.split(",")
    user_identifier = columns[0]  # 用户标识符
    record_date = columns[1]  # 报告日期
    return (user_identifier, record_date)

# 判断日期是否为2014年8月
def filter_august_2014(date_str):
    date_format = dt.datetime.strptime(date_str, "%Y%m%d")
    return date_format.year == 2014 and date_format.month == 8

# 统计每个用户在2014年8月的活跃天数
monthly_active_users = record_rows.map(extract_user_and_date) \
                                 .filter(lambda x: filter_august_2014(x[1])) \
                                 .distinct() \
                                 .map(lambda x: (x[0], 1)) \
                                 .reduceByKey(lambda a, b: a + b)

# 筛选出活跃用户（至少5天记录）
threshold_active_users = monthly_active_users.filter(lambda x: x[1] >= 5).count()

# 输出结果
output_file = "active_user_summary.txt"
with open(output_file, "w") as output:
    output.write(f"{threshold_active_users}")