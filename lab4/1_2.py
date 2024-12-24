from pyspark import SparkContext
from datetime import datetime

sc = SparkContext("local", "UserBalanceAnalysis")

# 加载数据
data = sc.textFile("user_balance_table.csv")

# 去掉表头
header = data.first()
rows = data.filter(lambda line: line != header)

# 解析数据，提取用户和日期
def parse_user_date(line):
    fields = line.split(",")
    user_id = fields[0]  # 用户 ID
    report_date = fields[1]  # 日期
    return (user_id, report_date)

# 过滤出 2014 年 8 月的数据
def is_august_2014(report_date):
    date_obj = datetime.strptime(report_date, "%Y%m%d")
    return date_obj.year == 2014 and date_obj.month == 8

# 统计每个用户在 2014 年 8 月的活跃天数
active_users = rows.map(parse_user_date) \
                   .filter(lambda x: is_august_2014(x[1])) \
                   .distinct() \
                   .map(lambda x: (x[0], 1)) \
                   .reduceByKey(lambda x, y: x + y)

# 筛选出活跃用户（至少 5 天记录）
active_user_count = active_users.filter(lambda x: x[1] >= 5).count()

# 输出结果
with open("output_task2.txt", "w") as f:
    f.write(f"{active_user_count}")