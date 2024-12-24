from pyspark import SparkContext
from datetime import datetime

sc = SparkContext("local", "UserBalanceAnalysis")

# 加载数据
data = sc.textFile("user_balance_table.csv")

# 去掉表头
header = data.first()
rows = data.filter(lambda line: line != header)

# 解析数据
def parse_line(line):
    fields = line.split(",")
    report_date = fields[1]  # 日期
    total_purchase_amt = float(fields[4])  # 资金流入
    total_redeem_amt = float(fields[8])  # 资金流出
    return (report_date, (total_purchase_amt, total_redeem_amt))

# 计算每天的资金流入和流出
daily_flow = rows.map(parse_line) \
                 .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
                 .sortByKey()

# 格式化输出
result_task1 = daily_flow.map(lambda x: f"{x[0]} {int(x[1][0])} {int(x[1][1])}")
result_task1.saveAsTextFile("output_task1.txt")