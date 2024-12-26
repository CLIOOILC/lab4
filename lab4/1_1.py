from pyspark import SparkContext
import datetime

# 初始化Spark上下文
spark_context = SparkContext("local", "BalanceAnalysis")

# 读取数据文件
balance_data = spark_context.textFile("balance_data.csv")

# 移除数据文件的首行（标题行）
header_line = balance_data.first()
data_rows = balance_data.filter(lambda record: record != header_line)

# 数据解析函数
def extract_info(record):
    elements = record.split(",")
    date_of_report = elements[1]  # 报告日期
    inflow_funds = float(elements[4])  # 资金流入量
    outflow_funds = float(elements[8])  # 资金流出量
    return (date_of_report, (inflow_funds, outflow_funds))

# 聚合计算每日资金流动
daily_funds_flow = data_rows.map(extract_info) \
                           .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                           .sortByKey()

# 结果格式化并保存
formatted_output = daily_funds_flow.map(lambda pair: f"{pair[0]} {int(pair[1][0])} {int(pair[1][1])}")
formatted_output.saveAsTextFile("daily_funds_flow_output.txt")