import pandas as pd
import pymysql

# 1. 读取 CSV 文件
df = pd.read_csv("davis_details.csv")

# 确保列名一致（可以用 print(df.columns) 来检查）
df.columns = [col.strip() for col in df.columns]  # 去除空格

# 2. 连接到 MySQL 数据库
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="dti",       # 你的数据库名
    charset="utf8mb4"
)
cursor = conn.cursor()

# 3. 遍历 CSV，每行执行 UPDATE
for index, row in df.iterrows():
    query = row["Query"]
    pubchem_cid = str(row["PubChem CID"])  # 确保为字符串类型

    sql = """
    UPDATE drug
    SET query = %s
    WHERE compound_id = %s
    """
    cursor.execute(sql, (query, pubchem_cid))

# 4. 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有 query 字段已成功更新！")
