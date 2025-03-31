import pandas as pd
import pymysql

# 读取 CSV 文件（确保路径没错）
df = pd.read_csv("kiba_protein_full.csv")

# 连接到 MySQL
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='root',
    database='dti',
    charset='utf8mb4'
)

cursor = conn.cursor()

# 写入数据
for _, row in df.iterrows():
    sql = """
    REPLACE INTO protein (
        uniprot_id,uniprot_accession, gene_names, organism, protein_name, sequence, length
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (
        row['UniProt ID'],
        row['UniProt Accession'],
        row['Gene Names'],
        row['Organism'],
        row['Protein Name'],
        row['Sequence'],
        int(row['Length']),
    ))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("✅ 数据成功写入 MySQL 数据库中的 protein 表！")
