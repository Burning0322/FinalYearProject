import pandas as pd
import pymysql

# 读取你生成的 CSV 文件
df = pd.read_csv("davis_protein_full.csv")

# 连接 MySQL
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
    INSERT INTO protein (
        uniprot_id,uniprot_accession, gene_names, organism, protein_name, sequence, length
    ) VALUES (%s, %s, %s, %s, %s, %s,%s)
    """
    cursor.execute(sql, (
        row['UniProt ID'],
        row['UniProt Accession'],
        row['Gene Name'],
        row['Organism'],
        row['Protein Name'],
        row['Sequence'],
        int(row['Length']) if row['Length'] != 'Not found' else None
    ))

# 提交并关闭连接
conn.commit()
cursor.close()
conn.close()

print("✅ 数据成功写入到 protein 表！")
