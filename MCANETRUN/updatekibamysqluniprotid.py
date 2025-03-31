import requests
import mysql.connector
from tqdm import tqdm

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "dti"
}

# 获取 UniProt ID（entry name）
def get_uniprot_id_from_accession(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("uniProtkbId")  # e.g., EGFR_HUMAN
        else:
            print(f"❌ {accession} not found (status {response.status_code})")
    except Exception as e:
        print(f"⚠️ Error fetching {accession}: {e}")
    return None

# 批量处理并更新数据库
def update_uniprot_ids():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 读取所有 accession
        cursor.execute("SELECT uniprot_accession FROM protein")
        accessions = [row[0] for row in cursor.fetchall()]

        for acc in tqdm(accessions, desc="Updating UniProt IDs"):
            uid = get_uniprot_id_from_accession(acc)
            if uid:
                update_query = "UPDATE protein SET uniprot_id = %s WHERE uniprot_accession = %s"
                cursor.execute(update_query, (uid, acc))

        conn.commit()
        print("✅ All UniProt IDs updated.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

# 执行
update_uniprot_ids()