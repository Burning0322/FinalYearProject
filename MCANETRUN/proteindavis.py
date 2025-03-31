import pandas as pd
import requests
from tqdm import tqdm
import time

# 读取原始 CSV
df = pd.read_csv("davis_protein_full.csv")

# 找出缺失 protein name 的行
missing_df = df[df["Protein Name"].isna() | (df["Protein Name"] == "N/A")].copy()

# 定义提取逻辑（带 fallback）
def fetch_protein_name(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    time.sleep(0.2)
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            protein_block = data.get('proteinDescription', {})
            if "recommendedName" in protein_block:
                return protein_block["recommendedName"]["fullName"]["value"]
            elif "submissionNames" in protein_block:
                return protein_block["submissionNames"][0]["fullName"]["value"]
            elif "alternativeNames" in protein_block:
                return protein_block["alternativeNames"][0]["fullName"]["value"]
            else:
                return "Not found"
        else:
            return "Error"
    except:
        return "Error"

# 开始补全
missing_df["Protein Name"] = missing_df["UniProt ID"].apply(fetch_protein_name)

# 合并回原始 DataFrame
df.update(missing_df)

# 保存新文件
df.to_csv("davis_protein_full.csv", index=False)
print("✅ 已保存：davis_protein_full.csv")
