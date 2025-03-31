import requests
import pandas as pd
from tqdm import tqdm
import time

# 读取你的文件
df = pd.read_csv("kiba_protein_full_final.csv")

# 存结果
uniprot_ids = []

for gene in tqdm(df["Gene Names"]):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&fields=accession&format=json&size=1"
    response = requests.get(url)
    time.sleep(0.2)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            uniprot_id = data["results"][0]["primaryAccession"]
        else:
            uniprot_id = "Not found"
    else:
        uniprot_id = "Error"

    uniprot_ids.append(uniprot_id)

# 添加到原 DataFrame
df["UniProt ID"] = uniprot_ids

# 保存结果
df.to_csv("kiba_protein_full.csv", index=False)
print("✅ 已保存带 UniProt ID 的文件：kiba_protein_full.csv")
