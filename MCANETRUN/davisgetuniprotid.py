import requests
import pandas as pd
from tqdm import tqdm
import time

# 加载基因名列
df = pd.read_csv("davis_protein_full_new.csv")
gene_names = df["Gene Name (Original)"].tolist()

# 用来存放结果
results = []

for gene in tqdm(gene_names):
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

    results.append({"Gene Name": gene, "UniProt ID": uniprot_id})

# 保存为 CSV
output_df = pd.DataFrame(results)
output_df.to_csv("davis_protein.csv", index=False)
print("✅ 已生成修正后的 UniProt ID 文件：davis_protein.csv")
