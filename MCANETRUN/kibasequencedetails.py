import requests
import time
import csv
from tqdm import tqdm

# 🔍 获取 UniProt 蛋白信息（通过 accession ID）
def get_uniprot_info_by_accession(accession):
    """
    通过 UniProt accession（如 O00141）获取蛋白基本信息
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        time.sleep(0.2)  # 加一点延迟防止请求太快被限流
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            # 基因名称
            gene_names = []
            for g in data.get('genes', []):
                if 'geneName' in g and 'value' in g['geneName']:
                    gene_names.append(g['geneName']['value'])

            # 蛋白名称
            protein_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')

            # 物种名称
            organism = data.get('organism', {}).get('scientificName', '')

            return {
                'accession': accession,
                'gene_names': gene_names,
                'protein_name': protein_name,
                'organism': organism
            }

        else:
            print(f"Failed to fetch info for {accession}")
            return None

    except Exception as e:
        print(f"Error retrieving {accession}: {e}")
        return None


# 📂 读取 KIBA 数据文件
with open('KIBA.txt', 'r') as f:
    lines = f.readlines()

# 提取 protein accession 列表（去重 + 排序）
protein_ids = set()
for line in lines:
    parts = line.strip().split(' ', 4)
    if len(parts) == 5:
        protein_id = parts[1]
        protein_ids.add(protein_id)

sorted_proteins = sorted(protein_ids)
print(f"Total unique proteins: {len(sorted_proteins)}")

# 💾 写入 CSV
output_file = 'kiba_protein_full.csv'
error_list = []
count = 0

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['UniProt Accession', 'Gene Names', 'Organism', 'Protein Name'])

    for accession in tqdm(sorted_proteins, desc="Processing proteins"):
        info = get_uniprot_info_by_accession(accession)

        if info:
            writer.writerow([
                accession,
                ', '.join(info['gene_names']),
                info['organism'],
                info['protein_name']
            ])
            count += 1
        else:
            error_list.append(accession)

# ✅ 完成提示
print(f"\nFinished! Success: {count}, Failed: {len(error_list)}")
if error_list:
    print("\nFailed to fetch info for:")
    for e in error_list:
        print(f" - {e}")
