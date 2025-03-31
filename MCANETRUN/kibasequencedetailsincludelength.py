import csv
import time
import requests
from tqdm import tqdm

input_file = 'kiba_protein_full.csv'
output_file = 'kiba_protein_full_final.csv'

def fetch_sequence_info(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        time.sleep(0.2)  # 防止限速
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            sequence = data.get('sequence', {}).get('value', '')
            length = data.get('sequence', {}).get('length', 0)
            return sequence, length
        else:
            print(f"❌ Failed to fetch {accession}: HTTP {response.status_code}")
            return '', 0
    except Exception as e:
        print(f"❌ Error for {accession}: {e}")
        return '', 0

# 读取原始数据
proteins = []
with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        proteins.append(row)

# 扩展每个条目
for protein in tqdm(proteins, desc="Fetching sequences"):
    accession = protein['UniProt Accession']
    seq, length = fetch_sequence_info(accession)
    protein['Sequence'] = seq
    protein['Length'] = length

# 写入新的 CSV 文件
fieldnames = list(proteins[0].keys())  # 原字段 + 新字段
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for protein in proteins:
        writer.writerow(protein)

print(f"\n✅ Sequence-enriched dataset saved to: {output_file}")
