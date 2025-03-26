import requests
import time
from tqdm import tqdm
import csv

def get_uniprot_entry_api(gene_name, organism_priority=""):
    """
    通过 UniProt REST API 查询基因名称，返回对应的 UniProt Entry。

    参数:
    gene_name (str): 基因名称，例如 "AAK1"
    organism_priority (str): 优先选择的物种，默认为 "Homo sapiens"

    返回:
    str: 对应的 UniProt Entry（例如 "Q2M2I8"），如果未找到则返回 None
    """
    # 构造 API 查询 URL
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}&fields=accession,gene_names,organism_name&format=json"

    try:
        time.sleep(0.2)
        # 发送 GET 请求
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to access UniProt API for {gene_name}")
            return None

        # 解析 JSON 结果
        data = response.json()
        results = data.get("results", [])
        if not results:
            print(f"No results found for {gene_name}")
            return None

        # 遍历结果，优先选择指定物种
        for result in results:
            entry = result.get("primaryAccession")
            # 提取 Gene Names
            gene_names = []
            genes = result.get("genes", [])
            for gene in genes:
                # 提取 geneName 字段
                gene_name_dict = gene.get("geneName", {})
                if gene_name_dict and "value" in gene_name_dict:
                    gene_names.append(gene_name_dict["value"])
                # 提取 synonyms（别名）
                synonyms = gene.get("synonyms", [])
                for synonym in synonyms:
                    if "value" in synonym:
                        gene_names.append(synonym["value"])

            # 提取 Organism
            organism = result.get("organism", {}).get("scientificName", "")

            # 确认 Gene Names 包含查询的基因名称
            if gene_name.upper() not in [gn.upper() for gn in gene_names]:
                continue

            # 优先选择指定物种
            if organism_priority.lower() in organism.lower():
                print(f"\nFound UniProt Entry for {gene_name} ({organism}): {entry}")
                return entry

        # 如果没有找到指定物种，返回第一个匹配的条目
        for result in results:
            entry = result.get("primaryAccession")
            gene_names = []
            genes = result.get("genes", [])
            for gene in genes:
                gene_name_dict = gene.get("geneName", {})
                if gene_name_dict and "value" in gene_name_dict:
                    gene_names.append(gene_name_dict["value"])
                synonyms = gene.get("synonyms", [])
                for synonym in synonyms:
                    if "value" in synonym:
                        gene_names.append(synonym["value"])

            organism = result.get("organism", {}).get("scientificName", "Unknown organism")

            if gene_name.upper() in [gn.upper() for gn in gene_names]:
                print(f"Found UniProt Entry for {gene_name} ({organism}): {entry}")
                return entry

        print(f"No matching UniProt Entry found for {gene_name}")
        return None

    except Exception as e:
        print(f"Error querying UniProt API for {gene_name}: {e}")
        return None



with open('Davis.txt', 'r') as f:
    lines = f.readlines()

data =[]

for line in lines:
    parts = line.strip().split(' ', 4)
    if len(parts) == 5:
        compound_id, protein_name, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
        sequence, label = rest.rsplit(' ', 1)
        data.append({
            'compound_id': compound_id,
            'protein_name': protein_name,
            'smiles': smiles,
            'sequence': sequence,
            'label': int(label)
        })

protein = set(d['protein_name'] for d in data)
sort_protein = sorted(protein)

print(sort_protein)
print(len(sort_protein))

# 准备结果存储
results = []
count = 0
error = []

# 准备CSV文件
output_file = 'davis_protein_full.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Gene Name', 'UniProt Entry'])  # 写入表头

    # 处理每个蛋白质
    for protein_name in tqdm(sort_protein, desc="Processing proteins"):
        uniprot_entry = get_uniprot_entry_api(protein_name)

        if uniprot_entry:
            # 打印到控制台
            print(f"Gene Name: {protein_name}, UniProt Entry: {uniprot_entry}")
            # 写入CSV
            writer.writerow([protein_name, uniprot_entry])
            count += 1
        else:
            print(f"Could not find UniProt Entry for {protein_name}")
            error.append(protein_name)

print(f"\nProcessing completed! Success: {count}, Failed: {len(error)}")
print(f"Results saved to {output_file}")

# 打印错误列表（如果有）
if error:
    print("\nFailed to find UniProt entries for:")
    for protein in error:
        print(f" - {protein}")