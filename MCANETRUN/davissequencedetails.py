import requests
import time
from tqdm import tqdm
import csv

# 映射表：处理特殊/别名蛋白名
name_mapping = {
    "AMPK-alpha1": "PRKAA1",
    "AMPK-alpha2": "PRKAA2",
    "PKAC-alpha": "PRKACA",
    "PKAC-beta": "PRKACB",
    "p38-delta": "MAPK13",
    "p38-gamma": "MAPK12",
    "CDK4-cyclinD1": "CDK4",  # cyclinD1 是 CCND1，可拆分
    "JAK1(JH1domain-catalytic)": "JAK1",
    "JAK2(JH1domain-catalytic)": "JAK2",
    "JAK3(JH1domain-catalytic)": "JAK3",
    "TYK2(JH1domain-catalytic)": "TYK2",
    "RSK1(KinDom.1-N-terminal)": "RPS6KA1",
    "RSK2(KinDom.1-N-terminal)": "RPS6KA2",
    "RSK3(KinDom.1-N-terminal)": "RPS6KA3",
    "RSK4(KinDom.1-N-terminal)": "RPS6KA6",
    "RPS6KA4(KinDom.1-N-terminal)": "RPS6KA4",
    "RPS6KA5(KinDom.1-N-terminal)": "RPS6KA5",
    "GCN2(KinDom2S808G)": "EIF2AK4",
    "PFTAIRE2": "CDK15",
    "PFCDPK1(Pfalciparum)": "CDPK1",  # 仅当你改 organism 才能查到
    "PFPK5(Pfalciparum)": "PK5",
    "PKNB(Mtuberculosis)": "pknB",
}

# 查询 UniProt API
def get_uniprot_entry_api(gene_name, organism_priority="Homo sapiens"):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}&fields=accession,gene_names,organism_name&format=json"
    try:
        time.sleep(0.2)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to access UniProt API for {gene_name}")
            return None

        data = response.json()
        results = data.get("results", [])
        if not results:
            print(f"No results found for {gene_name}")
            return None

        # 优先返回匹配物种
        for result in results:
            entry = result.get("primaryAccession")
            gene_names = []
            genes = result.get("genes", [])
            for gene in genes:
                if "geneName" in gene and "value" in gene["geneName"]:
                    gene_names.append(gene["geneName"]["value"])
                synonyms = gene.get("synonyms", [])
                for synonym in synonyms:
                    if "value" in synonym:
                        gene_names.append(synonym["value"])

            organism = result.get("organism", {}).get("scientificName", "")

            if gene_name.upper() not in [gn.upper() for gn in gene_names]:
                continue

            if organism_priority.lower() in organism.lower():
                print(f"✅ Found: {gene_name} ({organism}) → {entry}")
                return entry

        # fallback
        for result in results:
            entry = result.get("primaryAccession")
            gene_names = []
            genes = result.get("genes", [])
            for gene in genes:
                if "geneName" in gene and "value" in gene["geneName"]:
                    gene_names.append(gene["geneName"]["value"])
                synonyms = gene.get("synonyms", [])
                for synonym in synonyms:
                    if "value" in synonym:
                        gene_names.append(synonym["value"])

            organism = result.get("organism", {}).get("scientificName", "Unknown organism")

            if gene_name.upper() in [gn.upper() for gn in gene_names]:
                print(f"⚠️ Found (fallback): {gene_name} ({organism}) → {entry}")
                return entry

        print(f"❌ No matching UniProt Entry for {gene_name}")
        return None

    except Exception as e:
        print(f"Error querying UniProt API for {gene_name}: {e}")
        return None


# 读取 Davis 数据
with open('Davis.txt', 'r') as f:
    lines = f.readlines()

data = []
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

protein_set = set(d['protein_name'] for d in data)
sort_protein = sorted(protein_set)
print(f"Total unique proteins: {len(sort_protein)}")

# 输出文件
output_file = 'davis_protein_full_new.csv'
count = 0
error = []

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Gene Name (Original)', 'Query Name', 'UniProt Entry'])

    for protein_name in tqdm(sort_protein, desc="Processing proteins"):
        query_name = name_mapping.get(protein_name, protein_name)
        uniprot_entry = get_uniprot_entry_api(query_name)

        if uniprot_entry:
            writer.writerow([protein_name, query_name, uniprot_entry])
            count += 1
        else:
            error.append(protein_name)

# 打印统计
print(f"\n✅ Processing completed! Success: {count}, Failed: {len(error)}")
print(f"Results saved to {output_file}")

if error:
    print("\n❗ Failed to find UniProt entries for:")
    for e in error:
        print(f" - {e}")


# ✅ Processing completed! Success: 372, Failed: 7
# Results saved to davis_protein_full_new.csv
#
# ❗ Failed to find UniProt entries for:
#  - CDC2L1
#  - CSK
#  - DCAMKL2
#  - DLK
#  - DYRK1B
#  - GRK4
#  - PDPK1