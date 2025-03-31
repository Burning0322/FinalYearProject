import requests
import time
from tqdm import tqdm
import csv

# Name mapping from the query
name_mapping = {
    "AMPK-alpha1": "PRKAA1",
    "AMPK-alpha2": "PRKAA2",
    "PKAC-alpha": "PRKACA",
    "PKAC-beta": "PRKACB",
    "p38-delta": "MAPK13",
    "p38-gamma": "MAPK12",
    "CDK4-cyclinD1": "CDK4",
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
    "PFCDPK1(Pfalciparum)": "CDPK1",
    "PFPK5(Pfalciparum)": "PK5",
    "PKNB(Mtuberculosis)": "pknB",
}

# Organism mapping for non-human proteins
organism_mapping = {
    "PFCDPK1(Pfalciparum)": "Plasmodium falciparum",
    "PFPK5(Pfalciparum)": "Plasmodium falciparum",
    "PKNB(Mtuberculosis)": "Mycobacterium tuberculosis"
}

# [Insert the get_uniprot_entry_api function from Step 4 here]

# Read Davis.txt and extract protein names
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

protein = set(d['protein_name'] for d in data)
sort_protein = sorted(protein)
print("Protein names:", sort_protein)
print("Number of unique proteins:", len(sort_protein))

# Process proteins and write to CSV
count = 0
error = []
output_file = 'davis_protein_full_new_final.csv'


def get_uniprot_entry_api(gene_name, organism_priority="Homo sapiens"):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}&fields=accession,gene_names,organism_name&format=json"
    try:
        time.sleep(0.2)  # Rate limiting
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to access UniProt API for {gene_name}: {response.status_code}")
            return None
        data = response.json()
        results = data.get("results", [])
        if not results:
            print(f"No UniProt entries found for {gene_name}")
            return None

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
            organism = result.get("organism", {}).get("scientificName", "")

            if gene_name.upper() in [gn.upper() for gn in gene_names]:
                if organism_priority and organism_priority.lower() in organism.lower():
                    print(f"Found UniProt Entry for {gene_name} ({organism}): {entry}")
                    return entry

        print(f"No UniProt entry found for {gene_name} in {organism_priority}")
        return None
    except Exception as e:
        print(f"Error querying UniProt API for {gene_name}: {e}")
        return None

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Protein Name', 'UniProt Entry'])

    for protein_name in tqdm(sort_protein, desc="Processing proteins"):
        query_name = name_mapping.get(protein_name, protein_name)
        organism = organism_mapping.get(protein_name, "Homo sapiens")
        uniprot_entry = get_uniprot_entry_api(query_name, organism)

        if uniprot_entry:
            print(f"Protein Name: {protein_name}, UniProt Entry: {uniprot_entry}")
            writer.writerow([protein_name, uniprot_entry])
            count += 1
        else:
            print(f"Could not find UniProt Entry for {protein_name}")
            error.append(protein_name)

print(f"\nProcessing completed! Success: {count}, Failed: {len(error)}")
print(f"Results saved to {output_file}")

if error:
    print("\nFailed to find UniProt entries for:")
    for protein in error:
        print(f" - {protein}")