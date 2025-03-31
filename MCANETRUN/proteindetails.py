import pandas as pd
import requests
from tqdm import tqdm
import time

df = pd.read_csv("davis_protein_with_accession.csv")
error_df = df[df["UniProt Accession"] == "Error"].copy()
uniprot_ids = error_df["UniProt ID"].dropna().unique()

results = []

for entry in tqdm(uniprot_ids):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={entry}&fields=accession&format=json&size=1"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                acc = data["results"][0]["primaryAccession"]
            else:
                acc = "Not found"
        else:
            acc = "Error"
    except Exception:
        acc = "Error"

    results.append({"UniProt ID": entry, "Fixed Accession": acc})
    time.sleep(0.5)  # 避免被限速

# 合并回原始数据
fix_df = pd.DataFrame(results)
df = df.merge(fix_df, on="UniProt ID", how="left")
df["UniProt Accession"] = df.apply(
    lambda row: row["Fixed Accession"] if row["UniProt Accession"] == "Error" and row["Fixed Accession"] not in [
        "Error", "Not found"] else row["UniProt Accession"],
    axis=1
)
df.drop(columns=["Fixed Accession"], inplace=True)
df.to_csv("davis_protein_full.csv", index=False)
