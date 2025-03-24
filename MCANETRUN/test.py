import pandas as pd
import requests
import time

# 读取已有的 CID 列表
df = pd.read_csv("details.csv")

# PubChem 支持的字段（建议根据需要精简）
fields = [
    "MolecularFormula", "MolecularWeight", "SMILES", "CanonicalSMILES", "IsomericSMILES",
    "IUPACName", "InChI", "InChIKey", "XLogP", "ExactMass", "MonoisotopicMass",
    "TPSA", "Complexity", "Charge", "HBondDonorCount", "HBondAcceptorCount",
    "RotatableBondCount", "HeavyAtomCount", "IsotopeAtomCount",
    "DefinedAtomStereoCount", "UndefinedAtomStereoCount",
    "DefinedBondStereoCount", "UndefinedBondStereoCount",
    "CovalentUnitCount", "ConformerCount3D", "Volume3D",
    "XStericQuadrupole3D", "YStericQuadrupole3D", "ZStericQuadrupole3D",
    "FeatureAcceptorCount3D", "FeatureDonorCount3D", "FeatureAnionCount3D",
    "FeatureCationCount3D", "FeatureRingCount3D", "FeatureHydrophobeCount3D",
    "EffectiveRotorCount3D", "Fingerprint2D"
]

error_cid = []

def fetch_pubchem_data(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{','.join(fields)}/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"❌ HTTP {r.status_code} for CID {cid}")
            return {}
        json_data = r.json()
        return json_data["PropertyTable"]["Properties"][0]
    except Exception as e:
        print(f"❌ Error fetching CID {cid}: {e}")
        error_cid.append(cid)
        return {}

results = []
for cid in df["PubChem CID"]:
    print(f"Fetching CID {cid}...")
    data = fetch_pubchem_data(cid)
    if data:
        data["PubChem CID"] = cid
        results.append(data)
    print(data)
    time.sleep(0.2)  # 避免请求过快被封

# 保存
df_new = pd.DataFrame(results)
print(error_cid)
df_final = pd.merge(df, df_new, on="PubChem CID", how="left")
df_final.to_csv("drug_full_extended.csv", index=False)
print("✅ All data saved to drug_full_extended.csv")
