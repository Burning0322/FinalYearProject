import requests
import pandas as pd

# 读取CSV文件
df = pd.read_csv('drug_full_extended.csv')

# 获取有null值的cid列表
null_cids = df[df.isnull().any(axis=1)]['PubChem CID'].tolist()

# 定义PubChem API URL模板
url_template = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/MolecularFormula,MolecularWeight,SMILES,CanonicalSMILES,IsomericSMILES,IUPACName,InChI,InChIKey,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,IsotopeAtomCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount,ConformerCount3D,Volume3D,XStericQuadrupole3D,YStericQuadrupole3D,ZStericQuadrupole3D,FeatureAcceptorCount3D,FeatureDonorCount3D,FeatureAnionCount3D,FeatureCationCount3D,FeatureRingCount3D,FeatureHydrophobeCount3D,EffectiveRotorCount3D,Fingerprint2D/JSON"

# 获取数据
def fetch_pubchem_data(cids):
    data = []
    for cid in cids:
        url = url_template.format(cid)
        response = requests.get(url)
        if response.status_code == 200:
            compound_data = response.json().get('PropertyTable', {}).get('Properties', [{}])[0]
            data.append(compound_data)
            print(f"Fetched data for CID {cid}")
        else:
            data.append(None)
    return data

# 获取PubChem数据
pubchem_data = fetch_pubchem_data(null_cids)

# 更新缺失的值
for idx, cid in enumerate(null_cids):
    if pubchem_data[idx] is not None:
        for key, value in pubchem_data[idx].items():
            # 更新对应列
            df.loc[df['PubChem CID'] == cid, key] = value

# 保存更新后的文件
df.to_csv('drug_full_extended_updated.csv', index=False)

print("Updated CSV file saved as 'drug_full_extended_updated.csv'.")