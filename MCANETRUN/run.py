import os
import requests
import pandas as pd
import time
import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 设置 requests 重试策略
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# 从 KIBA.txt 读取化合物
with open('KIBA.txt', 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    parts = line.strip().split(' ', 4)
    if len(parts) == 5:
        compound_id = parts[0]
        data.append({'compound_id': compound_id})

ligands = sorted(set(d['compound_id'] for d in data))
print(f"待处理化合物数量: {len(ligands)}")

# 初始化列表
results = []
PubChem_CID = []
molecular_weight = []
molecular_formula = []
error_cid = []
count = 0
total = 0

# 通过 PUG REST API 获取数据
for query in ligands:
    try:
        start = time.time()
        print(f"处理化合物: {query}")

        # Step 1: 根据 Identifier（如 CHEMBL ID）获取 CID
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/cids/JSON"
        cid_response = session.get(cid_url, timeout=10)
        cid_response.raise_for_status()
        cid_data = cid_response.json()
        cid = cid_data['IdentifierList']['CID'][0] if 'IdentifierList' in cid_data else None
        if not cid:
            raise ValueError("未找到 CID")
        PubChem_CID.append(str(cid))
        print(f"compound_id={query}, PubChem CID={cid}")

        # Step 2: 获取属性（分子量和分子式）
        property_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularWeight,MolecularFormula/JSON"
        prop_response = session.get(property_url, timeout=10)
        prop_response.raise_for_status()
        prop_data = prop_response.json()
        props = prop_data['PropertyTable']['Properties'][0]

        mw = props.get('MolecularWeight', 'N/A')
        formula = props.get('MolecularFormula', 'N/A')
        molecular_weight.append(mw)
        molecular_formula.append(formula)
        print(f"Molecular Weight: {mw}")
        print(f"Molecular Formula: {formula}")

        results.append({
            'Query': query,
            'PubChem CID': cid,
            'Molecular Weight': mw,
            'Molecular Formula': formula
        })

        end = time.time()
        elapsed_time = end - start
        total += elapsed_time
        count += 1
        print(f"处理 {query} 完成，耗时: {elapsed_time:.2f} 秒, 处理了：{count}个，共用了{total:.2f}秒")
        time.sleep(0.5)  # API 请求间隔，避免触发限速

    except Exception as e:
        print(f"处理 {query} 失败: {str(e)}")
        error_cid.append(query)

print(f"成功处理 {count} 个化合物, 共用了 {total:.2f} 秒")
df = pd.DataFrame(results)
df.to_csv("details.csv", index=False)


# 打印错误列表
if error_cid:
    print("以下化合物处理失败：")
    for cid in error_cid:
        print(f"  - {cid}")
else:
    print("所有化合物处理成功！")