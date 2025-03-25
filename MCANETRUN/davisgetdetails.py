import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import time

# 设置 requests 重试策略
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# 读取 Davis.txt 文件
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

# 从数据中获取所有独特的化合物 ID，并按升序排序
compound_ids = sorted(set(d['compound_id'] for d in data))
print(f"待处理化合物数量: {len(compound_ids)}")

# 初始化列表
PubChem_CID = []
molecular_weight = []
molecular_formula = []
error_cid = []
count = 0
total = 0

# 查询 PubChem API 获取化合物详细信息
def get_pubchem_data(compound_id):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{compound_id}/property/MolecularWeight,MolecularFormula/JSON"
    try:
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            # 检查响应是否包含所需数据
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                properties = data['PropertyTable']['Properties'][0]
                mol_weight = properties.get('MolecularWeight', None)
                mol_formula = properties.get('MolecularFormula', None)
                return mol_weight, mol_formula
            else:
                print(f"没有返回数据: {compound_id}")
                return None, None
        else:
            print(f"请求失败: {compound_id}，状态码: {response.status_code}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {compound_id} 错误: {str(e)}")
        return None, None

# 处理每个化合物
for compound_id in compound_ids:
    try:
        start = time.time()
        print(f"处理化合物: {compound_id}")

        # 获取化合物的分子量和分子式
        mol_weight, mol_formula = get_pubchem_data(compound_id)

        if mol_weight and mol_formula:
            print(f"compound_id={compound_id}, PubChem CID={compound_id}, Molecular Weight={mol_weight}, Molecular Formula={mol_formula}")
            PubChem_CID.append(compound_id)  # 使用 compound_id 作为 PubChem CID
            molecular_weight.append(mol_weight)
            molecular_formula.append(mol_formula)
        else:
            print(f"没有找到 {compound_id} 的相关信息")
            error_cid.append(compound_id)

        end = time.time()  # 记录结束时间
        elapsed_time = end - start  # 计算耗时
        total = total + elapsed_time
        count += 1
        print(f"处理 {compound_id} 完成，耗时: {elapsed_time:.2f} 秒, 处理了：{count}个，共用了{total}")
        time.sleep(1)  # 每次请求后等待 1 秒，以避免请求过于频繁

    except Exception as e:
        print(f"处理 {compound_id} 失败: {str(e)}")
        error_cid.append(compound_id)

# 保存结果到 CSV 文件
results = {
    'compound_id': compound_ids,  # 使用 compound_ids 列表作为查询列
    'PubChem CID': PubChem_CID,
    'Molecular Weight': molecular_weight,
    'Molecular Formula': molecular_formula
}
df = pd.DataFrame(results)
df.to_csv('davis_details.csv', index=False)
print("已将结果保存到 davis_details.csv")

# 打印错误列表
if error_cid:
    print("以下化合物处理失败：")
    for cid in error_cid:
        print(f"  - {cid}")
else:
    print("所有化合物处理成功！")
