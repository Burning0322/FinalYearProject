import os
import pandas as pd

# 2D 文件夹路径
folder_path = "2D"  # 替换为您的 2D 文件夹路径

# 提取 2D 文件夹中所有 SDF 文件的 CID
sdf_cids = []
for filename in os.listdir(folder_path):
    if filename.startswith("Structure2D_COMPOUND_CID_") and filename.endswith(".sdf"):
        # 从文件名中提取 CID
        cid = filename.replace("Structure2D_COMPOUND_CID_", "").replace(".sdf", "")
        sdf_cids.append(cid)

# 读取 kiba_data.csv
df = pd.read_csv("kiba_data.csv")

# 提取 kiba_data.csv 中的 compound_id
kiba_cids = df['compound_id'].astype(str).tolist()  # 转换为字符串以确保类型一致

# 统计数量
print(f"2D 文件夹中的 SDF 文件数量：{len(sdf_cids)}")
print(f"kiba_data.csv 中的记录数量：{len(kiba_cids)}")

# 找出多余的 CID（在 sdf_cids 中但不在 kiba_cids 中）
extra_cids = set(sdf_cids) - set(kiba_cids)

# 打印多余的 CID 和对应的文件名
print("\n多余的 SDF 文件（不在 kiba_data.csv 中）：")
for cid in extra_cids:
    filename = f"Structure2D_COMPOUND_CID_{cid}.sdf"
    print(f"CID={cid}, 文件名={filename}")

# 找出缺失的 CID（在 kiba_cids 中但不在 sdf_cids 中）
missing_cids = set(kiba_cids) - set(sdf_cids)
print("\n缺失的 SDF 文件（在 kiba_data.csv 中但不在 2D 文件夹中）：")
for cid in missing_cids:
    filename = f"Structure2D_COMPOUND_CID_{cid}.sdf"
    print(f"CID={cid}, 文件名={filename}")