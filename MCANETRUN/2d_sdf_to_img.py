import os
from rdkit import Chem
from rdkit.Chem import Draw

sdf_file = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/2D/Structure2D_COMPOUND_CID_1392.sdf"

# 检查文件是否存在
if not os.path.exists(sdf_file):
    raise FileNotFoundError(f"File not found: {sdf_file}")

# 加载分子
supplier = Chem.SDMolSupplier(sdf_file)

# 尝试提取第一个有效分子
mol = next((m for m in supplier if m is not None), None)

if mol:
    img = Draw.MolToImage(mol)
    img.show()
else:
    print("⚠️ 无法从 SDF 文件中读取到有效分子。")
