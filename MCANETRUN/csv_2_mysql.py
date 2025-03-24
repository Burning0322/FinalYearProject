import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "dti"
}

# 读取 CSV 文件
df = pd.read_csv("kiba_data.csv")

# 打印 CSV 文件的列，检查是否与表结构匹配
print("CSV 文件的列：", df.columns.tolist())

# 将 NaN 替换为 None
df = df.replace([np.nan, 'nan', 'NaN'], None)

# 定义 drug 表的所有列
drug_columns = [
    "compound_id", "molecular_formula", "molecular_weight", "smiles", "canonical_smiles",
    "isomeric_smiles", "iupac_name", "inchi", "inchi_key", "xlogp", "exact_mass",
    "monoisotopic_mass", "tpsa", "complexity", "charge", "h_bond_donor_count",
    "h_bond_acceptor_count", "rotatable_bond_count", "heavy_atom_count",
    "isotope_atom_count", "defined_atom_stereo_count", "undefined_atom_stereo_count",
    "defined_bond_stereo_count", "undefined_bond_stereo_count", "covalent_unit_count",
    "conformer_count_3d", "volume_3d", "x_steric_quadrupole_3d", "y_steric_quadrupole_3d",
    "z_steric_quadrupole_3d", "feature_acceptor_count_3d", "feature_donor_count_3d",
    "feature_anion_count_3d", "feature_cation_count_3d", "feature_ring_count_3d",
    "feature_hydrophobe_count_3d", "effective_rotor_count_3d", "fingerprint_2d"
]

# 检查 CSV 文件的列数是否与 drug 表匹配
if len(df.columns) != len(drug_columns):
    print(f"警告：CSV 文件有 {len(df.columns)} 列，但 drug 表需要 {len(drug_columns)} 列")
    # 为缺失的列填充 None
    for col in drug_columns:
        if col not in df.columns:
            df[col] = None

# 确保 DataFrame 的列顺序与 drug 表一致
df = df[drug_columns]

# 连接到 MySQL
try:
    connection = mysql.connector.connect(**db_config)
    if connection.is_connected():
        cursor = connection.cursor()
        print("成功连接到 MySQL 数据库")

        # 插入数据的 SQL 语句
        insert_query = """
        INSERT INTO drug (
            compound_id, molecular_formula, molecular_weight, smiles, canonical_smiles, 
            isomeric_smiles, iupac_name, inchi, inchi_key, xlogp, exact_mass, 
            monoisotopic_mass, tpsa, complexity, charge, h_bond_donor_count, 
            h_bond_acceptor_count, rotatable_bond_count, heavy_atom_count, 
            isotope_atom_count, defined_atom_stereo_count, undefined_atom_stereo_count, 
            defined_bond_stereo_count, undefined_bond_stereo_count, covalent_unit_count, 
            conformer_count_3d, volume_3d, x_steric_quadrupole_3d, y_steric_quadrupole_3d, 
            z_steric_quadrupole_3d, feature_acceptor_count_3d, feature_donor_count_3d, 
            feature_anion_count_3d, feature_cation_count_3d, feature_ring_count_3d, 
            feature_hydrophobe_count_3d, effective_rotor_count_3d, fingerprint_2d
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # 将 DataFrame 转换为元组列表，并进行批量插入
        data = [tuple(row) for _, row in df.iterrows()]
        cursor.executemany(insert_query, data)

        # 提交事务
        connection.commit()
        print(f"成功插入 {cursor.rowcount} 条记录")

except Error as e:
    print(f"错误: {e}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL 连接已关闭")