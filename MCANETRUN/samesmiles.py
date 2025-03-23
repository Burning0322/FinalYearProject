from rdkit import Chem

smiles1 = "CC1(CCC2(CCC3(C(=CCC4C3(CCC5C4(CC(C6C5(COC(=O)C7=CC(=C(C(=C7C8=C(C(=C(C=C8C(=O)O6)O)O)O)O)O)O)CO)O)C)C)C2C1)C)C(=O)OC9C(C(C(C(O9)CO)O)O)O)C"
smiles2 = "CC1(CCC2(CCC3(C(=CCC4C3(CCC5C4(CC(C6C5(COC(=O)C7=CC(=C(C(=C7C8=C(C(=C(C=C8C(=O)O6)O)O)O)O)O)O)COC(=O)C9=CC(=C(C(=C9)O)O)O)O)C)C)C2C1)C)C(=O)OC1C(C(C(C(O1)CO)O)OC(=O)C1=CC(=C(C(=C1)O)O)O)O)C"

mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)

are_equal = mol1 is not None and mol2 is not None and Chem.MolToInchi(mol1) == Chem.MolToInchi(mol2)

print("是否相同分子结构：", are_equal)
