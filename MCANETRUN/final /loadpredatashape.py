import torch

ligands_davis = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/ligands_davis.pt"
ligands_kiba = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/ligands_kiba.pt"
protein_davis = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/protein_davis.pt"
protein_kiba = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/protein_kiba.pt"

ligands = torch.load(ligands_davis)
protiens = torch.load(protein_davis)

ligand = torch.load(ligands_kiba)
protein = torch.load(protein_kiba)

print("Ligands davis",ligands.shape)
print("Protein davis",protiens.shape)
print("Ligands kiba",ligand.shape)
print("Protein kiba",protein.shape)