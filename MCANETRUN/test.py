from Bio.PDB import PDBParser
import nglview as nv

parser = PDBParser()
structure = parser.get_structure("molecule", "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/pdb_files/1A0N.pdb")
view = nv.show_biopython(structure)
view