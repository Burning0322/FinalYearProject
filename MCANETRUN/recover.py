import requests
import csv_2_mysql
from tqdm import tqdm  # Import tqdm for the progress bar
import time

# Initialize the list for storing error compound IDs
error_cid = []

# Create an empty list to store PubChem data
pubchem_data = []

# Define the fields to retrieve from PubChem
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

# Open the input file
# Open the input file (assuming it's a CSV)
with open('details.csv', 'r') as f:
    lines = f.readlines()

data = []
for line in lines[1:]:
    parts = line.strip().split(",")
    cid = parts[1].strip()
    data.append(cid)

print(len(data))

# Define function to query PubChem API with retries
def query_pubchem_with_retry(cid, retries=3, delay=2):
    properties = ",".join(fields)
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{properties}/JSON'
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
            return response.json()  # Return the response JSON if successful
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for CID {cid}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                print(f"Failed to query CID {cid} after {retries} attempts")
                return None


# Query PubChem for each compound ID and store results or errors
for compound in tqdm(data, desc="Querying PubChem CIDs", unit="CID"):
    result = query_pubchem_with_retry(compound)
    if result:
        # If successful, extract relevant data and add to pubchem_data list
        compound_info = result.get('PropertyTable', {}).get('Properties', [])[0]
        if compound_info:
            compound_data = {
                'compound_id': compound,
                'molecular_formula': compound_info.get('MolecularFormula', 'N/A'),
                'molecular_weight': compound_info.get('MolecularWeight', 'N/A'),
                'smiles': compound_info.get('SMILES', 'N/A'),
                'canonical_smiles': compound_info.get('CanonicalSMILES', 'N/A'),
                'isomeric_smiles': compound_info.get('IsomericSMILES', 'N/A'),
                'iupac_name': compound_info.get('IUPACName', 'N/A'),
                'inchi': compound_info.get('InChI', 'N/A'),
                'inchi_key': compound_info.get('InChIKey', 'N/A'),
                'xlogp': compound_info.get('XLogP', 'N/A'),
                'exact_mass': compound_info.get('ExactMass', 'N/A'),
                'monoisotopic_mass': compound_info.get('MonoisotopicMass', 'N/A'),
                'tpsa': compound_info.get('TPSA', 'N/A'),
                'complexity': compound_info.get('Complexity', 'N/A'),
                'charge': compound_info.get('Charge', 'N/A'),
                'h_bond_donor_count': compound_info.get('HBondDonorCount', 'N/A'),
                'h_bond_acceptor_count': compound_info.get('HBondAcceptorCount', 'N/A'),
                'rotatable_bond_count': compound_info.get('RotatableBondCount', 'N/A'),
                'heavy_atom_count': compound_info.get('HeavyAtomCount', 'N/A'),
                'isotope_atom_count': compound_info.get('IsotopeAtomCount', 'N/A'),
                'defined_atom_stereo_count': compound_info.get('DefinedAtomStereoCount', 'N/A'),
                'undefined_atom_stereo_count': compound_info.get('UndefinedAtomStereoCount', 'N/A'),
                'defined_bond_stereo_count': compound_info.get('DefinedBondStereoCount', 'N/A'),
                'undefined_bond_stereo_count': compound_info.get('UndefinedBondStereoCount', 'N/A'),
                'covalent_unit_count': compound_info.get('CovalentUnitCount', 'N/A'),
                'conformer_count_3d': compound_info.get('ConformerCount3D', 'N/A'),
                'volume_3d': compound_info.get('Volume3D', 'N/A'),
                'x_steric_quadrupole_3d': compound_info.get('XStericQuadrupole3D', 'N/A'),
                'y_steric_quadrupole_3d': compound_info.get('YStericQuadrupole3D', 'N/A'),
                'z_steric_quadrupole_3d': compound_info.get('ZStericQuadrupole3D', 'N/A'),
                'feature_acceptor_count_3d': compound_info.get('FeatureAcceptorCount3D', 'N/A'),
                'feature_donor_count_3d': compound_info.get('FeatureDonorCount3D', 'N/A'),
                'feature_anion_count_3d': compound_info.get('FeatureAnionCount3D', 'N/A'),
                'feature_cation_count_3d': compound_info.get('FeatureCationCount3D', 'N/A'),
                'feature_ring_count_3d': compound_info.get('FeatureRingCount3D', 'N/A'),
                'feature_hydrophobe_count_3d': compound_info.get('FeatureHydrophobeCount3D', 'N/A'),
                'effective_rotor_count_3d': compound_info.get('EffectiveRotorCount3D', 'N/A'),
                'fingerprint_2d': compound_info.get('Fingerprint2D', 'N/A')
            }
            pubchem_data.append(compound_data)
        print(compound_data)
    else:
        # If the query failed, add the compound_id to error_cid list
        error_cid.append(compound)

    # Optional: Add a small delay to avoid server overload
    time.sleep(1)  # Adjust this delay as needed

# Save the successful results to a CSV file
with open('kiba_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['compound_id', 'molecular_formula', 'molecular_weight', 'smiles', 'canonical_smiles',
                  'isomeric_smiles',
                  'iupac_name', 'inchi', 'inchi_key', 'xlogp', 'exact_mass', 'monoisotopic_mass', 'tpsa', 'complexity',
                  'charge', 'h_bond_donor_count', 'h_bond_acceptor_count', 'rotatable_bond_count', 'heavy_atom_count',
                  'isotope_atom_count', 'defined_atom_stereo_count', 'undefined_atom_stereo_count',
                  'defined_bond_stereo_count',
                  'undefined_bond_stereo_count', 'covalent_unit_count', 'conformer_count_3d', 'volume_3d',
                  'x_steric_quadrupole_3d', 'y_steric_quadrupole_3d', 'z_steric_quadrupole_3d',
                  'feature_acceptor_count_3d',
                  'feature_donor_count_3d', 'feature_anion_count_3d', 'feature_cation_count_3d',
                  'feature_ring_count_3d',
                  'feature_hydrophobe_count_3d', 'effective_rotor_count_3d', 'fingerprint_2d']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(pubchem_data)

# Print the results and error CIDs
print(f"Successfully queried {len(pubchem_data)} compounds.")
print(f"Failed to query {len(error_cid)} compounds.")
print(f"Failed compound CIDs: {error_cid}")