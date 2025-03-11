# with open('davis.txt', 'r') as f:
#     lines = f.readlines()

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

smiles = set()
for smile in data:
    smiles.add(smile['smiles'])

smiles = list(smiles)

import torch
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')

from transformers import AutoTokenizer, RobertaModel

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
model = RobertaModel.from_pretrained("DeepChem/ChemBERTa-10M-MTR").to(device)
model.eval()

input = tokenizer(smiles,padding=True, truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    output = model(**input)

print(output.last_hidden_state.shape)