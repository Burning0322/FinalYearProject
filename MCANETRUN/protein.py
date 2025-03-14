import torch
with open('/kaggle/input/davis-n-kiba/Davis.txt', 'r') as f:
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

proteins = set()
for protein in data:
    proteins.add(protein['sequence'])

proteins = list(proteins)

print(len(proteins))

sequence_examples = proteins

import re
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

from transformers import T5Tokenizer, T5EncoderModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
model.eval()

input = tokenizer(sequence_examples, return_tensors="pt", padding=True, truncation=True,max_length=1000).to(device)

with torch.no_grad():
    outputs = model(**input)
