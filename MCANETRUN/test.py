with open('/data/coding/Davis.txt', 'r') as f:
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

proteins =list(proteins)
# for protein in proteins:
#     print(protein)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')
import re

# Load model directly
from transformers import T5Tokenizer, T5EncoderModel

# local_path_model = "prot_t5_xl_uniref50/"
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
model.eval()

sequence_examples = proteins
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest", return_tensors="pt", truncation=True,max_length=512)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

print(f"Shape of embeddings: {embedding_repr.last_hidden_state.shape}")

# extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7])
emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")
# do the same for the second ([1,:]) sequence in the batch while taking into account different sequence lengths ([1,:8])
emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

print(f"Shape of per-protein embedding of first sequences: {emb_0_per_protein.shape}")