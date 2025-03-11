from transformers import AutoTokenizer, AutoModel
import torch
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')

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

# print(len(data))

smiles = set()
for smile in data:
    smiles.add(smile['smiles'])

smiles = list(smiles)

# print(len(smiles))

from transformers import AutoTokenizer, AutoModel
local_path = "/Volumes/PASSPORT/FinalYearProject/ChemBERTa-77M-MLM"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModel.from_pretrained(local_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
# model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
model.eval()

input = tokenizer(smiles,padding=True, truncation=True,max_length=512, return_tensors="pt").to(device)
with torch.no_grad():
    output = model(**input)

print(output.last_hidden_state.shape)
print(output.pooler_output.shape)

# 处理 SMILES
# smiles_list = list(set([d['smiles'] for d in data]))
# print(f"Number of unique SMILES: {len(smiles_list)}")
# chem_encodings = chem_tokenizer(smiles_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
# chem_input_ids = chem_encodings['input_ids'].to(device)
# chem_attention_mask = chem_encodings['attention_mask'].to(device)

# 生成药物嵌入
# with torch.no_grad():
#     chem_outputs = chem_model(input_ids=chem_input_ids, attention_mask=chem_attention_mask)
#     chem_embeddings = chem_outputs.last_hidden_state.mean(dim=1)  # (num_samples, 768)
# print(f"Shape of drug embeddings: {chem_embeddings.shape}")
#
# molecular_vector = chem_outputs.pooler_output
# print(molecular_vector.shape)
#
# print(chem_outputs.keys())

