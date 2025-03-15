import torch
import os
import re
from torch.nn.parallel import DataParallel

# 设置环境变量以减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


with open('Davis.txt', 'r') as f:
    lines = f.readlines()

# with open('KIBA.txt', 'r') as f:
#     lines = f.readlines()

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

print(f"Number of unique proteins: {len(proteins)}")


sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in proteins]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from transformers import T5Tokenizer, T5EncoderModel

local_path_model = "Rostlab/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(local_path_model, legacy=False)
model = T5EncoderModel.from_pretrained(local_path_model)

# 使用多个GPU 因为Memory不够
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)

model = model.to(device)
model.eval()

batch_size = 32
embeddings = []
max_length = 1000

for i in range(0, len(sequence_examples), batch_size):
    batch_sequences = sequence_examples[i:i + batch_size]

    inputs = tokenizer(batch_sequences, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(**inputs)


    batch_embeddings = outputs.last_hidden_state
    embeddings.append(batch_embeddings.cpu())

    del inputs, outputs, batch_embeddings
    torch.cuda.empty_cache()

embeddings = torch.cat(embeddings, dim=0)
print(f"Embeddings shape: {embeddings.shape}")

torch.save(embeddings, "protein_davis.pt")