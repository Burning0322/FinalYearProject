import torch
from transformers import T5Tokenizer, T5EncoderModel
import re
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

model_name = "Rostlab/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5EncoderModel.from_pretrained(model_name)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = torch.nn.DataParallel(model)

model = model.to(device)
model.eval()

# 读取 Davis 数据集
with open('/kaggle/input/davis-n-kiba/KIBA.txt', 'r') as f:
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

proteins = list(set([d['sequence'] for d in data]))
print(f"Number of unique proteins: {len(proteins)}")

sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in proteins]

batch_size = 16
max_length = 1000
embeddings = []

for i in tqdm(range(0, len(sequence_examples), batch_size), desc="Processing batches"):
    batch_seqs = sequence_examples[i:i + batch_size]

    inputs = tokenizer(batch_seqs, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state

    embeddings.append(last_hidden.cpu())

    del inputs, outputs, last_hidden
    torch.cuda.empty_cache()

embeddings = torch.cat(embeddings, dim=0)
print(f"Final embedding shape: {embeddings.shape}")

# 检查嵌入有效性
if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
    print("Warning: Embeddings contain NaN or Inf values!")
else:
    print("Embeddings look good!")

# 保存
torch.save(embeddings, "protein_kiba.pt")
print("✅ Saved: protein_kiba.pt")