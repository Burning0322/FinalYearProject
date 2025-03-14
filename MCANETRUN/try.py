import torch
from transformers import T5Tokenizer, T5EncoderModel
import re

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

proteins = set()
for protein in data:
    proteins.add(protein['sequence'])

proteins = list(proteins)

# 检查 GPU 是否可用
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device= torch.device("mps")
print(device)

# 加载模型和分词器
# local_path_model = "autodl-tmp/prot_t5_xl_uniref50"
# tokenizer = T5Tokenizer.from_pretrained(local_path_model)
# model = T5EncoderModel.from_pretrained(local_path_model).to(device)
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
model.eval()

# 蛋白质序列列表
sequence_examples = proteins

# 将序列中的稀有氨基酸替换为 X，并在氨基酸之间添加空格
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# 分批处理
batch_size = 4  # 根据 GPU 内存调整批次大小
max_length = 512  # 最大序列长度
all_embeddings = []

for i in range(0, len(sequence_examples), batch_size):
    batch_sequences = sequence_examples[i:i + batch_size]

    # 分词并生成输入
    ids = tokenizer.batch_encode_plus(
        batch_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)

    # 生成嵌入
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # 提取每个蛋白质的嵌入（取均值）
    for j in range(len(batch_sequences)):
        seq_embedding = embedding_repr.last_hidden_state[j]  # 形状: (序列长度, 1024)
        # 去除填充和特殊标记
        seq_length = attention_mask[j].sum().item()  # 实际序列长度
        seq_embedding = seq_embedding[:seq_length]  # 去除填充部分
        # 计算整个蛋白质的嵌入（均值池化）
        protein_embedding = seq_embedding.mean(dim=0)  # 形状: (1024)
        all_embeddings.append(protein_embedding.cpu())  # 将结果移回 CPU

    # 释放内存
    del input_ids, attention_mask, embedding_repr
    torch.cuda.empty_cache()  # 释放 GPU 内存

print(f"Number of unique proteins: {len(proteins)}")
print(f"Total data entries: {len(data)}")

# 将所有嵌入转换为张量
all_embeddings = torch.stack(all_embeddings)  # 形状: (蛋白质数量, 1024)
print(f"Shape of all protein embeddings: {all_embeddings.shape}")

torch.save(all_embeddings, 'protein_davis.pt')