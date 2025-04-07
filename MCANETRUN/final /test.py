import torch
from transformers import T5Tokenizer, T5EncoderModel
import re

# 设置模型 & tokenizer
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "/Volumes/PASSPORT/FinalYearProject/Rostlab:prot_t5_xl_uniref50/"

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5EncoderModel.from_pretrained(model_name).to(device)
model.eval()

# 假设有两条蛋白质序列（不同长度）
sequences = [
    "MSTNPKPQRK",         # 10个氨基酸
    "MEEPQSDPSVEPPLSQETF" # 18个氨基酸
]

# 替换非法字符并加空格（ProtT5 要求输入是用空格隔开的）
processed_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

# 编码为 tokens（填充到 max_length=20）
inputs = tokenizer(processed_sequences, return_tensors="pt",
                   padding="max_length", truncation=True, max_length=20)

# 转到 GPU（或 CPU）
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# 获取输出和 attention_mask
last_hidden = outputs.last_hidden_state            # [batch, seq_len, hidden_size]
attention_mask = inputs["attention_mask"]          # [batch, seq_len]

# Masked mean pooling
mask = attention_mask.unsqueeze(-1).expand_as(last_hidden)      # [batch, seq_len, hidden_size]
masked_sum = (last_hidden * mask).sum(dim=1)                    # [batch, hidden_size]
masked_mean = masked_sum / mask.sum(dim=1)                      # [batch, hidden_size]

# 打印结果
print("Protein Embedding Shape:", masked_mean.shape)
print("Embedding for sequence 1:", masked_mean[0][:10])  # 显示前10个数
