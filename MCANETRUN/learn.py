from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')

local_path_model = "prot_t5_xl_uniref50/"
tokenizer = T5Tokenizer.from_pretrained(local_path_model)
model = T5EncoderModel.from_pretrained(local_path_model).to(device)
model.eval()

sequence_examples = ["PRTEINO", "SEQWENCE"]
print(f"sequence 0 {len(sequence_examples[0])}")
print(f"sequence 1 {len(sequence_examples[1])}")
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

# extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7])
emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")
# do the same for the second ([1,:]) sequence in the batch while taking into account different sequence lengths ([1,:8])
emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

print(f"Shape of per-protein embedding of first sequences: {emb_0_per_protein.shape}")

