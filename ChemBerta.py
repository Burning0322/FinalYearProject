import json

# from transformers import AutoTokenizer, AutoModelForMaskedLM

local_model_path = "/Users/renhonglow/Desktop/FYP/DTI/ChemBERTa-10M-MTR"

# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# model = AutoModelForMaskedLM.from_pretrained(local_model_path)

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

from transformers import AutoTokenizer,RobertaModel
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = RobertaModel.from_pretrained(local_model_path).to(device)
model.eval()

# import os
# filepath= os.path.abspath('davis/ligands_iso.txt')

# with open(filepath,'r') as file:
#     smile_dict =json.load(file)

import os
filepath= os.path.abspath('kiba/ligands_iso.txt')

with open(filepath,'r') as file:
    smile_dict =json.load(file)

# 计算最大的smiles长度
# i=0
# max_length=0
# for smile in smile_dict.values():
#     print(f"Smiles {i}: {smile}，length: {len(smile)}")
#     i=i+1
#     current_length = len(smile)
#     if current_length > max_length:
#         max_length = current_length

# print(f"Max SMILES length: {max_length}")

smiles = list()
for smile in smile_dict.values():
    smiles.append(smile)

input = tokenizer(smiles,padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
with torch.no_grad():
    output = model(**input)
    print(output.last_hidden_state.shape)

features = output.last_hidden_state.cpu()
# torch.save(features, 'ligands_davis.pt')
torch.save(features, 'ligands_kiba.pt')