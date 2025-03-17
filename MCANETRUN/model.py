import torch
import torch.nn as nn
import time

learning_rate = 0.001
epochs = 200
batch_size = 32
drug_max_length = 100
protein_max_length = 1000
drug_kernel = [4,6,8]
protein_kernel = [4,8,12]
drug_afterCNN = drug_max_length - drug_kernel[0] - drug_kernel[1] - drug_kernel[2] + 3
protein_afterCNN = protein_max_length - protein_kernel[0] - protein_kernel[1] - protein_kernel[2] + 3

drug_embedding = torch.load("/Volumes/PASSPORT/FinalYearProject/MCANETRUN/ligands_davis.pt")
print(drug_embedding.shape)

protein_embedding = torch.load("/Volumes/PASSPORT/FinalYearProject/MCANETRUN/protein_davis.pt")
print(protein_embedding.shape)

drug_dim = drug_embedding.shape[2]
protein_dim = protein_embedding.shape[2]
conv = 40
attention_dim = conv * 4
mix_attention_head = 5

class Model(nn.Module):
    def __init__(self,drug_embedding,protein_embedding):
        super(Model, self).__init__()

        self.drug_embedding = drug_embedding
        self.protein_embedding = protein_embedding

        self.drug_CNN = nn.Sequential(
            nn.Conv1d(in_channels=drug_dim,out_channels=conv,kernel_size=drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv,out_channels=conv*2,kernel_size=drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv*2,out_channels=conv*4,kernel_size=drug_kernel[2]),
            nn.ReLU(),
        )
        self.drug_CNN_max_pool = nn.MaxPool1d(drug_afterCNN)

        self.protein_CNN = nn.Sequential(
            nn.Conv1d(in_channels=protein_dim,out_channels=conv,kernel_size=protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv,out_channels=conv*2,kernel_size=protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv*2,out_channels=conv*4,kernel_size=protein_kernel[2]),
            nn.ReLU(),
        )
        self.protein_CNN_max_pool = nn.MaxPool1d(protein_afterCNN)

        self.mix_attention_layer = nn.MultiheadAttention(
            attention_dim, mix_attention_head)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

        self.leaky_relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(conv * 8 , 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug_idx, protein_idx):

        #获取预训练的嵌入
        drug_embed = self.drug_embedding[drug_idx]
        protein_embed = self.protein_embedding[protein_idx]

        #转换成CNN的输入
        drug_embed = drug_embed.permute(0,2,1)
        protein_embed = protein_embed.permute(0,2,1)

        #CNN处理
        drug_conv = self.drug_CNN(drug_embed)
        protein_conv = self.protein_CNN(protein_embed)

        #转至注意力层输入
        drug_QKV = drug_conv.permute(2,0,1)
        protein_QKV = protein_conv.permute(2,0,1)

        # 交叉注意力
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)

        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drug_conv = drug_conv * 0.5 + drug_att * 0.5
        protein_conv = protein_conv * 0.5 + protein_att * 0.5

        drug_conv = self.drug_CNN_max_pool(drug_conv).squeeze(2)
        protein_conv = self.protein_CNN_max_pool(protein_conv).squeeze(2)

        pair = torch.cat([drug_conv, protein_conv], dim=1)
        pair = self.dropout1(pair)

        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict
import torch
from torch.utils.data import Dataset, DataLoader

drug_embedding = torch.load("/Volumes/PASSPORT/FinalYearProject/MCANETRUN/ligands_davis.pt")
protein_embedding = torch.load("/Volumes/PASSPORT/FinalYearProject/MCANETRUN/protein_davis.pt")


class Dataset(Dataset):
    def __init__(self, file_path, drug_embedding, protein_embedding):
        self.drug_embedding = drug_embedding
        self.protein_embedding = protein_embedding

        self.unique_smiles = []
        self.unique_protein = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        self.data = []
        for line in lines:
            parts = line.strip().split(' ',4)
            if len(parts) == 5:
                compound_id, protein_name, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
                sequence, label = rest.rsplit(' ', 1)

                if smiles not in self.unique_smiles:
                    self.unique_smiles.append(smiles)
                drug_idx = self.unique_smiles.index(smiles)
                if sequence not in self.unique_protein:
                    self.unique_protein.append(sequence)
                protein_idx = self.unique_protein.index(sequence)
                label = int(label)

                self.data.append((drug_idx, protein_idx, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug_idx,protein_idx,label = self.data[idx]

        drug_feature =self.drug_embedding[drug_idx]
        protein_feature = self.protein_embedding[protein_idx]

        return {
            'drug_idx': drug_feature,
            'protein_idx': protein_feature,
            'label': label
        }

train_dataset = Dataset('davis.txt',drug_embedding,protein_embedding)

train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)

print(f"数据集总条数: {len(train_dataset)}")
print(f"独特药物数量: {len(train_dataset.unique_smiles)}")
print(f"独特蛋白质数量: {len(train_dataset.unique_protein)}")

model = Model()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


total_start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()

    for batch in train_loader:
        drug_idx = batch['drug_idx'].to(device)
        protein_idx = batch['protein_idx'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(drug_idx, protein_idx)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time: {epoch_time:.2f} seconds")

total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"Total training time: {total_time:.2f} seconds")