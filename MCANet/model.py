import torch
import torch.nn as nn

learning_rate = 1e-4
epoch = 200
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
    def __init__(self):
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
