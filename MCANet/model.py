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

drug_embedding = torch.load("/Volumes/PASSPORT/FinalYearProject/ligands_davis.pt")
print(drug_embedding.shape)

protein_embedding = torch.load()

drug_dim = drug_embedding.shape[2]
protein_dim = protein_embedding.shape[2]
conv = 40
attention_dim = conv * 4
mix_attention_head = 5

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

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

    def forward(self, x):

        return x
