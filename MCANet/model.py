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
conv = 40
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
        )

    def forward(self, x):

        return x
