import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import numpy as np
import math

# Set seed and device
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
learning_rate = 0.001
epochs = 100
batch_size = 64
k_folds = 5
threshold = 0.6
drug_kernel = [4, 6, 8]
protein_kernel = [4, 8, 12]
drug_max_length = 94
protein_max_length = 1000
drug_afterCNN = drug_max_length - sum(drug_kernel) + 3
protein_afterCNN = protein_max_length - sum(protein_kernel) + 3

# Load embeddings
drug_embedding = torch.load("/kaggle/input/ligands-n-protein/ligands_davis.pt").to(device)
protein_embedding = torch.load("/kaggle/input/ligands-n-protein/protein_davis.pt").to(device)

drug_dim = drug_embedding.shape[2]
protein_dim = protein_embedding.shape[2]
conv = 40
attention_dim = conv * 4
mix_attention_head = 5

class SharedMultiheadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, drug_feat, protein_feat):
        B, L_d, _ = drug_feat.size()
        _, L_p, _ = protein_feat.size()

        Q_d = self.W_q(drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
        K_p = self.W_k(protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)
        V_p = self.W_v(protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)

        Q_p = self.W_q(protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)
        K_d = self.W_k(drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
        V_d = self.W_v(drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output_d = torch.matmul(torch.softmax(torch.matmul(Q_d, K_p.transpose(-2, -1)) / self.scale, dim=-1), V_p)
        attn_output_p = torch.matmul(torch.softmax(torch.matmul(Q_p, K_d.transpose(-2, -1)) / self.scale, dim=-1), V_d)

        attn_output_d = attn_output_d.transpose(1, 2).contiguous().view(B, L_d, self.embed_dim)
        attn_output_p = attn_output_p.transpose(1, 2).contiguous().view(B, L_p, self.embed_dim)

        out_d = self.out_proj(attn_output_d)
        out_p = self.out_proj(attn_output_p)

        return 0.5 * drug_feat + 0.5 * out_d, 0.5 * protein_feat + 0.5 * out_p

class Model(nn.Module):
    def __init__(self, drug_embedding, protein_embedding):
        super().__init__()
        self.drug_embedding = nn.Parameter(drug_embedding, requires_grad=True)
        self.protein_embedding = nn.Parameter(protein_embedding, requires_grad=True)

        self.drug_CNN = nn.Sequential(
            nn.Conv1d(drug_dim, conv, drug_kernel[0]),
            nn.BatchNorm1d(conv),  # Adding BatchNorm
            nn.ReLU(),
            nn.Conv1d(conv, conv * 2, drug_kernel[1]),
            nn.BatchNorm1d(conv * 2),  # Adding BatchNorm
            nn.ReLU(),
            nn.Conv1d(conv * 2, conv * 4, drug_kernel[2]),
            nn.BatchNorm1d(conv * 4),  # Adding BatchNorm
            nn.ReLU(),
        )
        self.protein_CNN = nn.Sequential(
            nn.Conv1d(protein_dim, conv, protein_kernel[0]),
            nn.BatchNorm1d(conv),  # Adding BatchNorm
            nn.ReLU(),
            nn.Conv1d(conv, conv * 2, protein_kernel[1]),
            nn.BatchNorm1d(conv * 2),  # Adding BatchNorm
            nn.ReLU(),
            nn.Conv1d(conv * 2, conv * 4, protein_kernel[2]),
            nn.BatchNorm1d(conv * 4),  # Adding BatchNorm
            nn.ReLU(),
        )
        self.drug_pool = nn.MaxPool1d(drug_afterCNN)
        self.protein_pool = nn.MaxPool1d(protein_afterCNN)
        self.attention = SharedMultiheadCrossAttention(attention_dim, mix_attention_head)

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(conv * 8, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, drug_idx, protein_idx):
        drug = self.drug_embedding[drug_idx].permute(0, 2, 1)
        protein = self.protein_embedding[protein_idx].permute(0, 2, 1)
        drug_feat = self.drug_CNN(drug).permute(0, 2, 1)
        protein_feat = self.protein_CNN(protein).permute(0, 2, 1)
        drug_att, protein_att = self.attention(drug_feat, protein_feat)
        drug_att = self.drug_pool(drug_att.permute(0, 2, 1)).squeeze(2)
        protein_att = self.protein_pool(protein_att.permute(0, 2, 1)).squeeze(2)
        return self.fc(torch.cat([drug_att, protein_att], dim=1))

class Dataset(Dataset):
    def __init__(self, file_path):
        self.smiles2idx, self.protein2idx, self.data = {}, {}, []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 4)
                if len(parts) == 5:
                    _, _, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
                    sequence, label = rest.rsplit(' ', 1)
                    if smiles not in self.smiles2idx:
                        self.smiles2idx[smiles] = len(self.smiles2idx)
                    if sequence not in self.protein2idx:
                        self.protein2idx[sequence] = len(self.protein2idx)
                    self.data.append((self.smiles2idx[smiles], self.protein2idx[sequence], int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d, p, y = self.data[idx]
        return {'drug_idx': torch.tensor(d), 'protein_idx': torch.tensor(p), 'label': torch.tensor(y)}

def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
            out = model(d, p)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = (prob >= threshold).long()
            y_true += y.tolist()
            y_pred += pred.tolist()
            y_prob += prob.tolist()
    acc = 100 * (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    return acc, precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred), roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)

# Load dataset and prepare K-Fold
dataset = Dataset("/kaggle/input/davis-n-kiba/Davis.txt")
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
all_indices = list(range(len(dataset)))
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
    print(f"\n=== Fold {fold + 1}/{k_folds} ===")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = Model(drug_embedding, protein_embedding).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    labels = [dataset[i]['label'].item() for i in train_idx]
    pos = sum(1 for y in labels if y == 1)
    neg = len(labels) - pos
    ratio = neg / pos
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, ratio], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)

    best_val_loss = float('inf')
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            out = model(d, p)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
                out = model(d, p)
                val_loss += criterion(out, y).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 10:
            break

    acc, pre, rec, f1, auc, prc = evaluate(model, val_loader)
    fold_results.append((acc, pre, rec, f1, auc, prc))

# Average metrics across folds
avg = np.mean(fold_results, axis=0)
print(f"\n=== Average over {k_folds} folds ===")
print(f"Accuracy: {avg[0]:.2f}%, Precision: {avg[1]:.4f}, Recall: {avg[2]:.4f}, F1: {avg[3]:.4f}, AUC: {avg[4]:.4f}, PRC: {avg[5]:.4f}")