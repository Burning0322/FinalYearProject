import torch
import torch.nn as nn
import time
from tqdm import tqdm

learning_rate = 0.001
epochs = 200
batch_size = 64
drug_max_length = 94
protein_max_length = 1000
drug_kernel = [4, 6, 8]
protein_kernel = [4, 8, 12]
drug_afterCNN = drug_max_length - drug_kernel[0] - drug_kernel[1] - drug_kernel[2] + 3
protein_afterCNN = protein_max_length - protein_kernel[0] - protein_kernel[1] - protein_kernel[2] + 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("mps")

drug_embedding = torch.load("ligands_davis.pt").to(device)
protein_embedding = torch.load("protein_davis.pt").to(device)

drug_dim = drug_embedding.shape[2]
protein_dim = protein_embedding.shape[2]

conv = 40
attention_dim = conv * 4
mix_attention_head = 5


class Model(nn.Module):
    def __init__(self, drug_embedding, protein_embedding):
        super(Model, self).__init__()

        self.drug_embedding = nn.Parameter(drug_embedding, requires_grad=False)
        self.protein_embedding = nn.Parameter(protein_embedding, requires_grad=False)

        self.drug_CNN = nn.Sequential(
            nn.Conv1d(in_channels=drug_dim, out_channels=conv, kernel_size=drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv, out_channels=conv * 2, kernel_size=drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv * 2, out_channels=conv * 4, kernel_size=drug_kernel[2]),
            nn.ReLU(),
        )
        self.drug_CNN_max_pool = nn.MaxPool1d(drug_afterCNN)

        self.protein_CNN = nn.Sequential(
            nn.Conv1d(in_channels=protein_dim, out_channels=conv, kernel_size=protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv, out_channels=conv * 2, kernel_size=protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv * 2, out_channels=conv * 4, kernel_size=protein_kernel[2]),
            nn.ReLU(),
        )
        self.protein_CNN_max_pool = nn.MaxPool1d(protein_afterCNN)

        self.mix_attention_layer = nn.MultiheadAttention(
            attention_dim, mix_attention_head)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        self.leaky_relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, drug_idx, protein_idx):
        # 获取预训练的嵌入
        drug_embed = self.drug_embedding[drug_idx]
        protein_embed = self.protein_embedding[protein_idx]

        # 转换成CNN的输入
        drug_embed = drug_embed.permute(0, 2, 1)
        protein_embed = protein_embed.permute(0, 2, 1)

        # CNN处理
        drug_conv = self.drug_CNN(drug_embed)
        protein_conv = self.protein_CNN(protein_embed)

        # 转至注意力层输入
        drug_QKV = drug_conv.permute(2, 0, 1)
        protein_QKV = protein_conv.permute(2, 0, 1)

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
            parts = line.strip().split(' ', 4)
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
        drug_idx, protein_idx, label = self.data[idx]

        return {
            'drug_idx': torch.tensor(drug_idx, dtype=torch.long),
            'protein_idx': torch.tensor(protein_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


dataset = Dataset('Davis.txt', drug_embedding, protein_embedding)

train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - test_size

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Model(drug_embedding, protein_embedding).to(device)
print(device)

# 定义损失函数
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))  # 之前是 1.0


# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

best_val_loss = float('inf')
patience = 10  # 连续多少个epoch没改进就停止
no_improve_epoch = 0  # 记录没有改进的epoch数
delta = 0.001  # 视为显著改进的最小阈值

total_start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_start_time = time.time()

    # Add tqdm progress bar for training
    train_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training")

    for batch in train_loader_tqdm:
        drug_idx = batch['drug_idx'].to(device)
        protein_idx = batch['protein_idx'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(drug_idx, protein_idx)
        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item())

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation")

    with torch.no_grad():
        for batch in val_loader_tqdm:
            drug_idx = batch['drug_idx'].to(device)
            protein_idx = batch['protein_idx'].to(device)
            labels = batch['label'].to(device)
            outputs = model(drug_idx, protein_idx)

            loss = criterion(outputs.squeeze(1), labels.float())  # Fix: Squeeze outputs
            val_loss += loss.item()

            # Correct prediction for binary classification
            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_loader_tqdm.set_postfix(val_loss=loss.item())  # Update tqdm display with validation loss

    current_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    # 检查是否是最佳模型
    if current_val_loss < (best_val_loss - delta):
        print(f"Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}")
        best_val_loss = current_val_loss
        no_improve_epoch = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "best_model.pth")
    else:
        no_improve_epoch += 1
        print(
            f"No improvement for {no_improve_epoch} epochs (Current: {current_val_loss:.4f}, Best: {best_val_loss:.4f})")

    # 检查是否提前停止
    if no_improve_epoch >= patience:
        print(f"\nEarly stopping triggered after {patience} epochs without improvement!")
        break
    # ============================================

    print(f"\nEpoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Val Loss: {current_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"Epoch Time: {epoch_time:.2f} seconds")

print("\nLoading best model for testing...")
model.load_state_dict(torch.load("best_model.pth"))
torch.save(model.state_dict(), "model.pth")

total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"Total training time: {total_time:.2f} seconds")

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 存储模型预测的概率

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            drug_idx = batch['drug_idx'].to(device)
            protein_idx = batch['protein_idx'].to(device)
            labels = batch['label'].to(device)

            outputs = model(drug_idx, protein_idx)
            probs = torch.sigmoid(outputs.squeeze(1))  # 获取概率值
            predicted = (probs > 0.35).long()  # 二值化预测

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 存储概率值，用于 AUC 和 PRC 计算

    # 计算各项指标
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)  # 计算 AUC（ROC AUC Score）
    prc = average_precision_score(all_labels, all_probs)  # 计算 PRC（PR AUC Score）

    return accuracy, precision, recall, f1, auc, prc


print("\nEvaluating on test set:")
test_accuracy, test_precision, test_recall, test_f1, test_auc, test_prc = evaluate(model, test_loader, device)

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test AUC (ROC-AUC): {test_auc:.4f}")
print(f"Test PRC (PR-AUC): {test_prc:.4f}")

# Evaluating on test set:
# Test Accuracy: 84.50%
# Test Precision: 0.6982
# Test Recall: 0.8007
# Test F1 Score: 0.7459
# Test AUC (ROC-AUC): 0.9198
# Test PRC (PR-AUC): 0.8446