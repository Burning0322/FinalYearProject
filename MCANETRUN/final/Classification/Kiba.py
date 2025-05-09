import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import math
import random
from rdkit import Chem
import time
from tqdm import tqdm
import pandas as pd

# 设置随机种子和设备
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数
learning_rate = 0.001
epochs = 200
batch_size = 64
k_folds = 5
drug_kernel = [4, 6, 8]
protein_kernel = [4, 8, 12]
dropout = 0.5

# 加载嵌入
drug_embedding = torch.load("autodl-tmp/ligands_kiba.pt").to(device)
protein_embedding = torch.load("autodl-tmp/protein_kiba.pt").to(device)

drug_max_length = drug_embedding.shape[1]
protein_max_length = protein_embedding.shape[1]
drug_dim = drug_embedding.shape[2]
protein_dim = protein_embedding.shape[2]

drug_afterCNN = drug_max_length - sum(drug_kernel) + 3
protein_afterCNN = protein_max_length - sum(protein_kernel) + 3

conv = 16
attention_dim = conv * 4
mix_attention_head = 8

class Model(nn.Module):
    def __init__(self, drug_embedding, protein_embedding):
        super().__init__()
        self.drug_embedding = nn.Parameter(drug_embedding, requires_grad=True)
        self.protein_embedding = nn.Parameter(protein_embedding, requires_grad=True)

        self.drug_CNN = nn.Sequential(
            nn.Conv1d(drug_dim, conv, drug_kernel[0]),
            nn.BatchNorm1d(conv),
            nn.ReLU(),
            nn.Conv1d(conv, conv * 2, drug_kernel[1]),
            nn.BatchNorm1d(conv * 2),
            nn.ReLU(),
            nn.Conv1d(conv * 2, conv * 4, drug_kernel[2]),
            nn.BatchNorm1d(conv * 4),
            nn.ReLU(),
        )

        self.protein_CNN = nn.Sequential(
            nn.Conv1d(protein_dim, conv, protein_kernel[0]),
            nn.BatchNorm1d(conv),
            nn.ReLU(),
            nn.Conv1d(conv, conv * 2, protein_kernel[1]),
            nn.BatchNorm1d(conv * 2),
            nn.ReLU(),
            nn.Conv1d(conv * 2, conv * 4, protein_kernel[2]),
            nn.BatchNorm1d(conv * 4),
            nn.ReLU(),
        )

        self.drug_pool = nn.MaxPool1d(drug_afterCNN)
        self.protein_pool = nn.MaxPool1d(protein_afterCNN)
        self.attention = nn.MultiheadAttention(attention_dim, mix_attention_head)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv * 8, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, drug_idx, protein_idx):
        # 获取嵌入
        drug = self.drug_embedding[drug_idx]  # [B, L_d, drug_dim]
        protein = self.protein_embedding[protein_idx]  # [B, L_p, protein_dim]

        # 调整形状为 [batch, channels, seq_len] 以适应 CNN
        drug = drug.permute(0, 2, 1)  # [B, drug_dim, L_d]
        protein = protein.permute(0, 2, 1)  # [B, protein_dim, L_p]

        # 通过 CNN 提取特征
        drug_feat = self.drug_CNN(drug)  # [B, conv*4, L_d']
        protein_feat = self.protein_CNN(protein)  # [B, conv*4, L_p']

        # 调整形状为 [seq_len, batch, embed_dim] 以适应 MultiheadAttention
        drug_QKV = drug_feat.permute(2, 0, 1)  # [L_d', B, conv*4]
        protein_QKV = protein_feat.permute(2, 0, 1)  # [L_p', B, conv*4]

        # 交叉注意力
        drug_att, _ = self.attention(drug_QKV, protein_QKV, protein_QKV)  # [L_d', B, conv*4]
        protein_att, _ = self.attention(protein_QKV, drug_QKV, drug_QKV)  # [L_p', B, conv*4]

        # 调整回 [batch, channels, seq_len] 以适应池化
        drug_att = drug_att.permute(1, 2, 0)  # [B, conv*4, L_d']
        protein_att = protein_att.permute(1, 2, 0)  # [B, conv*4, L_p']

        # 池化
        drug_att = self.drug_pool(drug_att).squeeze(2)  # [B, conv*4]
        protein_att = self.protein_pool(protein_att).squeeze(2)  # [B, conv*4]

        # 全连接层
        return self.fc(torch.cat([drug_att, protein_att], dim=1))


class Dataset(Dataset):
    def __init__(self, file_path, augment_smiles=True, augment_factor=2, augment_protein=True,
                 mutation_rate=0.01, augment_affinity=False, noise_std=0.1):
        self.drug_embedding = drug_embedding
        self.protein_embedding = protein_embedding
        self.smiles2idx, self.protein2idx, self.data = {}, {}, []
        self.smiles_list = []
        self.protein_list = []
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 4)
                if len(parts) == 5:
                    _, _, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
                    sequence, label = rest.rsplit(' ', 1)
                    self.smiles_list.append(smiles)
                    self.protein_list.append(sequence)
                    if smiles not in self.smiles2idx:
                        self.smiles2idx[smiles] = len(self.smiles2idx)
                    if sequence not in self.protein2idx:
                        self.protein2idx[sequence] = len(self.protein2idx)
                    self.data.append((self.smiles2idx[smiles], self.protein2idx[sequence], int(label)))

        if augment_smiles:
            original_data = self.data.copy()
            for _ in range(augment_factor - 1):
                for smiles, (d_idx, p_idx, label) in zip(self.smiles_list, original_data):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        new_smiles = Chem.MolToSmiles(mol, doRandom=True)
                        new_mol = Chem.MolFromSmiles(new_smiles)
                        if new_mol is not None:
                            self.data.append((d_idx, p_idx, label))

        if augment_protein:
            original_data = self.data.copy()
            for _ in range(augment_factor - 1):
                for sequence, (d_idx, p_idx, label) in zip(self.protein_list, original_data):
                    new_sequence = list(sequence)
                    for i in range(len(new_sequence)):
                        if random.random() < mutation_rate:
                            new_sequence[i] = random.choice(self.amino_acids)
                    new_sequence = ''.join(new_sequence)
                    self.data.append((d_idx, p_idx, label))

        if augment_affinity:
            original_data = self.data.copy()
            for _ in range(augment_factor - 1):
                for d_idx, p_idx, label in original_data:
                    prob = 0.9 if label == 1 else 0.1
                    new_prob = prob + np.random.normal(0, noise_std)
                    new_prob = np.clip(new_prob, 0.0, 1.0)
                    new_label = 1 if new_prob >= 0.5 else 0
                    self.data.append((d_idx, p_idx, new_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d, p, y = self.data[idx]
        return {'drug_idx': torch.tensor(d), 'protein_idx': torch.tensor(p), 'label': torch.tensor(y)}

def evaluate(model, loader, return_probs=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    eval_bar = tqdm(loader, desc="评估", leave=False)
    with torch.no_grad():
        for batch in eval_bar:
            d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
            out = model(d, p)
            prob = torch.softmax(out, dim=1)[:, 1]  # 用于AUC和AUPR
            _, pred = torch.max(out, 1)  # 使用argmax获取预测类别
            y_true += y.tolist()
            y_pred += pred.tolist()
            y_prob += prob.tolist()
            del out, prob, pred
        torch.cuda.empty_cache()

    acc = 100 * (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    metrics = (acc, precision_score(y_true, y_pred), recall_score(y_true, y_pred),
               f1_score(y_true, y_pred), roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob))
    if return_probs:
        return metrics, np.array(y_prob), np.array(y_true)
    return metrics

dataset = Dataset("KIBA.txt",
                  augment_smiles=False,
                  augment_factor=2,
                  augment_protein=True,
                  mutation_rate=0.001,
                  augment_affinity=False,
                  noise_std=0.1)

total_start_time = time.time()

# 第一步：将数据集分为 80% 训练集和 20% 测试集
all_indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)

# 第二步：在训练集上进行 5 折交叉验证
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
    print(f"\n=== 第 {fold + 1}/{k_folds} 折 ===")

    log_dir = f"/root/tf-logs/fold_{fold + 1}"
    writer = SummaryWriter(log_dir=log_dir)

    train_loader = DataLoader(Subset(dataset, [train_indices[i] for i in train_idx]),
                              batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(Subset(dataset, [train_indices[i] for i in val_idx]),
                            batch_size=batch_size, shuffle=False)

    model = Model(drug_embedding, protein_embedding).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    labels = [dataset[train_indices[i]]['label'].item() for i in train_idx]
    pos = sum(1 for y in labels if y == 1)
    neg = len(labels) - pos
    ratio = neg / pos if pos > 0 else 1.0
    print(f"正样本: {pos}, 负样本: {neg}, 比率: {ratio:.2f}")

    weight_factor = ratio
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight_factor], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=epochs)

    best_val_loss = float('inf')
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"第 {epoch + 1}/{epochs} 轮 训练", leave=False)
        for batch in train_bar:
            d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            out = model(d, p)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            train_bar.set_postfix({"损失": f"{loss.item():.4f}"})

        train_loss = total_loss / len(train_loader)

        val_loss = 0
        model.eval()
        val_bar = tqdm(val_loader, desc=f"第 {epoch + 1}/{epochs} 轮 验证", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
                out = model(d, p)
                val_loss += criterion(out, y).item()

        val_loss /= len(val_loader)

        # 计算验证集指标（使用argmax）
        (acc, pre, rec, f1, auc, prc) = evaluate(model, val_loader)
        print(f"第 {epoch + 1} 轮, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证 F1: {f1:.4f}")

        with open("process.txt", "a") as f:
            f.write(f"第 {epoch + 1} 轮, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证 F1: {f1:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/accuracy', acc, epoch)
        writer.add_scalar('Metrics/precision', pre, epoch)
        writer.add_scalar('Metrics/recall', rec, epoch)
        writer.add_scalar('Metrics/f1', f1, epoch)
        writer.add_scalar('Metrics/auc', auc, epoch)
        writer.add_scalar('Metrics/prc', prc, epoch)

        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), f"model_fold_{fold}.pt")
        else:
            no_improve += 1
        if no_improve >= 7:
            print(f"在第 {epoch + 1} 轮提前停止")
            break

    writer.close()

    model.load_state_dict(torch.load(f"model_fold_{fold}.pt"))
    models.append(model)

    (acc, pre, rec, f1, auc, prc) = evaluate(model, val_loader)
    fold_results.append((acc, pre, rec, f1, auc, prc))
    print(f"第 {fold + 1} 折验证 - 准确率: {acc:.2f}%, 精确率: {pre:.4f}, 召回率: {rec:.4f}, "
          f"F1: {f1:.4f}, AUC: {auc:.4f}, PRC: {prc:.4f}")

# 计算验证集的平均指标
avg_val = np.mean(fold_results, axis=0)
print(f"\n=== {k_folds} 折验证的平均结果 ===")
print(f"准确率: {avg_val[0]:.2f}%, 精确率: {avg_val[1]:.4f}, 召回率: {avg_val[2]:.4f}, "
      f"F1: {avg_val[3]:.4f}, AUC: {avg_val[4]:.4f}, PRC: {avg_val[5]:.4f}")

# 第三步：在测试集上进行集成预测
print("\n=== 测试集上的集成预测 ===")
logits_list = []  # 存储每个模型的logits
y_true = None
for model in tqdm(models, desc="测试模型"):
    model.eval()
    y_true_batch, y_logits_batch = [], []
    with torch.no_grad():
        for batch in test_loader:
            d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
            out = model(d, p)
            y_true_batch += y.tolist()
            y_logits_batch += out.tolist()
    logits_list.append(np.array(y_logits_batch))
    if y_true is None:
        y_true = np.array(y_true_batch)

# 计算 5 个模型的平均logits
avg_logits = np.mean(logits_list, axis=0)
avg_probs = torch.softmax(torch.tensor(avg_logits), dim=1).numpy()[:, 1]  # 用于AUC和AUPR
y_pred = np.argmax(avg_logits, axis=1)  # 使用argmax获取预测类别

# 计算测试集指标
test_acc = 100 * (y_true == y_pred).sum() / len(y_true)
test_pre = precision_score(y_true, y_pred)
test_rec = recall_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred)
test_auc = roc_auc_score(y_true, avg_probs)
test_prc = average_precision_score(y_true, avg_probs)

print(f"测试集 - 准确率: {test_acc:.2f}%, 精确率: {test_pre:.4f}, 召回率: {test_rec:.4f}, "
      f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}, PRC: {test_prc:.4f}")

# 计算总运行时间
total_end_time = time.time()
total_time = total_end_time - total_start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

# 第四步：将结果写入文件
with open("results.txt", "a") as f:
    f.write("\n=== 结果 ===\n")
    f.write(f"{k_folds} 折验证的平均结果:\n")
    f.write(f"准确率: {avg_val[0]:.2f}%\n")
    f.write(f"精确率: {avg_val[1]:.4f}\n")
    f.write(f"召回率: {avg_val[2]:.4f}\n")
    f.write(f"F1: {avg_val[3]:.4f}\n")
    f.write(f"AUC: {avg_val[4]:.4f}\n")
    f.write(f"PRC: {avg_val[5]:.4f}\n")
    f.write("\n测试集 (集成预测):\n")
    f.write(f"准确率: {test_acc:.2f}%\n")
    f.write(f"精确率: {test_pre:.4f}\n")
    f.write(f"召回率: {test_rec:.4f}\n")
    f.write(f"F1: {test_f1:.4f}\n")
    f.write(f"AUC: {test_auc:.4f}\n")
    f.write(f"PRC: {test_prc:.4f}\n")
    f.write(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n")
    f.write("=" * 50 + "\n")
