import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_curve, auc
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import random
from rdkit import Chem
import time
from tqdm import tqdm
import pandas as pd
from lifelines.utils import concordance_index as lifelines_ci

# 设置随机种子和设备
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数
learning_rate = 0.001
epochs = 200
batch_size = 128
k_folds = 5
drug_kernel = [4, 6, 8]
protein_kernel = [4, 8, 12]
dropout = 0.5

# 加载嵌入
drug_embedding = torch.load("autodl-tmp/kiba_ligands.pt", weights_only=True).to(device)
protein_embedding = torch.load("autodl-tmp/protein_kiba.pt", weights_only=True).to(device)

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
            nn.Linear(512, 1)  # 回归任务，输出1维
        )

    def forward(self, drug_idx, protein_idx):
        drug = self.drug_embedding[drug_idx]
        protein = self.protein_embedding[protein_idx]

        drug = drug.permute(0, 2, 1)
        protein = protein.permute(0, 2, 1)

        drug_feat = self.drug_CNN(drug)
        protein_feat = self.protein_CNN(protein)

        drug_QKV = drug_feat.permute(2, 0, 1)
        protein_QKV = protein_feat.permute(2, 0, 1)

        drug_att, _ = self.attention(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.attention(protein_QKV, drug_QKV, drug_QKV)

        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drug_att = self.drug_pool(drug_att).squeeze(2)
        protein_att = self.protein_pool(protein_att).squeeze(2)

        output = self.fc(torch.cat([drug_att, protein_att], dim=1))
        return output.squeeze(1)

class Dataset(Dataset):
    def __init__(self, csv_file, augment_smiles=True, augment_factor=2, augment_protein=True,
                 mutation_rate=0.01, augment_affinity=False, noise_std=0.1):
        self.df = pd.read_csv(csv_file)
        self.smiles_list = self.df['drug_smiles'].tolist()
        self.protein_list = self.df['target_sequence'].tolist()
        self.labels = self.df['affinity'].tolist()

        if pd.isna(self.labels).any():
            print(f"警告：affinity 列中找到 {pd.isna(self.labels).sum()} 个 NaN 值")
            self.df = self.df.dropna(subset=['affinity'])
            self.smiles_list = self.df['drug_smiles'].tolist()
            self.protein_list = self.df['target_sequence'].tolist()
            self.labels = self.df['affinity'].tolist()

        # 检查 affinity 列中的非数值数据
        try:
            self.labels = [float(label) for label in self.labels]
        except ValueError as e:
            print(f"错误：affinity 列中包含非数值数据：{e}")
            raise

        self.smiles2idx = {smiles: idx for idx, smiles in enumerate(set(self.smiles_list))}
        self.protein2idx = {seq: idx for idx, seq in enumerate(set(self.protein_list))}

        self.data = [(self.smiles2idx[smiles], self.protein2idx[seq], float(label))
                     for smiles, seq, label in zip(self.smiles_list, self.protein_list, self.labels)]

        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

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
                    new_label = label + np.random.normal(0, noise_std)
                    self.data.append((d_idx, p_idx, new_label))

        labels = np.array([item[2] for item in self.data])
        mean, std = labels.mean(), labels.std()
        self.data = [(d, p, (label - mean) / std) for d, p, label in self.data]
        self.label_mean = mean
        self.label_std = std

        labels = np.array([item[2] for item in self.data])
        print(f"标准化后标签统计: 最小值: {labels.min():.4f}, 最大值: {labels.max():.4f}, "
              f"均值: {labels.mean():.4f}, 标准差: {labels.std():.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d, p, y = self.data[idx]
        return {'drug_idx': torch.tensor(d), 'protein_idx': torch.tensor(p), 'label': torch.tensor(y, dtype=torch.float)}

def evaluate(model, loader, dataset, return_preds=False):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估", leave=False):
            d, p, y = batch['drug_idx'].to(device), batch['protein_idx'].to(device), batch['label'].to(device)
            out = model(d, p)  # 移除 autocast，直接使用 FP32
            y_true += y.tolist()
            y_pred += out.tolist()
            del d, p, y, out
        torch.cuda.empty_cache()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_original = y_true * dataset.label_std + dataset.label_mean
    y_pred_original = y_pred * dataset.label_std + dataset.label_mean

    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    pearson = np.corrcoef(y_true_original, y_pred_original)[0, 1] if len(y_true) > 1 else 0.0
    ci = lifelines_ci(y_true_original, y_pred_original)
    r2m = r2_score(y_true_original, y_pred_original)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    spearman = spearmanr(y_true_original, y_pred_original)[0] if len(y_true) > 1 else 0.0

    threshold = np.percentile(y_true_original, 75)
    y_true_binary = (y_true_original >= threshold).astype(int)
    print(f"阈值: {threshold:.4f}, 正类样本数量: {np.sum(y_true_binary)}")
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_original)
    aupr = auc(recall, precision) if len(np.unique(y_true_binary)) > 1 else 0.0

    metrics = (mse, rmse, pearson, ci, r2m, aupr, mae, spearman)
    if return_preds:
        return metrics, y_pred_original, y_true_original
    return metrics

dataset = Dataset("autodl-tmp/kiba.csv",
                  augment_smiles=False,
                  augment_factor=2,
                  augment_protein=True,  # 修改为 False，与原始代码一致
                  mutation_rate=0.001,
                  augment_affinity=False,
                  noise_std=0.1)

total_start_time = time.time()

all_indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
    print(f"\n=== 第 {fold + 1}/{k_folds} 折 ===")

    log_dir = f"/root/tf-logs/fold_{fold + 1}"
    writer = SummaryWriter(log_dir=log_dir)

    train_loader = DataLoader(Subset(dataset, [train_indices[i] for i in train_idx]),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, [train_indices[i] for i in val_idx]),
                            batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = Model(drug_embedding, protein_embedding).to(device)
    # 移除 nn.DataParallel，明确使用单 GPU
    criterion = nn.MSELoss()
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
            out = model(d, p)  # 移除 autocast，直接使用 FP32
            loss = criterion(out, y)
            loss.backward()  # 直接反向传播，无需 GradScaler
            optimizer.step()  # 直接优化器更新
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
                out = model(d, p)  # 移除 autocast，直接使用 FP32
                val_loss += criterion(out, y).item()

        val_loss /= len(val_loader)

        (mse, rmse, pearson, ci, r2m, aupr, mae, spearman) = evaluate(model, val_loader, dataset)
        print(f"第 {epoch + 1} 轮, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
              f"验证 MSE: {mse:.4f}, RMSE: {rmse:.4f}, Pearson: {pearson:.4f}, CI: {ci:.4f}, "
              f"R2m: {r2m:.4f}, AUPR: {aupr:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}")

        with open("process.txt", "a") as f:
            f.write(f"第 {epoch + 1} 轮, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                    f"验证 MSE: {mse:.4f}, RMSE: {rmse:.4f}, Pearson: {pearson:.4f}, CI: {ci:.4f}, "
                    f"R2m: {r2m:.4f}, AUPR: {aupr:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}\n")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/mse', mse, epoch)
        writer.add_scalar('Metrics/rmse', rmse, epoch)
        writer.add_scalar('Metrics/pearson', pearson, epoch)
        writer.add_scalar('Metrics/ci', ci, epoch)
        writer.add_scalar('Metrics/r2m', r2m, epoch)
        writer.add_scalar('Metrics/aupr', aupr, epoch)
        writer.add_scalar('Metrics/mae', mae, epoch)
        writer.add_scalar('Metrics/spearman', spearman, epoch)

        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), f"autodl-tmp/model_fold_{fold}.pt")
        else:
            no_improve += 1
        if no_improve >= 7:
            print(f"在第 {epoch + 1} 轮提前停止")
            break

    writer.close()

    model.load_state_dict(torch.load(f"autodl-tmp/model_fold_{fold}.pt", map_location=device))  # 添加 map_location
    models.append(model)

    (mse, rmse, pearson, ci, r2m, aupr, mae, spearman) = evaluate(model, val_loader, dataset)
    fold_results.append((mse, rmse, pearson, ci, r2m, aupr, mae, spearman))
    print(f"第 {fold + 1} 折验证 - MSE: {mse:.4f}, RMSE: {rmse:.4f}, Pearson: {pearson:.4f}, "
          f"CI: {ci:.4f}, R2m: {r2m:.4f}, AUPR: {aupr:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}")

avg_val = np.mean(fold_results, axis=0)
print(f"\n=== {k_folds} 折验证的平均结果 ===")
print(f"MSE: {avg_val[0]:.4f}, RMSE: {avg_val[1]:.4f}, Pearson: {avg_val[2]:.4f}, "
      f"CI: {avg_val[3]:.4f}, R2m: {avg_val[4]:.4f}, AUPR: {avg_val[5]:.4f}")

print("\n=== 测试集上的集成预测 ===")
preds_list = []
y_true = None
for model in tqdm(models, desc="测试模型"):
    metrics, preds, true_labels = evaluate(model, test_loader, dataset, return_preds=True)
    preds_list.append(preds)
    if y_true is None:
        y_true = true_labels

avg_preds = np.mean(preds_list, axis=0)

mse = mean_squared_error(y_true, avg_preds)
rmse = np.sqrt(mse)
pearson = np.corrcoef(y_true, avg_preds)[0, 1]
ci = lifelines_ci(y_true, avg_preds)
r2m = r2_score(y_true, avg_preds)

threshold = y_true.mean()
y_true_binary = (y_true >= threshold).astype(int)
print(f"测试集阈值: {threshold:.4f}, 正类样本数量: {np.sum(y_true_binary)}")
precision, recall, _ = precision_recall_curve(y_true_binary, avg_preds)
aupr = auc(recall, precision) if len(np.unique(y_true_binary)) > 1 else 0.0

print(
    f"测试集 - MSE: {mse:.4f}, RMSE: {rmse:.4f}, Pearson: {pearson:.4f}, CI: {ci:.4f}, R2m: {r2m:.4f}, AUPR: {aupr:.4f}")

total_end_time = time.time()
total_time = total_end_time - total_start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

with open("results.txt", "a") as f:
    f.write("\n=== 结果 ===\n")
    f.write(f"{k_folds} 折验证的平均结果:\n")
    f.write(f"MSE: {avg_val[0]:.4f}\n")
    f.write(f"RMSE: {avg_val[1]:.4f}\n")
    f.write(f"Pearson: {avg_val[2]:.4f}\n")
    f.write(f"CI: {avg_val[3]:.4f}\n")
    f.write(f"R2m: {r2m:.4f}\n")
    f.write(f"AUPR: {avg_val[5]:.4f}\n")
    f.write("\n测试集 (集成预测):\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"Pearson: {pearson:.4f}\n")
    f.write(f"CI: {ci:.4f}\n")
    f.write(f"R2m: {r2m:.4f}\n")
    f.write(f"AUPR: {aupr:.4f}\n")
    f.write(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n")
    f.write("=" * 50 + "\n")
