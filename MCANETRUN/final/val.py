import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
threshold = 0.7
batch_size = 5

# 加载嵌入
drug_embedding = torch.load("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/ligands_davis.pt", map_location=device, weights_only=True).to(device)
protein_embedding = torch.load("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/protein_davis.pt", map_location=device, weights_only=True).to(device)

# 定义双向多头交叉注意力模块
class BidirectionalMultiheadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q_drug = nn.Linear(embed_dim, embed_dim)
        self.W_k_drug = nn.Linear(embed_dim, embed_dim)
        self.W_v_drug = nn.Linear(embed_dim, embed_dim)

        self.W_q_protein = nn.Linear(embed_dim, embed_dim)
        self.W_k_protein = nn.Linear(embed_dim, embed_dim)
        self.W_v_protein = nn.Linear(embed_dim, embed_dim)

        self.out_proj_d = nn.Linear(embed_dim, embed_dim)
        self.out_proj_p = nn.Linear(embed_dim, embed_dim)

    def forward(self, drug_feat, protein_feat):
        B, L_d, _ = drug_feat.size()
        _, L_p, _ = protein_feat.size()

        Q_d = self.W_q_drug(drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
        K_p = self.W_k_protein(protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)
        V_p = self.W_v_protein(protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)

        Q_p = self.W_q_protein(protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)
        K_d = self.W_k_drug(drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
        V_d = self.W_v_drug(drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output_d1 = torch.matmul(torch.softmax(torch.matmul(Q_d, K_p.transpose(-2, -1)) / self.scale, dim=-1), V_p)
        attn_output_p1 = torch.matmul(torch.softmax(torch.matmul(Q_p, K_d.transpose(-2, -1)) / self.scale, dim=-1), V_d)

        attn_output_d1 = attn_output_d1.transpose(1, 2).contiguous().view(B, L_d, self.embed_dim)
        attn_output_p1 = attn_output_p1.transpose(1, 2).contiguous().view(B, L_p, self.embed_dim)

        updated_drug_feat = self.out_proj_d(attn_output_d1)
        updated_protein_feat = self.out_proj_p(attn_output_p1)

        Q_d2 = self.W_q_drug(updated_drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
        K_p2 = self.W_k_protein(updated_protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)
        V_p2 = self.W_v_protein(updated_protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)

        Q_p2 = self.W_q_protein(updated_protein_feat).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)
        K_d2 = self.W_k_drug(updated_drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
        V_d2 = self.W_v_drug(updated_drug_feat).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output_d2 = torch.matmul(torch.softmax(torch.matmul(Q_d2, K_p2.transpose(-2, -1)) / self.scale, dim=-1), V_p2)
        attn_output_p2 = torch.matmul(torch.softmax(torch.matmul(Q_p2, K_d2.transpose(-2, -1)) / self.scale, dim=-1), V_d2)

        attn_output_d2 = attn_output_d2.transpose(1, 2).contiguous().view(B, L_d, self.embed_dim)
        attn_output_p2 = attn_output_p2.transpose(1, 2).contiguous().view(B, L_p, self.embed_dim)

        final_drug_feat = self.out_proj_d(attn_output_d2)
        final_protein_feat = self.out_proj_p(attn_output_p2)

        return final_drug_feat, final_protein_feat

# 定义模型类
class Model(nn.Module):
    def __init__(self, drug_embedding, protein_embedding):
        super().__init__()
        self.drug_embedding = nn.Parameter(drug_embedding, requires_grad=True)
        self.protein_embedding = nn.Parameter(protein_embedding, requires_grad=True)

        drug_dim = drug_embedding.shape[2]
        protein_dim = protein_embedding.shape[2]
        conv = 40
        attention_dim = conv * 4
        mix_attention_head = 5
        drug_max_length = 94
        protein_max_length = 1000
        drug_kernel = [4, 6, 8]
        protein_kernel = [4, 8, 12]
        drug_afterCNN = drug_max_length - sum(drug_kernel) + 3
        protein_afterCNN = protein_max_length - sum(protein_kernel) + 3
        dropout = 0.5

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

        # self.drug_pool = nn.AdaptiveMaxPool1d(1)
        # self.protein_pool = nn.AdaptiveMaxPool1d(1)
        self.drug_pool = nn.MaxPool1d(drug_afterCNN)
        self.protein_pool = nn.MaxPool1d(protein_afterCNN)
        self.attention = BidirectionalMultiheadCrossAttention(attention_dim, mix_attention_head)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv * 8, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, drug_idx, protein_idx):
        drug = self.drug_embedding[drug_idx]
        protein = self.protein_embedding[protein_idx]
        drug = drug.permute(0, 2, 1)
        protein = protein.permute(0, 2, 1)
        drug_feat = self.drug_CNN(drug).permute(0, 2, 1)
        protein_feat = self.protein_CNN(protein).permute(0, 2, 1)
        drug_att, protein_att = self.attention(drug_feat, protein_feat)
        drug_att = self.drug_pool(drug_att.permute(0, 2, 1)).squeeze(2)
        protein_att = self.protein_pool(protein_att.permute(0, 2, 1)).squeeze(2)
        return self.fc(torch.cat([drug_att, protein_att], dim=1))

# 定义数据集类
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

# 加载整个Davis数据集
dataset = Dataset("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/KIBA.txt")
print(f"加载Davis数据集完成，总样本数: {len(dataset)}")

# 创建前五个样本的子集
subset_indices = list(range(len(dataset)))
subset_dataset = Subset(dataset, subset_indices)
davis_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)
print(f"加载Davis数据集前五个样本完成，总样本数: {len(subset_dataset)}")

# 加载5个训练好的模型
models = []
for fold in range(5):
    model = Model(drug_embedding, protein_embedding).to(device)
    state_dict = torch.load(f"/Volumes/PASSPORT/FinalYearProject/final/finalstatic/TrueFalseDavis/model_fold_{fold}.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
print("5个模型加载完成")

# 检查数据集标签分布（完整数据集）
labels = [data[2] for data in dataset.data]
print(f"完整数据集 - 正样本数: {sum(labels)}")
print(f"完整数据集 - 负样本数: {len(labels) - sum(labels)}")

# 检查子集标签分布
subset_labels = [dataset[i]['label'].item() for i in subset_indices]
print(f"子集 - 正样本数: {sum(subset_labels)}")
print(f"子集 - 负样本数: {len(subset_labels) - sum(subset_labels)}")

# 定义预测和统计函数
def calculate_predictions(models, loader):
    y_true = []
    y_pred = []
    y_prob = []  # 保存预测概率用于其他指标
    with torch.no_grad():
        for batch in tqdm(loader, desc="预测中"):
            drug_idx = batch['drug_idx'].to(device)
            protein_idx = batch['protein_idx'].to(device)
            labels = batch['label'].to(device)

            probs = []
            for model in models:
                out = model(drug_idx, protein_idx)
                prob = torch.softmax(out, dim=1)[:, 1]
                probs.append(prob)
            avg_prob = torch.mean(torch.stack(probs), dim=0)
            pred = (avg_prob >= threshold).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(avg_prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    correct = (y_true == y_pred).sum()
    incorrect = len(y_true) - correct

    return y_true, y_pred, y_prob, correct, incorrect

# 计算预测结果
y_true, y_pred, y_prob, correct, incorrect = calculate_predictions(models, davis_loader)

# 输出基本结果
print(f"\n=== Davis数据集前五个样本预测结果 ===")
print(f"总样本数: {len(subset_dataset)}")
print(f"预测正确的数量: {correct}")
print(f"预测错误的数量: {incorrect}")
print(f"准确率: {(correct / len(subset_dataset)) * 100:.2f}%")

# 检查预测分布
print(f"预测为正样本数: {sum(y_pred)}")
print(f"预测为负样本数: {len(y_pred) - sum(y_pred)}")

# 计算其他指标
print(f"精确率: {precision_score(y_true, y_pred):.4f}")
print(f"召回率: {recall_score(y_true, y_pred):.4f}")
print(f"F1 分数: {f1_score(y_true, y_pred):.4f}")

# print("\n=== 前五个样本的详细预测结果 ===")
# for idx in subset_indices:
#     sample = dataset[idx]
#     drug_idx = sample['drug_idx'].unsqueeze(0).to(device)
#     protein_idx = sample['protein_idx'].unsqueeze(0).to(device)
#     label = sample['label'].item()
#     with torch.no_grad():
#         probs = [model(drug_idx, protein_idx) for model in models]
#         avg_prob = torch.mean(torch.stack([torch.softmax(out, dim=1)[0, 1] for out in probs])).item()
#         pred = 1 if avg_prob >= threshold else 0
#     # 获取对应的SMILES和序列
#     smiles = [k for k, v in dataset.smiles2idx.items() if v == sample['drug_idx'].item()][0]
#     sequence = [k for k, v in dataset.protein2idx.items() if v == sample['protein_idx'].item()][0]
#     print(f"样本 {idx}:")
#     print(f"  药物SMILES: {smiles[:20]}...")  # 截断以便显示
#     print(f"  蛋白质序列: {sequence[:20]}...")
#     print(f"  真实标签: {label}, 预测概率: {avg_prob:.4f}, 预测标签: {pred}")

# Davis 嵌入 KIBA 数据集 TrueFalseDavis required_grad=True
# 总样本数: 116350
# 预测正确的数量: 92397
# 预测错误的数量: 23953
# 准确率: 79.41%
# 预测为正样本数: 2809
# 预测为负样本数: 113541
# 精确率: 0.1798
# 召回率: 0.0228
# F1 分数: 0.0405