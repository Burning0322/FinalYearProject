import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    """
    将 SMILES 字符串转换为数值向量表示。

    参数:
        line (str): SMILES 字符串
        smi_ch_ind (dict): SMILES 字符到索引的映射字典
        MAX_SMI_LEN (int): 最大 SMILES 长度 (默认 100)

    返回:
        np.ndarray: 长度为 MAX_SMI_LEN 的数值向量
    """
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    """
    将蛋白质序列字符串转换为数值向量表示。

    参数:
        line (str): 蛋白质序列字符串
        smi_ch_ind (dict): 序列字符到索引的映射字典
        MAX_SEQ_LEN (int): 最大序列长度 (默认 1000)

    返回:
        np.ndarray: 长度为 MAX_SEQ_LEN 的数值向量
    """
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savepath=None, patience=7, verbose=False, delta=0, num_n_fold=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath

    def __call__(self, score, model, num_epoch):

        if self.best_score == -np.inf:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, num_epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Have a new best checkpoint: ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.savepath +
                   '/valid_best_checkpoint.pth')

class PolyLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = torch.zeros((16, 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)


class MCANet(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(MCANet, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
                                     self.protein_kernel[0] - self.protein_kernel[1] - \
                                     self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = 5

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)

        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict


class onlyPolyLoss(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(onlyPolyLoss, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.durg_dim_afterCNNs = self.drug_MAX_LENGH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGH - \
                                     self.protein_kernel[0] - self.protein_kernel[1] - \
                                     self.protein_kernel[2] + 3

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.durg_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)

        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

class hyperparameter():
    def __init__(self):
        self.Learning_rate = 0.001
        self.Epoch = 200
        self.Batch_size = 16
        self.Patience = 50
        #学习率衰减的间隔轮数
        self.decay_interval = 10
        #学习率衰退
        self.lr_decay = 0.5
        # 权重衰减（用于正则化）
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.char_dim = 64
        self.loss_epsilon = 1

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset

from torch.utils.data import Dataset
class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

def collate_fn(batch_data):
    N = len(batch_data)
    drug_ids, protein_ids = [], []
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        drug_id, protein_id, compoundstr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        #labels_new[i] = np.int(label)
        labels_new[i] = int(label)
    return (compound_new, protein_new, labels_new)

def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            #在测试的时候使用eval()函数
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.to(DEVICE)
            proteins = proteins.to(DEVICE)
            labels = labels.to(DEVICE)

            if isinstance(MODEL, list):
                predicted_scores = torch.zeros(2).to(DEVICE)
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + \
                        MODEL[i](compounds, proteins)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores = MODEL(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(
                predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC

def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_precess(
        MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                "/{}_{}_prediction.txt".format(DATASET, dataset_class)
        else:
            filepath = save_path + \
                "/{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test

def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, Ensemble=False):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(
        Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)

    if Ensemble == False:
        print("The model's results:")
        filepath = "./{}/results.txt".format(DATASET)
    else:
        print("The ensemble model's results:")
        filepath = "./{}/ensemble_results.txt".format(DATASET)
    with open(filepath, 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(
            Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(
            Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(
            Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(
        Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

import random
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, auc


def run_model(SEED, DATASET, MODEL, K_Fold, LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load data'''
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None

    '''shuffle data'''
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    train_data_list = data_list[0:split_pos]
    test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

    train_dataset, valid_dataset = get_kfold_data(
        i_fold, train_data_list, k=K_Fold)
    train_dataset = CustomDataSet(train_dataset)
    valid_dataset = CustomDataSet(valid_dataset)
    test_dataset = CustomDataSet(test_data_list)
    train_size = len(train_dataset)

    train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                     collate_fn=collate_fn, drop_last=True)
    """ create model"""
    model = MODEL(hp).to(DEVICE)

    """Initialize weights"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    """create optimizer and scheduler"""
    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
        lr=hp.Learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
                                            cycle_momentum=False, step_size_up=train_size // hp.Batch_size)

    if LOSS == 'PolyLoss':
        Loss = PolyLoss(weight_loss=weight_loss,
                        DEVICE=DEVICE, epsilon=hp.loss_epsilon)
    else:
        Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

    """Output files"""
    save_path = "./" + DATASET + "/{}".format(i_fold + 1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

    early_stopping = EarlyStopping(
        savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

    for epoch in range(1, hp.Epoch + 1):
        if early_stopping.early_stop == True:
            break
        train_pbar = tqdm(
            enumerate(BackgroundGenerator(train_dataset_loader)),
            total=len(train_dataset_loader))
        """train"""
        train_losses_in_epoch = []
        model.train()

        for train_i, train_data in train_pbar:
            train_compounds, train_proteins, train_labels = train_data
            train_compounds = train_compounds.to(DEVICE)
            train_proteins = train_proteins.to(DEVICE)
            train_labels = train_labels.to(DEVICE)

            optimizer.zero_grad()
            # 前向传播
            predicted_interaction = model(train_compounds, train_proteins)
            train_loss = Loss(predicted_interaction, train_labels)
            train_losses_in_epoch.append(train_loss.item())
            # 反向传播
            train_loss.backward()
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
        train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

        """valid"""
        valid_pbar = tqdm(enumerate(BackgroundGenerator(valid_dataset_loader)), total=len(valid_dataset_loader))

        valid_losses_in_epoch = []
        model.eval()

        # 标签，预测结果，预测分数
        Y, P, S = [], [], []
        with torch.no_grad():
            for valid_i, valid_data in valid_pbar:
                valid_compounds, valid_proteins, valid_labels = valid_data

                valid_compounds = valid_compounds.to(DEVICE)
                valid_proteins = valid_proteins.to(DEVICE)
                valid_labels = valid_labels.to(DEVICE)

                # 预测结果
                valid_scores = model(valid_compounds, valid_proteins)
                # 计算验证损失
                valid_loss = Loss(valid_scores, valid_labels)
                valid_losses_in_epoch.append(valid_loss.item())
                # 计算验证指标
                # valid_labels = valid_labels.to('cpu').data.numpy()
                valid_labels = valid_labels.detach().cpu().numpy()
                # Softmax函数
                valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                # 预测类别
                valid_predictions = np.argmax(valid_scores, axis=1)
                # 提取正类概率
                valid_scores = valid_scores[:, 1]

                Y.extend(valid_labels)
                P.extend(valid_predictions)
                S.extend(valid_scores)

        """
        1. 准确率 输入: Y:真实标签,P:预测标签
        2. 召回率 输入: Y:真实标签,P:预测标签
        3. 准确率 输入: Y:真实标签,P:预测标签
        4. ROC曲线下的面积 输入: Y:真实标签,S:预测分数
        5. PRC (Precision-Recall Curve 下的面积) 输入: Y:真实标签,S:预测分数
        """
        Precision_dev = precision_score(Y, P)
        Reacll_dev = recall_score(Y, P)
        Accuracy_dev = accuracy_score(Y, P)
        AUC_dev = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC_dev = auc(fpr, tpr)

        # 验证集平均损失
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)

        epoch_len = len(str(hp.Epoch))
        print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_a_epoch:.5f} ' +
                     f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                     f'valid_AUC: {AUC_dev:.5f} ' +
                     f'valid_PRC: {PRC_dev:.5f} ' +
                     f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                     f'valid_Precision: {Precision_dev:.5f} ' +
                     f'valid_Reacll: {Reacll_dev:.5f} ')
        print(print_msg)

        '''save checkpoint and make decision when early stop'''
        early_stopping(Accuracy_dev, model, epoch)

        '''load best checkpoint'''
    model.load_state_dict(torch.load(early_stopping.savepath + '/valid_best_checkpoint.pth'))

    '''test model'''
    trainset_test_stable_results, _, _, _, _, _ = test_model(
        model, train_dataset_loader, save_path, DATASET, Loss,DEVICE, dataset_class="Train", FOLD_NUM=1)
    validset_test_stable_results, _, _, _, _, _ = test_model(
        model, valid_dataset_loader, save_path, DATASET, Loss,DEVICE, dataset_class="Valid", FOLD_NUM=1)
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
        model,test_dataset_loader,save_path,DATASET,Loss,DEVICE,dataset_class="Test",FOLD_NUM=1)

    AUC_List_stable.append(AUC_test)
    Accuracy_List_stable.append(Accuracy_test)
    AUPR_List_stable.append(PRC_test)
    Recall_List_stable.append(Recall_test)
    Precision_List_stable.append(Precision_test)

    with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
        f.write("Test the stable model" + '\n')
        f.write(trainset_test_stable_results + '\n')
        f.write(validset_test_stable_results + '\n')
        f.write(testset_test_stable_results + '\n')
    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(DEVICE)


if __name__ == '__main__':
    run_model(42, "Davis", MCANet, 5, "PolyLoss")