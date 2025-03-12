# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-26 19:28
LastEditTime: 2022-11-23 15:22
LastEditors: MrZQAQ
Description: hyperparameter config file
FilePath: /MCANet/config.py
'''

class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4         # 学习率（learning rate）
        self.Epoch = 200                 # 最大训练轮数（epoch）
        self.Batch_size = 16             # 每个批次（batch）的样本数量
        self.Patience = 50               # 提前停止（early stopping）策略中的等待轮数
        self.decay_interval = 10         # 学习率衰减的间隔轮数
        self.lr_decay = 0.5              # 学习率衰减系数
        self.weight_decay = 1e-4         # 权重衰减（L2 正则化系数）
        self.embed_dim = 64              # 嵌入层维度（embedding dimension）
        self.protein_kernel = [4, 8, 12] # 可能用于蛋白质序列的卷积核大小列表
        self.drug_kernel = [4, 6, 8]     # 可能用于小分子（SMILES）序列的卷积核大小列表
        self.conv = 40                   # 卷积输出通道数（或卷积层数量），需结合具体模型结构理解
        self.char_dim = 64               # 字符级别嵌入维度（若对 SMILES 或蛋白序列做字符级 embedding）
        self.loss_epsilon = 1           # 用于损失函数中的一个平滑因子或系数，具体用法看实际实现
