import dash
import torch
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import mysql.connector
from mysql.connector import Error
import re
import requests
import pandas as pd
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
import torch.nn as nn
import math
import numpy as np
from rdkit import Chem

dash.register_page(__name__, path="/dti")

# 数据库配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "dti"
}

# 导航栏
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Drug Target Interaction", className="ms-2", style={"fontWeight": "bold", "fontSize": "24px"}, id="navbar-title"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="/", id="navbar-home")),
            dbc.NavItem(dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("All", href="/resources", id="navbar-resources"),
                    dbc.DropdownMenuItem("Drug", href="/drug"),
                    dbc.DropdownMenuItem("Protein", href="/protein")
                ],
                nav=True,
                in_navbar=True,
                label="Resources",
                id="navbar-resources-dropdown"
            )),
            dbc.NavItem(dbc.NavLink("DTI", href="/dti", id="navbar-dti")),
            dbc.NavItem(dbc.NavLink("About Us", href="/about", id="navbar-about")),
            dbc.NavItem(dbc.Button("Contact", color="primary", className="ms-2", id="navbar-contact")),
            dbc.NavItem(
                dcc.Dropdown(
                    id="language-dropdown",
                    options=[
                        {"label": "English", "value": "en"},
                        {"label": "中文", "value": "cn"},
                    ],
                    value="en",
                    clearable=False,
                    style={"width": "100px", "marginLeft": "10px"}
                )
            ),
        ], className="ms-auto", navbar=True)
    ]),
    color="light",
    dark=False,
    sticky="top"
)

# 页脚
footer = html.Footer([
    html.Div([
        html.Span("Follow us", style={"fontWeight": "bold", "marginRight": "10px"}),
        html.A(html.I(className="bi bi-link"), href="https://www.zstu.edu.cn", target="_blank", style={"marginRight": "10px", "color": "black"}),
        html.A(html.I(className="bi bi-github"), href="https://github.com/Burning0322/FinalYearProject.git", target="_blank", style={"color": "black"}),
    ], style={"textAlign": "center", "padding": "10px 0"}),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H5("About"),
            html.Ul([
                html.Li(html.A("About DTI", href="#", style={"color": "black", "textDecoration": "none"})),
                html.Li(html.A("DTI Project", href="https://github.com/Burning0322/FinalYearProject.git", style={"color": "black", "textDecoration": "none"})),
                html.Li(html.A("Research", href="https://github.com/Burning0322/FinalYearProject.git", style={"color": "black", "textDecoration": "none"})),
                html.Li(html.A("Team", href="https://github.com/Burning0322/FinalYearProject.git", style={"color": "black", "textDecoration": "none"})),
                html.Li(html.A("Contact Us", href="#", style={"color": "black", "textDecoration": "none"})),
            ], style={"listStyleType": "none", "padding": 0}),
        ], md=3),
        dbc.Col([
            html.H5("Learn more"),
            html.Ul([
                html.Li(html.A("Davis", href="#", style={"color": "black", "textDecoration": "none"})),
                html.Li(html.A("Kiba", href="#", style={"color": "black", "textDecoration": "none"})),
            ], style={"listStyleType": "none", "padding": 0}),
        ], md=3),
        dbc.Col([
            html.H5("Sign up for updates on our latest innovations"),
            dcc.Input(
                placeholder="Email address",
                type="email",
                style={
                    "width": "100%",
                    "padding": "8px",
                    "marginBottom": "10px",
                    "borderRadius": "5px",
                    "border": "1px solid #ccc"
                }
            ),
            html.Div([
                html.Small([
                    "I accept Zhejiang Sci-Tech University's ",
                    html.A("Terms and Conditions", href="#", style={"color": "black"}),
                    " and acknowledge that my information will be used in accordance with ZSTU's ",
                    html.A("Privacy Policy", href="#", style={"color": "black"}),
                    "."
                ], style={"marginBottom": "10px", "display": "block"}),
                html.Button("Sign up", className="btn btn-primary", style={"width": "100%"}),
            ]),
        ], md=4, style={"textAlign": "left"}),
    ], className="p-4"),
    html.Hr(),
    html.Div([
        html.Span("ZSTU", style={"fontWeight": "bold", "marginRight": "20px"}),
        html.A("About ZSTU", href="https://www.zstu.edu.cn", style={"color": "black", "textDecoration": "none", "marginRight": "20px"}),
        html.A("Privacy", href="#", style={"color": "black", "textDecoration": "none", "marginRight": "20px"}),
        html.A("Terms", href="#", style={"color": "black", "textDecoration": "none"}),
    ], style={"textAlign": "center", "padding": "10px 0"})
], style={"backgroundColor": "#f8f9fa", "padding": "20px 0", "marginTop": "20px"})

# 输入部分
input_section = dbc.Row([
    dbc.Col([
        html.Label("Drug", style={"fontWeight": "bold", "fontSize": "18px"}),
        dcc.Input(
            id="drug-input",
            type="text",
            placeholder="Enter SMILES (e.g., CCO for ethanol)",
            style={"width": "100%", "padding": "8px", "marginBottom": "10px"}
        ),
    ], md=6),
    dbc.Col([
        html.Label("Protein", style={"fontWeight": "bold", "fontSize": "18px"}),
        dcc.Input(
            id="protein-input",
            type="text",
            placeholder="Enter Protein Sequence (e.g., MKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGR...)",
            style={"width": "100%", "padding": "8px", "marginBottom": "10px"}
        ),
    ], md=6)
], className="mb-4")

# 预测按钮
predict_button = html.Button(
    "Predict",
    id="predict-button",
    className="btn btn-primary",
    style={"width": "200px", "marginBottom": "20px"}
)

# 药物详情表格
drug_details_table = dash_table.DataTable(
    id="drug-details-table",
    columns=[
        {"name": "Property", "id": "Property"},
        {"name": "Value", "id": "Value", "presentation": "markdown"}
    ],
    data=[],
    style_table={"width": "100%", "maxHeight": "400px", "overflowY": "auto"},
    style_cell={"textAlign": "left", "padding": "5px", "whiteSpace": "normal"},
    markdown_options={"html": True},
    tooltip_duration=None
)

# 蛋白质详情表格
protein_details_table = dash_table.DataTable(
    id="protein-details-table",
    columns=[
        {"name": "Property", "id": "Property"},
        {"name": "Value", "id": "Value"}
    ],
    data=[],
    style_table={"width": "100%", "maxHeight": "400px", "overflowY": "auto"},
    style_cell={"textAlign": "left", "padding": "5px", "whiteSpace": "normal"}
)

# 详情部分
details_section = dbc.Row([
    dbc.Col([html.H4("Drug Details", style={"fontWeight": "bold"}), drug_details_table], md=6),
    dbc.Col([html.H4("Protein Details", style={"fontWeight": "bold"}), protein_details_table], md=6)
])

# 预测结果
prediction_result = html.Div(
    id="prediction-result",
    style={
        "marginTop": "20px",
        "padding": "15px",
        "border": "1px solid #eee",
        "borderRadius": "5px",
        "textAlign": "center"
    }
)

# 页面布局
layout = html.Div([navbar, dbc.Container([input_section, predict_button, details_section, prediction_result]), footer])

# 数据库连接辅助函数
def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except Error as e:
        print(f"数据库连接错误: {e}")
        return None

# 格式化分子式
def format_molecular_formula(formula):
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(r'([A-Za-z])(\d+)', lambda m: m.group(1) + m.group(2).translate(subscript_map), formula)

# 从数据库或 PubChem 获取药物数据
def fetch_drug_data(smiles):
    conn = get_db_connection()
    if conn and conn.is_connected():
        try:
            query = "SELECT * FROM drug WHERE smiles = %s"
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (smiles,))
            result = cursor.fetchone()
            if result:
                return result
        except Error as e:
            print(f"数据库错误: {e}")
        finally:
            cursor.close()
            conn.close()

    try:
        properties = (
            "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,"
            "IUPACName,InChI,InChIKey,XLogP,ExactMass,MonoisotopicMass,TPSA,"
            "Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,"
            "HeavyAtomCount,IsotopeAtomCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,"
            "DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount"
        )
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/{properties}/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                props = data['PropertyTable']['Properties'][0]
                cid = str(props.get('CID', 'N/A'))
                result = {
                    'query': smiles,
                    'compound_id': cid,
                    'molecular_formula': props.get('MolecularFormula'),
                    'molecular_weight': str(props.get('MolecularWeight')),
                    'smiles': smiles,
                    'canonical_smiles': props.get('CanonicalSMILES'),
                    'isomeric_smiles': props.get('IsomericSMILES'),
                    'iupac_name': props.get('IUPACName'),
                    'inchi': props.get('InChI'),
                    'inchi_key': props.get('InChIKey'),
                    'xlogp': str(props.get('XLogP')) if props.get('XLogP') is not None else None,
                    'exact_mass': str(props.get('ExactMass')),
                    'monoisotopic_mass': str(props.get('MonoisotopicMass')),
                    'tpsa': str(props.get('TPSA')),
                    'complexity': str(props.get('Complexity')),
                    'charge': str(props.get('Charge')),
                    'h_bond_donor_count': str(props.get('HBondDonorCount')),
                    'h_bond_acceptor_count': str(props.get('HBondAcceptorCount')),
                    'rotatable_bond_count': str(props.get('RotatableBondCount')),
                    'heavy_atom_count': str(props.get('HeavyAtomCount')),
                    'isotope_atom_count': str(props.get('IsotopeAtomCount')),
                    'defined_atom_stereo_count': str(props.get('DefinedAtomStereoCount')),
                    'undefined_atom_stereo_count': str(props.get('UndefinedAtomStereoCount')),
                    'defined_bond_stereo_count': str(props.get('DefinedBondStereoCount')),
                    'undefined_bond_stereo_count': str(props.get('UndefinedBondStereoCount')),
                    'covalent_unit_count': str(props.get('CovalentUnitCount')),
                    'conformer_count_3d': None,
                    'volume_3d': None,
                    'x_steric_quadrupole_3d': None,
                    'y_steric_quadrupole_3d': None,
                    'z_steric_quadrupole_3d': None,
                    'feature_acceptor_count_3d': None,
                    'feature_donor_count_3d': None,
                    'feature_anion_count_3d': None,
                    'feature_cation_count_3d': None,
                    'feature_ring_count_3d': None,
                    'feature_hydrophobe_count_3d': None,
                    'effective_rotor_count_3d': None,
                    'fingerprint_2d': None
                }
                conformer_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/conformers/JSON"
                conformer_response = requests.get(conformer_url, timeout=10)
                if conformer_response.status_code == 200:
                    conformer_data = conformer_response.json()
                    if 'PC_Compounds' in conformer_data:
                        result['conformer_count_3d'] = str(len(conformer_data['PC_Compounds'][0].get('conformers', [])))
                return result
        return {"error": "药物不存在", "smiles": smiles}
    except requests.RequestException as e:
        print(f"在线获取数据失败: {e}")
        return {"error": "药物不存在", "smiles": smiles}

# 从数据库或 UniProt 获取蛋白质数据
def fetch_protein_data(sequence):
    conn = get_db_connection()
    if conn and conn.is_connected():
        try:
            query = "SELECT * FROM protein WHERE sequence = %s"
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (sequence,))
            result = cursor.fetchone()
            if result:
                return result
        except Error as e:
            print(f"数据库错误: {e}")
        finally:
            cursor.close()
            conn.close()

    try:
        df = pd.read_csv("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/DavisNKiba.csv")
        matched_row = df[df['Protein Sequence'] == sequence]
        if not matched_row.empty:
            protein_name = matched_row.iloc[0]['Protein Name']
            conn = get_db_connection()
            if conn and conn.is_connected():
                cursor = conn.cursor(dictionary=True)
                query = "SELECT * FROM protein WHERE gene_names = %s"
                cursor.execute(query, (protein_name,))
                results = cursor.fetchall()
                result = results[0] if results else None
                cursor.close()
                conn.close()
                if result:
                    return result
    except Exception as e:
        print(f"数据查询错误: {e}")

    try:
        url = f"https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=sequence:{sequence}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            lines = response.text.splitlines()
            if len(lines) > 1:
                data = lines[1].split('\t')
                protein_data = {
                    'uniprot_accession': data[0],
                    'uniprot_id': data[1],
                    'protein_name': data[2],
                    'gene_names': data[3],
                    'organism': data[4],
                    'sequence': sequence,
                    'length': len(sequence)
                }
                conn = get_db_connection()
                if conn and conn.is_connected():
                    cursor = conn.cursor()
                    insert_query = """
                        INSERT INTO protein (uniprot_accession, uniprot_id, protein_name, gene_names, organism, sequence, length)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            uniprot_id = VALUES(uniprot_id),
                            protein_name = VALUES(protein_name),
                            gene_names = VALUES(gene_names),
                            organism = VALUES(organism),
                            sequence = VALUES(sequence),
                            length = VALUES(length)
                    """
                    cursor.execute(insert_query, (
                        protein_data['uniprot_accession'], protein_data['uniprot_id'], protein_data['protein_name'],
                        protein_data['gene_names'], protein_data['organism'], protein_data['sequence'], protein_data['length']
                    ))
                    conn.commit()
                    cursor.close()
                    conn.close()
                return protein_data
            else:
                return {"error": "未找到对应的蛋白质数据"}
        else:
            print(f"UniProt API 错误: {response.status_code}")
            return {"error": "无法通过序列获取数据"}
    except requests.RequestException as e:
        print(f"在线获取蛋白质数据失败: {e}")
        return {"error": "蛋白质不存在"}

# 超参数
drug_max_length = 94
protein_max_length = 1000
drug_kernel = [4, 6, 8]
protein_kernel = [4, 8, 12]
drug_afterCNN = drug_max_length - sum(drug_kernel) + 3
protein_afterCNN = protein_max_length - sum(protein_kernel) + 3
conv = 40
attention_dim = conv * 4
mix_attention_head = 5
dropout = 0.5
threshold = 0.7

# 设备设置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("当前设备:", device)

# 加载预训练模型用于动态嵌入
drug_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/ChemBERTa-77M-MLM"
chemberta_model = AutoModel.from_pretrained(drug_path).to(device)
chemberta_tokenizer = AutoTokenizer.from_pretrained(drug_path)
chemberta_model.eval()
print("ChemBERTa模型加载完成")

protein_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/prot_t5_xl_uniref50"
prot_t5_model = T5EncoderModel.from_pretrained(protein_path).to(device)
prot_t5_tokenizer = T5Tokenizer.from_pretrained(protein_path, legacy=False)
prot_t5_model.eval()
print("Prot-T5模型加载完成")

# 加载静态嵌入
drug_embedding = torch.load("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/ligands_davis.pt", map_location=device)
protein_embedding = torch.load("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/protein_davis.pt", map_location=device)
print("静态嵌入数据加载完成")

# 特征提取函数（动态嵌入）
def extract_drug_features(smiles):
    inputs = chemberta_tokenizer(smiles, padding=True, truncation=True, max_length=drug_max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = chemberta_model(**inputs)
        features = outputs.last_hidden_state
    return features

def extract_protein_features(sequence):
    spaced_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence.strip())))
    inputs = prot_t5_tokenizer(spaced_sequence, padding="max_length", truncation=True, max_length=protein_max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = prot_t5_model(**inputs)
        features = outputs.last_hidden_state
    return features

# 双向多头交叉注意力模块
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

# 模型定义
class Model(nn.Module):
    def __init__(self, drug_embedding, protein_embedding):
        super().__init__()
        self.drug_embedding = nn.Parameter(drug_embedding, requires_grad=True)
        self.protein_embedding = nn.Parameter(protein_embedding, requires_grad=True)

        drug_dim = drug_embedding.shape[2]
        protein_dim = protein_embedding.shape[2]

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

        # self.drug_pool = nn.MaxPool1d(drug_afterCNN)
        # self.protein_pool = nn.MaxPool1d(protein_afterCNN)
        self.drug_pool = nn.AdaptiveMaxPool1d(1)
        self.protein_pool = nn.AdaptiveMaxPool1d(1)
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

    def predict_from_features(self, drug_feat, protein_feat):
        drug = drug_feat.permute(0, 2, 1)
        protein = protein_feat.permute(0, 2, 1)
        drug_feat = self.drug_CNN(drug).permute(0, 2, 1)
        protein_feat = self.protein_CNN(protein).permute(0, 2, 1)
        drug_att, protein_att = self.attention(drug_feat, protein_feat)
        drug_att = self.drug_pool(drug_att.permute(0, 2, 1)).squeeze(2)
        protein_att = self.protein_pool(protein_att.permute(0, 2, 1)).squeeze(2)
        return self.fc(torch.cat([drug_att, protein_att], dim=1))

class Dataset:
    def __init__(self, file_path):
        self.smiles2idx, self.protein2idx, self.data = {}, {}, []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 4)  # 根据你的数据文件格式调整分割逻辑
                if len(parts) == 5:
                    _, _, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
                    sequence, label = rest.rsplit(' ', 1)
                    if smiles not in self.smiles2idx:
                        self.smiles2idx[smiles] = len(self.smiles2idx)
                    if sequence not in self.protein2idx:
                        self.protein2idx[sequence] = len(self.protein2idx)
                    self.data.append((self.smiles2idx[smiles], self.protein2idx[sequence], int(label)))

# 实例化 Dataset 并创建全局索引
dataset = Dataset("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/Davis.txt")
smiles2idx = dataset.smiles2idx
protein2idx = dataset.protein2idx
print("索引映射创建完成")

# 加载5个预训练模型
models = []
for fold in range(5):
    model = Model(drug_embedding, protein_embedding).to(device)
    state_dict = torch.load(f"/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/final/ligandsnprotein/model_fold_{fold}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
print("5个模型加载完成")

@dash.callback(
    [Output("drug-details-table", "data"),
     Output("protein-details-table", "data"),
     Output("prediction-result", "children")],
    Input("predict-button", "n_clicks"),
    [State("drug-input", "value"), State("protein-input", "value")],
    prevent_initial_call=True
)
def update_details(n_clicks, drug_input, protein_input):
    drug_table_data = []
    protein_table_data = []
    prediction_result = ""

    # 处理药物输入
    if drug_input:
        smiles = drug_input.strip()
        drug_data = fetch_drug_data(smiles)
        if "error" in drug_data:
            drug_table_data = [{"Property": "Error", "Value": f"No data found for SMILES: {smiles}"}]
        else:
            for key, value in drug_data.items():
                if value is None:
                    value = "N/A"
                if key == "molecular_formula":
                    value = format_molecular_formula(value)
                    value = f"<span style='font-size:1.1em'>{value}</span>"
                drug_table_data.append({"Property": key.replace("_", " ").title(), "Value": str(value)})
    else:
        drug_table_data = [{"Property": "Error", "Value": "请填写SMILES字符串"}]

    # 处理蛋白质输入
    if protein_input:
        sequence = protein_input.strip().upper()
        protein_data = fetch_protein_data(sequence)
        if "error" in protein_data:
            protein_table_data = [{"Property": "Error", "Value": f"No data found for sequence: {sequence[:30]}..."}]
        else:
            for key, value in protein_data.items():
                if value is None:
                    value = "N/A"
                protein_table_data.append({"Property": key.replace("_", " ").title(), "Value": str(value)})
    else:
        protein_table_data = [{"Property": "Error", "Value": "请填写蛋白质序列"}]

    # 进行预测
    if drug_input and protein_input:
        try:
            smiles = drug_input.strip()
            sequence = protein_input.strip().upper()

            # 获取索引
            drug_idx = smiles2idx.get(smiles)
            protein_idx = protein2idx.get(sequence)

            if drug_idx is not None and protein_idx is not None:
                # 使用静态嵌入进行预测
                drug_idx_tensor = torch.tensor([drug_idx]).to(device)
                protein_idx_tensor = torch.tensor([protein_idx]).to(device)
                probs = []
                with torch.no_grad():
                    for model in models:
                        out = model(drug_idx_tensor, protein_idx_tensor)
                        prob = torch.softmax(out, dim=1)[0, 1].item()
                        probs.append(prob)
                avg_prob = sum(probs) / len(probs)
                prediction_result = f"预测结果：结合概率 = {avg_prob:.4f}, " + ("结合" if avg_prob >= threshold else "不结合")
            else:
                # 使用动态嵌入进行预测
                drug_feat = extract_drug_features(smiles)
                protein_feat = extract_protein_features(sequence)
                probs = []
                with torch.no_grad():
                    for model in models:
                        out = model.predict_from_features(drug_feat, protein_feat)
                        prob = torch.softmax(out, dim=1)[0, 1].item()
                        probs.append(prob)
                avg_prob = sum(probs) / len(probs)
                prediction_result = f"预测结果（动态嵌入）：结合概率 = {avg_prob:.4f}, " + ("结合" if avg_prob >= threshold else "不结合")

        except Exception as e:
            prediction_result = f"预测失败：{str(e)}"

    return drug_table_data, protein_table_data, prediction_result