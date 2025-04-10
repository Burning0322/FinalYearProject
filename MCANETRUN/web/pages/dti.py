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
                    value="en",  # 默认语言为英文
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
                    "I accept ZheJiang Sci-Tech University's ",
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
    return mysql.connector.connect(**db_config)

# 格式化分子式
def format_molecular_formula(formula):
    """Convert C18H22N4O2 to C₁₈H₂₂N₄O₂"""
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(r'([A-Za-z])(\d+)', lambda m: m.group(1) + m.group(2).translate(subscript_map), formula)

# 从数据库或 PubChem 获取药物数据
def fetch_drug_data(smiles):
    conn = None
    try:
        conn = get_db_connection()
        if conn.is_connected():
            query = "SELECT * FROM drug WHERE smiles = %s"
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (smiles,))
            result = cursor.fetchone()
            if result:
                return result
    except Error as e:
        print(f"数据库错误: {e}")
    finally:
        if conn and conn.is_connected():
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
    conn = None
    try:
        conn = get_db_connection()
        if conn.is_connected():
            query = "SELECT * FROM protein WHERE sequence = %s"
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (sequence,))
            result = cursor.fetchone()
            if result:
                return result
    except Error as e:
        print(f"数据库错误: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

    try:
        df = pd.read_csv("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/DavisNKiba.csv")
        matched_row = df[df['Protein Sequence'] == sequence]

        if not matched_row.empty:
            protein_name = matched_row.iloc[0]['Protein Name']
            conn = get_db_connection()
            if conn.is_connected():
                cursor = conn.cursor(dictionary=True)
                query = "SELECT * FROM protein WHERE gene_names = %s"
                cursor.execute(query, (protein_name,))
                results = cursor.fetchall()
                result = results[0] if results else None
                cursor.close()
                conn.close()
                if result:
                    return result
    except Error as e:
        print(f"数据库错误: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

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
                if conn.is_connected():
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
            print(f"Error: {response.status_code}")
            return {"error": "无法通过序列获取数据"}
    except requests.RequestException as e:
        print(f"在线获取蛋白质数据失败: {e}")
        return {"error": "蛋白质不存在"}

# 超参数（与原始模型一致）
drug_max_length = 94
protein_max_length = 1000
drug_kernel = [4, 6, 8]
protein_kernel = [4, 8, 12]
drug_afterCNN = drug_max_length - sum(drug_kernel) + 3
protein_afterCNN = protein_max_length - sum(protein_kernel) + 3
conv = 40
attention_dim = conv * 4
mix_attention_head = 5
drug_dim = 384  # ChemBERTa 的输出维度
protein_dim = 1024  # ProtT5 的输出维度

# 设备设置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/final/best_model_fold_5.pth"

# 加载预训练模型用于特征提取
drug_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/ChemBERTa-77M-MLM"  # 使用在线模型
chemberta_model = AutoModel.from_pretrained(drug_path).to(device)
chemberta_tokenizer = AutoTokenizer.from_pretrained(drug_path)

protein_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/prot_t5_xl_uniref50"  # 使用在线模型
prot_t5_model = T5EncoderModel.from_pretrained(protein_path).to(device)
prot_t5_tokenizer = T5Tokenizer.from_pretrained(protein_path)

# 特征提取函数
def extract_drug_features(smiles):
    inputs = chemberta_tokenizer(smiles, padding="max_length", truncation=True, max_length=drug_max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = chemberta_model(**inputs)
        features = outputs.last_hidden_state  # [1, drug_max_length, 384]
    return features

def extract_protein_features(sequence):
    inputs = prot_t5_tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=protein_max_length).to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = prot_t5_model(**inputs)
        features = outputs.last_hidden_state  # [1, protein_max_length, 1024]
    return features

# 模型定义
class SharedMultiheadCrossAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        self.W_q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

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

class Model(torch.nn.Module):
    def __init__(self, drug_dim=384, protein_dim=1024):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim

        self.drug_CNN = torch.nn.Sequential(
            torch.nn.Conv1d(self.drug_dim, conv, drug_kernel[0]),
            torch.nn.BatchNorm1d(conv),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv, conv * 2, drug_kernel[1]),
            torch.nn.BatchNorm1d(conv * 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv * 2, conv * 4, drug_kernel[2]),
            torch.nn.BatchNorm1d(conv * 4),
            torch.nn.ReLU(),
        )
        self.protein_CNN = torch.nn.Sequential(
            torch.nn.Conv1d(self.protein_dim, conv, protein_kernel[0]),
            torch.nn.BatchNorm1d(conv),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv, conv * 2, protein_kernel[1]),
            torch.nn.BatchNorm1d(conv * 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv * 2, conv * 4, protein_kernel[2]),
            torch.nn.BatchNorm1d(conv * 4),
            torch.nn.ReLU(),
        )
        self.drug_pool = torch.nn.MaxPool1d(drug_afterCNN)
        self.protein_pool = torch.nn.MaxPool1d(protein_afterCNN)
        self.attention = SharedMultiheadCrossAttention(attention_dim, mix_attention_head)

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(conv * 8, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 2)
        )

    def forward(self, drug_input, protein_input):
        drug = drug_input.permute(0, 2, 1)  # [batch_size, drug_dim, drug_max_length]
        protein = protein_input.permute(0, 2, 1)  # [batch_size, protein_dim, protein_max_length]
        drug_feat = self.drug_CNN(drug).permute(0, 2, 1)
        protein_feat = self.protein_CNN(protein).permute(0, 2, 1)
        drug_att, protein_att = self.attention(drug_feat, protein_feat)
        drug_att = self.drug_pool(drug_att.permute(0, 2, 1)).squeeze(2)
        protein_att = self.protein_pool(protein_att.permute(0, 2, 1)).squeeze(2)
        return self.fc(torch.cat([drug_att, protein_att], dim=1))

# 全局初始化 Model 实例
model = Model(drug_dim=384, protein_dim=1024).to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
state_dict = {k: v for k, v in state_dict.items() if 'embedding' not in k}  # 过滤掉嵌入层权重
model.load_state_dict(state_dict, strict=False)
model.eval()

# 数据库查找和特征提取逻辑
def get_drug_embedding(smiles):
    drug_data = fetch_drug_data(smiles)
    if drug_data and "error" not in drug_data:
        print(f"从数据库中找到药物: {smiles}")
    else:
        print("未在数据库中找到药物，使用 ChemBERTa 提取特征")
    return extract_drug_features(smiles)

def get_protein_embedding(sequence):
    protein_data = fetch_protein_data(sequence)
    if protein_data and "error" not in protein_data:
        print(f"从数据库中找到蛋白质: {sequence[:30]}...")
    else:
        print("未在数据库中找到蛋白质，使用 ProtT5 提取特征")
    return extract_protein_features(sequence)

# 回调函数
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

    if drug_input:
        smiles = drug_input.strip()
        print("药物 SMILES:", smiles)
        try:
            drug_features = get_drug_embedding(smiles)  # [1, 94, 384]
            drug_table_data = [{"Property": "SMILES", "Value": smiles}]
        except Exception as e:
            drug_table_data = [{"Property": "错误", "Value": f"提取药物特征失败: {str(e)}"}]

    if protein_input:
        sequence = protein_input.strip()
        print("蛋白质序列:", sequence[:30] + "..." if len(sequence) > 30 else sequence)
        try:
            protein_features = get_protein_embedding(sequence)  # [1, 1000, 1024]
            protein_table_data = [{"Property": "蛋白质序列", "Value": sequence}]
        except Exception as e:
            protein_table_data = [{"Property": "错误", "Value": f"提取蛋白质特征失败: {str(e)}"}]

    if drug_input and protein_input:
        print(f"药物特征形状: {drug_features.shape}")
        print(f"蛋白质特征形状: {protein_features.shape}")
        try:
            with torch.no_grad():
                logits = model(drug_features, protein_features)
                prob_interaction = torch.softmax(logits, dim=1)[0, 1].item()

            prediction_result = html.Div([
                html.H5("预测结果", style={"color": "green", "fontWeight": "bold"}),
                html.P(f"药物-靶点相互作用概率: {prob_interaction:.4f}"),
                html.P(f"阈值判断: {'相互作用' if prob_interaction >= 0.6 else '无相互作用'}")
            ])

        except Exception as e:
            prediction_result = html.Div([
                html.H5("预测错误", style={"color": "red", "fontWeight": "bold"}),
                html.P(f"错误: {str(e)}")
            ])

    return drug_table_data, protein_table_data, prediction_result