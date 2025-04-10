import dash
import torch.backends.mps
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import mysql.connector
from mysql.connector import Error
import re
import requests
import pandas as pd

dash.register_page(__name__, path="/dti")

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "dti"
}

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Drug Target Interaction", className="ms-2", style={"fontWeight": "bold", "fontSize": "24px"},id="navbar-title"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="/",id="navbar-home")),
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
            dbc.NavItem(dbc.NavLink("DTI", href="/dti",id="navbar-dti")),
            dbc.NavItem(dbc.NavLink("About Us", href="/about",id="navbar-about")),
            dbc.NavItem(dbc.Button("Contact", color="primary", className="ms-2",id="navbar-contact")),
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
            dbc.NavItem(dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Drug", href="/drug"),
                    dbc.DropdownMenuItem("Protein", href="/protein")
                ],
                nav=True,
                in_navbar=True,
                label="Resources",
                id="navbar-resources-dropdown"
            )),
        ], className="ms-auto", navbar=True)
    ]),
    color="light",
    dark=False,
    sticky="top"
)

footer = html.Footer([
    html.Div([
        html.Span("Follow us", style={"fontWeight": "bold", "marginRight": "10px"}),
        html.A(html.I(className="bi bi-link"), href="https://www.zstu.edu.cn", target="_blank", style={"marginRight": "10px", "color": "black"}),
        html.A(html.I(className="bi bi-github"), href="https://github.com/Burning0322/FinalYearProject.git", target="_blank", style={"color": "black"}),
    ], style={"textAlign": "center", "padding": "10px 0"}),

    # 分割线
    html.Hr(),

    # 链接和订阅表单部分
    dbc.Row([
        # 左侧链接列
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

        # 右侧订阅表单
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

    # 分割线
    html.Hr(),

    # 底部版权信息
    html.Div([
        html.Span("ZSTU", style={"fontWeight": "bold", "marginRight": "20px"}),
        html.A("About ZSTU", href="https://www.zstu.edu.cn", style={"color": "black", "textDecoration": "none", "marginRight": "20px"}),
        html.A("Privacy", href="#", style={"color": "black", "textDecoration": "none", "marginRight": "20px"}),
        html.A("Terms", href="#", style={"color": "black", "textDecoration": "none"}),
    ], style={"textAlign": "center", "padding": "10px 0"})
], style={"backgroundColor": "#f8f9fa", "padding": "20px 0", "marginTop": "20px"})


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
            placeholder="Enter UniProt Accession (e.g., P00533)",
            style={"width": "100%", "padding": "8px", "marginBottom": "10px"}
        ),
    ], md=6)
], className="mb-4")

predict_button = html.Button(
    "Predict",
    id="predict-button",
    className="btn btn-primary",
    style={"width": "200px", "marginBottom": "20px"}
)

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

details_section = dbc.Row([
    dbc.Col([html.H4("Drug Details", style={"fontWeight": "bold"}), drug_details_table], md=6),
    dbc.Col([html.H4("Protein Details", style={"fontWeight": "bold"}), protein_details_table], md=6)
])

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

layout = html.Div([navbar, dbc.Container([input_section, predict_button, details_section,prediction_result]), footer])

# Helper function for database connection
def get_db_connection():
    return mysql.connector.connect(**db_config)

def format_molecular_formula(formula):
    """Convert C18H22N4O2 to C₁₈H₂₂N₄O₂"""
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(r'([A-Za-z])(\d+)', lambda m: m.group(1) + m.group(2).translate(subscript_map), formula)

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

def fetch_protein_data(sequence):
    conn = None
    try:
        # 首先尝试从数据库中查找 sequence
        conn = get_db_connection()
        if conn.is_connected():
            query = "SELECT * FROM protein WHERE sequence = %s"
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (sequence,))
            result = cursor.fetchone()
            if result:
                return result  # 返回从数据库中获取到的蛋白质数据
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


    # 如果数据库中没有找到数据，使用 UniProt API 获取蛋白质数据
    try:
        # 使用 UniProt REST API 搜索序列
        url = f"https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=sequence:{sequence}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            # 解析返回的数据（假设为TSV格式）
            lines = response.text.splitlines()
            if len(lines) > 1:  # 如果返回的数据包含结果
                data = lines[1].split('\t')  # 假设数据是用tab分隔的，获取第一行数据
                protein_data = {
                    'uniprot_accession': data[0],
                    'uniprot_id': data[1],
                    'protein_name': data[2],
                    'gene_names': data[3],
                    'organism': data[4],
                    'sequence': sequence,
                    'length': len(sequence)
                }

                # 保存数据到数据库
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

                return protein_data  # 返回从UniProt获取的蛋白质数据
            else:
                return {"error": "未找到对应的蛋白质数据"}
        else:
            print(f"Error: {response.status_code}")
            return {"error": "无法通过序列获取数据"}
    except requests.RequestException as e:
        print(f"在线获取蛋白质数据失败: {e}")
        return {"error": "蛋白质不存在"}

try:
    from pages.renhongmodel import Model
    from pages.renhongmodel import Dataset
except ImportError as e:
    print(f"Failed to import Model: {e}")
    raise

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/final/best_model_fold_5.pth"

dataset = Dataset("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/Davis.txt")

drug_embedding_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/ligands_davis.pt"
protein_embedding_path = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/protein_davis.pt"

drug_embedding_tensor = torch.load(drug_embedding_path, map_location=device)
protein_embedding_tensor = torch.load(protein_embedding_path, map_location=device)

model = Model(drug_embedding_tensor, protein_embedding_tensor)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def smiles_to_idx(smiles):
    smiles = smiles.strip()
    if smiles not in smiles2idx:
        raise ValueError(f"Unknown SMILES: {smiles}")
    return torch.tensor([smiles2idx[smiles]], dtype=torch.long, device=device)

def protein_sequence_to_idx(sequence):
    sequence = sequence.strip()
    if sequence not in protein2idx:
        raise ValueError(f"Unknown protein sequence: {sequence}")
    return torch.tensor([protein2idx[sequence]], dtype=torch.long, device=device)

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

    # Fetch drug data
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
                drug_table_data.append({
                    "Property": key.replace("_", " ").title(),
                    "Value": str(value)
                })
    else:
        drug_table_data = [{"Property": "Error", "Value": "Please enter a SMILES string"}]

    if protein_input:
        accession = protein_input.strip()
        protein_data = fetch_protein_data(accession)
        if "error" in protein_data:
            protein_table_data = [{"Property": "Error", "Value": f"No data found for UniProt Accession: {accession}"}]
        else:
            for key, value in protein_data.items():
                if value is None:
                    value = "N/A"
                protein_table_data.append({
                    "Property": key.replace("_", " ").title(),
                    "Value": str(value)
                })
    else:
        protein_table_data = [{"Property": "Error", "Value": "Please enter a UniProt Accession"}]

    if drug_input and protein_input and not any("error" in d for d in [drug_data, protein_data]):
        try:
            # 获取蛋白质序列（从数据库或API返回中提取，假设字段为"sequence"）
            protein_sequence = protein_data.get("sequence", "")
            if not protein_sequence:
                raise ValueError("Protein sequence not found in database")

            # 转换为模型所需的索引
            drug_idx = smiles_to_idx(drug_input)
            protein_idx = protein_sequence_to_idx(protein_sequence)

            # 模型预测
            with torch.no_grad():
                logits = model(drug_idx, protein_idx)
                prob_interaction = torch.softmax(logits, dim=1)[0, 1].item()

            prediction_result = html.Div([
                html.H5("Prediction Result", style={"color": "green", "fontWeight": "bold"}),
                html.P(f"Probability of Drug-Target Interaction: {prob_interaction:.4f}"),
                html.P(f"Threshold (if binary): {'Interacts' if prob_interaction >= 0.6 else 'Does not interact'}")
            ])

        except ValueError as ve:
            prediction_result = html.Div([
                html.H5("Input Error", style={"color": "red", "fontWeight": "bold"}),
                html.P(f"Error: {str(ve)}")
            ])
        except Exception as e:
            prediction_result = html.Div([
                html.H5("Prediction Error", style={"color": "red", "fontWeight": "bold"}),
                html.P(f"Model error: {str(e)}")
            ])

    return drug_table_data, protein_table_data, prediction_result