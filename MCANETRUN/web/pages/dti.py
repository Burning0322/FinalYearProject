import dash
from dash import html,dcc,dash_table,Input,Output,State
import dash_bootstrap_components as dbc
import mysql.connector
from mysql.connector import Error
import re
import requests

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
    # Drug 输入
    dbc.Col([
        html.Label("Drug", style={"fontWeight": "bold", "fontSize": "18px"}),
        dcc.Input(
            id="drug-input",
            type="text",
            placeholder="Enter SMILES (e.g., CCO for ethanol)",
            style={"width": "100%", "padding": "8px", "marginBottom": "10px"}
        ),
        dcc.Upload(
            id="drug-upload",
            children=html.Button("Upload Drug File (.sdf)", className="btn btn-secondary"),
            accept=".sdf",
            style={"marginBottom": "20px"}
        )
    ], md=6),

    # Protein 输入
    dbc.Col([
        html.Label("Protein", style={"fontWeight": "bold", "fontSize": "18px"}),
        dcc.Input(
            id="protein-input",
            type="text",
            placeholder="Enter Protein Name or Sequence",
            style={"width": "100%", "padding": "8px", "marginBottom": "10px"}
        ),
        dcc.Upload(
            id="protein-upload",
            children=html.Button("Upload Protein File (.pdb)", className="btn btn-secondary"),
            accept=".pdb"
        )
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
        {"name": "Value", "id": "Value", "presentation": "markdown"}  # 启用markdown渲染
    ],
    data=[],
    style_table={
        "width": "100%",
        "maxHeight": "400px",
        "overflowY": "auto"
    },
    style_cell={
        "textAlign": "left",
        "padding": "5px",
        "whiteSpace": "normal"
    },
    markdown_options={"html": True},  # 允许HTML标签
    tooltip_duration=None
)

# 修改回调函数中的格式化逻辑
def format_molecular_formula(formula):
    """将C18H22N4O2转换为C₁₈H₂₂N₄O₂"""
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(r'([A-Za-z])(\d+)', lambda m: m.group(1) + m.group(2).translate(subscript_map), formula)


protein_details_table = dash_table.DataTable(
    id="protein-details-table",
    columns=[
        {"name": "Property", "id": "Property"},
        {"name": "Value", "id": "Value"}
    ],
    data=[],
    style_table={"width": "100%"},
    style_cell={"textAlign": "left"}
)

details_section = dbc.Row([
    dbc.Col([
        html.H4("Drug Details", style={"fontWeight": "bold"}),
        drug_details_table
    ], md=6),
    dbc.Col([
        html.H4("Protein Details", style={"fontWeight": "bold"}),
        protein_details_table
    ], md=6)
])

layout = html.Div([
    navbar,
    dbc.Container([
        input_section,
        predict_button,
        details_section
    ]),
    footer
])


def fetch_drug_data(smiles):
    try:
        # First try database
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            query = "SELECT * FROM drug WHERE smiles = %s"
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, (smiles,))
            result = cursor.fetchone()
            if result:
                return result
    except Error as e:
        print(f"数据库错误: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

    # If not in database, fetch from PubChem
    try:
        # Basic properties request
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
                cid = str(props.get('CID', 'N/A'))  # Compound ID

                # Map PubChem properties to database fields
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
                    # 3D properties and others not available in basic properties endpoint
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
                    'fingerprint_2d': None  # Requires separate fingerprint endpoint
                }

                # Optional: Fetch 3D conformer data (limited availability)
                conformer_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/conformers/JSON"
                conformer_response = requests.get(conformer_url, timeout=10)
                if conformer_response.status_code == 200:
                    conformer_data = conformer_response.json()
                    if 'PC_Compounds' in conformer_data:
                        conformer = conformer_data['PC_Compounds'][0]
                        result['conformer_count_3d'] = str(len(conformer.get('conformers', [])))

                try:
                    connection = mysql.connector.connect(**db_config)
                    if connection.is_connected():
                        cursor = connection.cursor()
                        insert_query = """
                            INSERT INTO drug (
                                query, compound_id, molecular_formula, molecular_weight, smiles,
                                canonical_smiles, isomeric_smiles, iupac_name, inchi, inchi_key,
                                xlogp, exact_mass, monoisotopic_mass, tpsa, complexity, charge,
                                h_bond_donor_count, h_bond_acceptor_count, rotatable_bond_count,
                                heavy_atom_count, isotope_atom_count, defined_atom_stereo_count,
                                undefined_atom_stereo_count, defined_bond_stereo_count,
                                undefined_bond_stereo_count, covalent_unit_count, conformer_count_3d
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                molecular_formula = VALUES(molecular_formula),
                                molecular_weight = VALUES(molecular_weight),
                                canonical_smiles = VALUES(canonical_smiles),
                                isomeric_smiles = VALUES(isomeric_smiles),
                                iupac_name = VALUES(iupac_name),
                                inchi = VALUES(inchi),
                                inchi_key = VALUES(inchi_key),
                                xlogp = VALUES(xlogp),
                                exact_mass = VALUES(exact_mass),
                                monoisotopic_mass = VALUES(monoisotopic_mass),
                                tpsa = VALUES(tpsa),
                                complexity = VALUES(complexity),
                                charge = VALUES(charge),
                                h_bond_donor_count = VALUES(h_bond_donor_count),
                                h_bond_acceptor_count = VALUES(h_bond_acceptor_count),
                                rotatable_bond_count = VALUES(rotatable_bond_count),
                                heavy_atom_count = VALUES(heavy_atom_count),
                                isotope_atom_count = VALUES(isotope_atom_count),
                                defined_atom_stereo_count = VALUES(defined_atom_stereo_count),
                                undefined_atom_stereo_count = VALUES(undefined_atom_stereo_count),
                                defined_bond_stereo_count = VALUES(defined_bond_stereo_count),
                                undefined_bond_stereo_count = VALUES(undefined_bond_stereo_count),
                                covalent_unit_count = VALUES(covalent_unit_count),
                                conformer_count_3d = VALUES(conformer_count_3d)
                        """
                        cursor.execute(insert_query, tuple(result.values()))
                        connection.commit()
                except Error as e:
                    print(f"保存到数据库时出错: {e}")
                finally:
                    if 'connection' in locals() and connection.is_connected():
                        cursor.close()
                        connection.close()

                return result
            else:
                return {"error": "药物不存在", "smiles": smiles}
        else:
            print(f"PubChem API returned status code: {response.status_code}")
            return {"error": "药物不存在", "smiles": smiles}

    except requests.RequestException as e:
        print(f"在线获取数据失败: {e}")

    return {"error": "药物不存在", "smiles": smiles}


@dash.callback(
    Output("drug-details-table", "data"),
    Input("predict-button", "n_clicks"),
    State("drug-input", "value"),
    prevent_initial_call=True
)
def update_drug_details(n_clicks, drug_input):
    if not drug_input:
        return []

    smiles = drug_input.strip()
    drug_data = fetch_drug_data(smiles)

    if not drug_data:
        return [{"Property": "Error", "Value": f"No data found for SMILES: {smiles}"}]

    table_data = []
    for key, value in drug_data.items():
        if value is None:
            value = "N/A"

        # 特殊处理分子式
        if key == "molecular_formula":
            value = format_molecular_formula(value)
            # 使用Markdown的HTML渲染
            value = f"<span style='font-size:1.1em'>{value}</span>"

        table_data.append({
            "Property": key.replace("_", " ").title(),
            "Value": str(value)
        })

    return table_data