import dash
from dash import html,dcc,dash_table,Input,Output,State
import dash_bootstrap_components as dbc
import mysql.connector
from mysql.connector import Error
import re

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
            dbc.NavItem(dbc.NavLink("Resources", href="/resources",id="navbar-resources")),
            dbc.NavItem(dbc.NavLink("DTI", href="#",id="navbar-dti")),
            dbc.NavItem(dbc.NavLink("About Us", href="#",id="navbar-about")),
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
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            query = "SELECT * FROM drug WHERE smiles = %s"
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, (smiles,))
            result = cursor.fetchone()
            return result
    except Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


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