import dash
from dash import html, dcc, dash_table, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import py3Dmol
import mysql.connector
import os
from sqlalchemy import create_engine

dash.register_page(__name__, path="/protein")

# ------------------ Database config ------------------
db_config = {
    "host": "localhost",
    "user": "root",
    "port": 3306,
    "password": "root",
    "database": "dti"
}
#
# def get_protein_data():
#     try:
#         conn = mysql.connector.connect(**db_config)
#         query = """
#             SELECT
#                 uniprot_id AS "UniProt ID",
#                 uniprot_accession AS 'UniProt Accession',
#                 gene_names AS 'Gene Name',
#                 organism AS 'Organism',
#                 protein_name AS 'Protein Name',
#                 sequence AS 'Sequence',
#                 length AS 'Length'
#             FROM protein;
#         """
#         df = pd.read_sql(query, conn)
#         conn.close()
#         return df
#     except Exception as e:
#         print(f"Database error: {e}")
#         return pd.DataFrame()

def get_protein_data():
    try:
        # 构建 SQLAlchemy 数据库引擎
        engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        query = """
            SELECT
                uniprot_id AS "UniProt ID",
                uniprot_accession AS 'UniProt Accession',
                gene_names AS 'Gene Name',
                organism AS 'Organism',
                protein_name AS 'Protein Name',
                sequence AS 'Sequence',
                length AS 'Length'
            FROM protein;
        """

        df = pd.read_sql(query, engine)
        return df

    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

# ------------------ Load protein data ------------------
df = get_protein_data()

# ------------------ Load AlphaFold log ------------------
log_file = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/alphafold_download_log.csv"
log_df = pd.read_csv(log_file)

# 创建映射字典：UniProt Accession -> 是否有 PDB 文件
# 不仅依赖日志状态，还要检查文件是否实际存在
has_pdb_dict = {}
for _, row in log_df.iterrows():
    accession = row["UniProt"]
    file_path = os.path.join("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/pdb", f"AF-{accession}-F1.pdb")
    # 只有当日志状态为 ✅ 且文件实际存在时，才标记为 True
    has_pdb_dict[accession] = row["Status"].startswith("✅") and os.path.exists(file_path)

# 为 DataFrame 添加 "3D Structure" 列
df["3D Structure"] = df["UniProt Accession"].apply(
    lambda acc: f"[View 3D](#)" if has_pdb_dict.get(acc, False) else "无 3D 结构"
)

# ------------------ Navbar ------------------
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
            dbc.NavItem(dbc.NavLink("History", href="/history", id="navbar-history")),
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

# ------------------ Footer ------------------
footer = html.Footer([
    html.Div([
        html.Span("Follow us", style={"fontWeight": "bold", "marginRight": "10px"}),
        html.A(html.I(className="bi bi-link"), href="https://www.zstu.edu.cn", target="_blank",
               style={"marginRight": "10px", "color": "black"}),
        html.A(html.I(className="bi bi-github"), href="https://github.com/Burning0322/FinalYearProject.git",
               target="_blank", style={"color": "black"}),
    ], style={"textAlign": "center", "padding": "10px 0"}),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.H5("About"),
            html.Ul([
                html.Li(html.A("About DTI", href="#")),
                html.Li(html.A("DTI Project", href="https://github.com/Burning0322/FinalYearProject.git")),
                html.Li(html.A("Research", href="https://github.com/Burning0322/FinalYearProject.git")),
                html.Li(html.A("Team", href="https://github.com/Burning0322/FinalYearProject.git")),
                html.Li(html.A("Contact Us", href="#")),
            ], style={"listStyleType": "none", "padding": 0}),
        ], md=3),
        dbc.Col([
            html.H5("Learn more"),
            html.Ul([
                html.Li(html.A("Davis", href="#")),
                html.Li(html.A("Kiba", href="#")),
            ], style={"listStyleType": "none", "padding": 0}),
        ], md=3),
        dbc.Col([
            html.H5("Sign up for updates"),
            dcc.Input(placeholder="Email address", type="email", style={"width": "100%", "padding": "8px"}),
            html.Div([
                html.Small([
                    "I accept ZSTU's ",
                    html.A("Terms and Conditions", href="#"),
                    " and Privacy Policy."
                ], style={"display": "block", "marginBottom": "10px"}),
                html.Button("Sign up", className="btn btn-primary", style={"width": "100%"})
            ])
        ], md=4)
    ], className="p-4"),

    html.Hr(),

    html.Div([
        html.Span("ZSTU", style={"fontWeight": "bold", "marginRight": "20px"}),
        html.A("About ZSTU", href="https://www.zstu.edu.cn", style={"marginRight": "20px"}),
        html.A("Privacy", href="#", style={"marginRight": "20px"}),
        html.A("Terms", href="#"),
    ], style={"textAlign": "center", "padding": "10px 0"})
], style={"backgroundColor": "#f8f9fa", "padding": "20px 0", "marginTop": "20px"})

# ------------------ DataTable ------------------
table = dash_table.DataTable(
    id="protein-table",
    data=df.to_dict('records'),
    columns=[
        {"name": col, "id": col, "presentation": "markdown" if col == "3D Structure" else "input"}
        for col in df.columns
    ],
    style_table={"overflowX": "auto", "width": "100%"},
    style_cell={
        "textAlign": "left",
        "whiteSpace": "normal",
        "height": "auto",
        "maxWidth": "300px",
        "overflow": "auto"
    },
    style_data={"minHeight": "200px", "whiteSpace": "pre-wrap"},
    page_size=10,
    markdown_options={"html": True},
)

# ------------------ Modal for 3D Structure ------------------
modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("3D Structure")),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="modal-3d",
    size="lg",
    is_open=False,
)

# ------------------ Final Layout ------------------
layout = html.Div([
    # 确保加载本地 3dmol.js
    html.Script(src="/assets/3dmol-min.js"),
    navbar,
    html.H1("Protein List", style={"margin": "20px"}),
    dbc.Container([table, modal]),
    footer
])

# ------------------ Callback for 3D Structure Modal ------------------
@callback(
    Output("modal-3d", "is_open"),
    Output("modal-body", "children"),
    Input("protein-table", "active_cell"),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True
)
def display_3d_structure(active_cell, close_clicks):
    if active_cell and active_cell["column_id"] == "3D Structure":
        row = active_cell["row"]
        accession = df.iloc[row]["UniProt Accession"]

        # 检查是否有 PDB 文件
        if has_pdb_dict.get(accession, False):
            # 构造 PDB 文件路径
            pdb_file = f"/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/pdb/AF-{accession}-F1.pdb"
            if os.path.exists(pdb_file):
                try:
                    with open(pdb_file, "r") as f:
                        pdb_data = f.read()

                    # 使用 py3Dmol 渲染 3D 结构
                    viewer = py3Dmol.view(width=600, height=400)
                    viewer.addModel(pdb_data, "pdb")
                    viewer.setStyle({"cartoon": {"color": "spectrum"}})
                    viewer.zoomTo()
                    viewer_html = viewer._make_html()

                    return True, html.Iframe(srcDoc=viewer_html, style={"width": "100%", "height": "400px", "border": "none"})
                except Exception as e:
                    return True, html.Div(f"Error rendering 3D structure for {accession}: {str(e)}")
            else:
                return True, html.Div(f"Error: PDB file for {accession} not found on server.")
        else:
            return True, html.Div(f"无 3D 结构 for {accession}")

    return False, None