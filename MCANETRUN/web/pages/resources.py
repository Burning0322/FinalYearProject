import dash
from dash import html,dcc,dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import py3Dmol

dash.register_page(__name__, path="/resources")

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

import re
import os
from rdkit import Chem
from rdkit.Chem import Draw,AllChem

def format_formula(formula):
    return re.sub(r"([A-Z]+)(\d+)", r"\1<sub>\2</sub>", formula)
df = pd.read_csv("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/web/drug.csv")

df["Formatted Formula"] = df["Molecular Formula"].apply(format_formula)

sdf_2d = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/2D"
sdf_3d = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/3D"
assets = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/web/assets"

df["SDF File"] = df["PubChem CID"].apply(lambda cid: f"Structure2D_COMPOUND_CID_{cid}.sdf")
df["SDF 3D File"] = df["PubChem CID"].apply(lambda cid: f"Structure3D_COMPOUND_CID_{cid}.sdf")

def generate_2d_image(sdf_filename, output_filename):
    sdf_path = os.path.join(sdf_2d, sdf_filename)
    output_path = os.path.join(assets, output_filename)
    if os.path.exists(output_path):
        return output_filename
    try:
        if not os.path.exists(sdf_path):
            print(f"SDF file not found: {sdf_path}")
            return None
        supplier = Chem.SDMolSupplier(sdf_path)
        mol = next((m for m in supplier if m is not None), None)
        if mol:
            Draw.MolToFile(mol, output_path, size=(200, 200))
            return output_filename
        else:
            print(f"No valid molecule found in {sdf_filename}")
    except Exception as e:
        print(f"Error generating 2D image for {sdf_filename}: {e}")
    return None

for index, row in df.iterrows():
    sdf_filename = row["SDF File"]
    output_filename = f"{sdf_filename.split('.')[0]}_2d_{index}.png"
    generate_2d_image(sdf_filename, output_filename)

def format_2d_structure(sdf_filename, index):
    output_filename = f"{sdf_filename.split('.')[0]}_2d_{index}.png"
    output_path = os.path.join(assets, output_filename)
    if os.path.exists(output_path):
        return f'<img src="/assets/{output_filename}" style="width: 100px; height: auto;" />'
    return "2D structure not available"

df["Structure 2D"] = df.apply(
    lambda row: f'![2D](/assets/{row["SDF File"].split(".")[0]}_2d_{row.name}.png)',
    axis=1
)


df["Structure 3D"] = df["PubChem CID"].apply(
    lambda cid: f'<iframe src="/assets/Structure3D_COMPOUND_CID_{cid}_3d.html" style="width: 200px; height: 200px; border: none;"></iframe>'
)

def get_3d_iframe(sdf_filename):
    sdf_path = os.path.join(sdf_3d, sdf_filename)
    try:
        if not os.path.exists(sdf_path):
            return html.Div("3D structure not available")

        supplier = Chem.SDMolSupplier(sdf_path)
        mol = next((m for m in supplier if m is not None), None)
        if mol is None:
            return html.Div("No valid molecule")

        sdf_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=200, height=200)
        viewer.addModel(sdf_block, 'sdf')
        viewer.setStyle({'stick': {'radius': 0.1}, 'sphere': {'scale': 0.3}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        viewer_html = viewer._make_html()

        return html.Iframe(
            srcDoc=viewer_html,
            style={"width": "200px", "height": "200px", "border": "none"}
        )

    except Exception as e:
        return html.Div(f"3D error: {e}")

table = dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[
        {"name": "Query", "id": "Query", "type": "text"},
        {"name": "PubChem CID", "id": "PubChem CID", "type": "text"},
        {"name": "Weight (g/mol)", "id": "Molecular Weight", "type": "text"},
        {"name": "Molecular Formula", "id": "Formatted Formula", "type": "text", "presentation": "markdown"},
        {"name": "Structure 2D", "id": "Structure 2D", "type": "text", "presentation": "markdown"},
        {"name": "Structure 3D", "id": "Structure 3D", "type": "text", "presentation": "markdown"}
    ],
    style_table={
        "overflowX": "auto",  # 水平滚动
        "width": "100%"
    },
    style_cell={
        "textAlign": "left",
        "whiteSpace": "normal",
        "height": "auto"
    },
    style_data={
        "minHeight": "200px"
    },
    page_size=10,
    markdown_options={"html": True},
)

layout = html.Div([
    navbar,
    html.H1("Drug List"),
    dbc.Container([
        table,
        html.Div(id="3d-viewer", style={"width": "400px", "height": "400px"})
    ]),
    footer
])
