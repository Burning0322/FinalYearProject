import dash
from dash import html,dcc,dash_table
import dash_bootstrap_components as dbc
import pandas as pd

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

df = pd.read_csv("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/web/drug.csv")

table = dash_table.DataTable(
    columns=[
        {"name": "Query", "id": "Query", "presentation": "markdown"},
        {"name": "PubChem CID", "id": "PubChem CID"},
        {"name": "Weight (g/mol)", "id": "Molecular Weight"},
        {"name": "Molecular Formula", "id": "Molecular Formula", "presentation": "markdown"},
    ],
    data=df.to_dict("records"),
    style_cell={"textAlign": "left", "padding": "5px"},
    style_header={"backgroundColor": "#f5f5f5", "fontWeight": "bold"},
    style_table={"overflowX": "auto"},
    markdown_options={"html": True},
)

layout = html.Div([
    navbar,
    html.H1("Drug List"),
    dbc.Container([
        table
    ]),
    footer
])
