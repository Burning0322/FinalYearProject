import dash
from dash import html,dcc,dash_table,Input,Output
import dash_bootstrap_components as dbc
import mysql.connector
from mysql.connector import Error

dash.register_page(__name__, path="/history")

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "dti"
}

def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except Error as e:
        print(f"数据库连接错误: {e}")
        return None

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

history = html.Div([
        html.H1("Prediction History", style={"textAlign": "center", "marginBottom": "20px"}),
            dash_table.DataTable(
                id="history-table",
                columns=[
                    {"name": "Drug SMILES", "id": "drug_smiles"},
                    {"name": "Protein Sequence", "id": "protein_sequence"},
                    {"name": "Prediction", "id": "prediction"},
                    {"name": "Timestamp", "id": "timestamp"}
                ],
                style_table={"overflowX": "auto", "margin": "0 auto", "width": "90%"},
                style_cell={"textAlign": "left", "padding": "5px", "whiteSpace": "normal"},
                page_size=10
            )
    ])

# 页面布局
layout = html.Div([
    navbar,
    history,
    footer
], style={"fontFamily": "'Roboto', sans-serif", "backgroundColor": "#ecf0f1"})


@dash.callback(
    Output("history-table", "data"),
    Input("history-table", "id")
)
def update_history_table(_):
    conn = get_db_connection()
    if conn and conn.is_connected():
        print("History 数据库连接成功")
        try:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM history ORDER BY timestamp DESC"
            cursor.execute(query)
            data = cursor.fetchall()
            print(f"查询到 {len(data)} 条记录")
            if not data:
                return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "prediction": "暂无历史记录", "timestamp": "N/A"}]
            return data
        except Error as e:
            print(f"查询历史记录失败: {e}")
            return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "prediction": f"查询失败: {e}", "timestamp": "N/A"}]
        finally:
            cursor.close()
            conn.close()
    else:
        print("History 数据库连接失败")
        return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "prediction": "数据库连接失败", "timestamp": "N/A"}]