import dash
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import mysql.connector
from mysql.connector import Error
import pandas as pd

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

history = html.Div([
    html.H1("Prediction History", style={"textAlign": "center", "marginBottom": "20px"}),
    dbc.Row([
        dbc.Col([
            html.Button("Select All", id="select-all", className="btn btn-info", style={"marginRight": "10px", "backgroundColor": "#17a2b8", "color": "white"}),
            html.Button("Clear All", id="clear-history", className="btn btn-danger", style={"backgroundColor": "red", "color": "white", "marginRight": "10px"}),
            html.Button("Delete Selected", id="delete-selected", className="btn btn-warning", style={"backgroundColor": "orange", "color": "white", "marginRight": "10px"}),
            html.Button("Export CSV", id="export-csv", className="btn btn-success", style={"backgroundColor": "green", "color": "white"}),
        ], width=12, className="text-right mb-3"),
        dcc.ConfirmDialog(id="confirm-clear", message="确定要清除所有历史记录吗？此操作无法撤销。"),
        dcc.ConfirmDialog(id="confirm-delete-selected", message="确定要删除选中的历史记录吗？此操作无法撤销。"),
        dcc.Download(id="download-csv")
    ]),
    dash_table.DataTable(
        id="history-table",
        columns=[
            {"name": "Drug SMILES", "id": "drug_smiles"},
            {"name": "Protein Sequence", "id": "protein_sequence"},
            {"name": "Probability", "id": "probability"},
            {"name": "Prediction", "id": "prediction"},
            {"name": "Timestamp", "id": "timestamp"}
        ],
        style_cell={
            "textAlign": "center",
            "padding": "5px",
            "whiteSpace": "normal",
            "overflow": "hidden",
            "maxWidth": "100px",
            "height": "50px",
            "border": "1px solid #ddd"
        },
        style_cell_conditional=[
            {"if": {"column_id": "drug_smiles"}, "overflowX": "auto"},
            {"if": {"column_id": "protein_sequence"}, "overflowX": "auto"}
        ],
        page_size=10,
        row_selectable="multi",  # 允许多选
        selected_rows=[]  # 默认无选中行
    )
])

layout = html.Div([navbar, history, footer], style={"fontFamily": "'Roboto', sans-serif", "backgroundColor": "#ecf0f1"})

@dash.callback(
    Output("history-table", "selected_rows"),
    Input("select-all", "n_clicks"),
    State("history-table", "data"),
    State("history-table", "selected_rows"),
    prevent_initial_call=True
)
def toggle_select_all_rows(n_clicks, table_data, current_selected_rows):
    if not table_data:
        return []  # 如果表格数据为空，返回空列表

    # 根据 n_clicks 的奇偶性判断是选中还是取消选中
    if n_clicks % 2 == 1:  # 奇数次点击：选中所有行
        return list(range(len(table_data)))
    else:  # 偶数次点击：取消选中所有行
        return []


# 回调 1：加载历史记录表格数据
@dash.callback(
    Output("history-table", "data"),
    Input("history-table", "id")
)
def update_history_table(_):
    conn = get_db_connection()
    if conn and conn.is_connected():
        try:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM history ORDER BY timestamp DESC"
            cursor.execute(query)
            data = cursor.fetchall()
            print(f"查询到 {len(data)} 条记录")
            if not data:
                return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "probability": "N/A", "prediction": "暂无历史记录", "timestamp": "N/A"}]
            return data
        except Error as e:
            print(f"查询历史记录失败: {e}")
            return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "probability": "N/A", "prediction": f"查询失败: {e}", "timestamp": "N/A"}]
        finally:
            cursor.close()
            conn.close()
    else:
        print("History 数据库连接失败")
        return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "probability": "N/A", "prediction": "数据库连接失败", "timestamp": "N/A"}]

# 回调 2：显示“Clear All”确认对话框
@dash.callback(
    Output("confirm-clear", "displayed"),
    Input("clear-history", "n_clicks"),
    prevent_initial_call=True
)
def display_clear_dialog(n_clicks):
    return True if n_clicks else False

# 回调 3：显示“Delete Selected”确认对话框
@dash.callback(
    Output("confirm-delete-selected", "displayed"),
    Input("delete-selected", "n_clicks"),
    State("history-table", "selected_rows"),
    prevent_initial_call=True
)
def display_delete_selected_dialog(n_clicks, selected_rows):
    if n_clicks and selected_rows:  # 只有点击按钮且有选中行时显示对话框
        return True
    return False

# 回调 4：执行“Delete Selected”操作
@dash.callback(
    Output("history-table", "data", allow_duplicate=True),
    Input("confirm-delete-selected", "submit_n_clicks"),
    State("history-table", "selected_rows"),
    State("history-table", "data"),
    prevent_initial_call=True
)
def delete_selected(submit_n_clicks, selected_rows, current_data):
    if submit_n_clicks and selected_rows:
        conn = get_db_connection()
        if conn and conn.is_connected():
            try:
                cursor = conn.cursor()
                # 假设每条记录有唯一的 id 字段，从 current_data 中获取选中行的 id
                selected_ids = [current_data[i]["id"] for i in selected_rows]
                query = f"DELETE FROM history WHERE id IN ({','.join(map(str, selected_ids))})"
                cursor.execute(query)
                conn.commit()
                print(f"已删除 {len(selected_ids)} 条记录")
                # 重新加载数据
                cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
                data = cursor.fetchall()
                return data if data else [{"drug_smiles": "N/A", "protein_sequence": "N/A", "probability": "N/A", "prediction": "暂无历史记录", "timestamp": "N/A"}]
            except Error as e:
                print(f"删除失败: {e}")
                return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "probability": "N/A", "prediction": f"删除失败: {e}", "timestamp": "N/A"}]
            finally:
                cursor.close()
                conn.close()
        else:
            print("数据库连接失败")
            return [{"drug_smiles": "N/A", "protein_sequence": "N/A", "probability": "N/A", "prediction": "数据库连接失败", "timestamp": "N/A"}]
    return dash.no_update

# 回调 5：导出 CSV
@dash.callback(
    Output("download-csv", "data"),
    Input("export-csv", "n_clicks"),
    State("history-table", "selected_rows"),
    State("history-table", "data"),
    prevent_initial_call=True
)
def export_selected_csv(n_clicks, selected_rows, current_data):
    print(f"按钮点击次数: {n_clicks}, 选中行: {selected_rows}")
    if n_clicks and selected_rows:
        selected_data = [current_data[i] for i in selected_rows]
        if selected_data:
            df = pd.DataFrame(selected_data)
            return dcc.send_data_frame(df.to_csv, "selected_history.csv", index=False)
    print("未选中行或未点击按钮，无法导出")
    return None