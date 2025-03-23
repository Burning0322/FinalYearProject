import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 多语言内容字典
content = {
    "en": {
        "title": "Service Portfolio",
        "what_is_dti_title": "What is DTI",
        "what_is_dti_text": "Drug Target Interaction (DTI) refers to the interaction between drug molecules and their biological targets (e.g., proteins, enzymes, receptors, or nucleic acids). This interaction is the foundation of a drug's therapeutic effect, typically involving chemical binding, regulation, or inhibition of the target, thereby affecting specific physiological or pathological processes in the body. DTI research is a core component of drug discovery and development, widely applied in modern drug design and precision medicine.",
        "what_is_affinity_title": "Affinity",
        "what_is_affinity_text": "Affinity refers to the strength of the interaction between two molecules, such as a drug and its target protein. It is typically represented by the dissociation constant (Kd). A smaller Kd value indicates higher affinity, meaning the molecules bind more tightly. Affinity is influenced by molecular structure, non-covalent interactions (such as hydrogen bonds and van der Waals forces), and environmental conditions (such as temperature and pH).",
        "featured_usage_title": "How to use",
        "featured_usage_text": "Use our platform to predict drug-target interactions, visualize results, and gain insights for drug discovery.",
    },
    "cn": {
        "title": "服务组合",
        "what_is_dti_title": "什么是药物靶标相互作用",
        "what_is_dti_text": "药物靶点相互作用（DTI）是指药物分子与其生物靶点（如蛋白质、酶、受体或核酸）之间发生的相互作用。这种相互作用是药物发挥治疗作用的基础，通常涉及药物与靶点之间的化学结合、调控或抑制，从而影响生物体内特定的生理或病理过程。DTI的研究是药物发现和开发的核心环节，广泛应用于现代药物设计和精准医学。",
        "what_is_affinity_title": "什么是亲和力",
        "what_is_affinity_text": "亲和力是指两个分子（如药物与靶点蛋白）之间结合的强度，通常用 结合常数（Kd） 表示。Kd值越小，亲和力越高，说明分子间结合越紧密。亲和力受分子结构、非共价相互作用（如氢键、范德华力）和环境条件（如温度、pH值）影响。",
        "featured_usage_title": "如何使用",
        "featured_usage_text": "使用我们的平台预测药物-靶点相互作用，可视化结果，并获取有助于药物研发",
    }
}

# 导航栏
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Drug Target Interaction", className="ms-2", style={"fontWeight": "bold", "fontSize": "24px"}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("Resources", href="#")),
            dbc.NavItem(dbc.NavLink("DTI", href="#")),
            dbc.NavItem(dbc.NavLink("About Us", href="#")),
            dbc.NavItem(dbc.Button("Contact", color="primary", className="ms-2")),
            # 添加语言选择下拉菜单
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

# 页面布局
app.layout = html.Div([
    navbar,
    html.Div([
        html.H1(id="portfolio-title", className="text-center mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(id="what-is-dti-title", className="card-title"),
                    html.P(id="what-is-dti-text")
                ]),
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(id="what-is-affinity-title", className="card-title"),
                    html.P(id="what-is-affinity-text")
                ]),
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(id="featured-usage-title", className="card-title"),
                    html.P(id="featured-usage-text")
                ]),
            ]), md=4),
        ], className="p-4")
    ])
])

# 回调函数：根据语言选择更新内容
@app.callback(
    [Output("portfolio-title", "children"),
     Output("what-is-dti-title", "children"),
     Output("what-is-dti-text", "children"),
     Output("what-is-affinity-title", "children"),
     Output("what-is-affinity-text", "children"),
     Output("featured-usage-title", "children"),
     Output("featured-usage-text", "children")],
    [Input("language-dropdown", "value")]
)
def update_language(language):
    return (
        content[language]["title"],
        content[language]["what_is_dti_title"],
        content[language]["what_is_dti_text"],
        content[language]["what_is_affinity_title"],
        content[language]["what_is_affinity_text"],
        content[language]["featured_usage_title"],
        content[language]["featured_usage_text"]
    )

if __name__ == "__main__":
    app.run()