import dash
from dash import html,dcc,Output,Input
import dash_bootstrap_components as dbc
from dash import callback

dash.register_page(__name__, path="/")

styles = {
    "navbar": {
        "backgroundColor": "#2c3e50",  # 深色导航栏
        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        "padding": "10px 0",
    },
    "navbar-brand": {
        "color": "#ecf0f1",
        "fontWeight": "bold",
        "fontSize": "28px",
        "transition": "color 0.3s ease",
    },
    "navbar-link": {
        "color": "#ecf0f1",
        "fontSize": "16px",
        "marginLeft": "20px",
        "transition": "color 0.3s ease",
    },
    "navbar-link-hover": {
        "color": "#3498db",
    },
    "section": {
        "padding": "50px 0",
        "backgroundColor": "#f5f7fa",
        "borderRadius": "10px",
        "margin": "20px 0",
        "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.05)",
    },
    "section-title": {
        "color": "#2c3e50",
        "fontWeight": "bold",
        "fontSize": "32px",
        "marginBottom": "20px",
        "textAlign": "center",
    },
    "team-member": {
        "backgroundColor": "#ffffff",
        "borderRadius": "10px",
        "padding": "20px",
        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.1)",
        "textAlign": "center",
        "transition": "transform 0.3s ease",
    },
    "team-member-hover": {
        "transform": "translateY(-5px)",
    },
    "footer": {
        "backgroundColor": "#2c3e50",
        "color": "#ecf0f1",
        "padding": "40px 0",
        "marginTop": "40px",
    },
    "footer-link": {
        "color": "#ecf0f1",
        "textDecoration": "none",
        "marginRight": "20px",
        "transition": "color 0.3s ease",
    },
    "footer-link-hover": {
        "color": "#3498db",
    },
    "button": {
        "backgroundColor": "#3498db",
        "border": "none",
        "borderRadius": "5px",
        "padding": "10px 20px",
        "color": "#ffffff",
        "fontWeight": "bold",
        "transition": "backgroundColor 0.3s ease",
    },
    "button-hover": {
        "backgroundColor": "#2980b9",
    },
}

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

about = html.Div([
html.Div([
        html.H1(id="portfolio-title", className="text-center mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(id="what-is-dti-title", className="card-title"),
                    html.P(id="what-is-dti-text"),
                    html.Img(src="/assets/what-is-dti-img.jpg",
                             style={"height": "100%", "width": "100%"},)
                ]),
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(id="what-is-affinity-title", className="card-title"),
                    html.P(id="what-is-affinity-text"),
                    html.Img(src="/assets/what-is-affinity-img.jpg",
                             style={"height": "100%", "width": "100%"},)
                ]),
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(id="featured-usage-title", className="card-title"),
                    html.P(id="featured-usage-text"),
                    html.Img(src="/assets/featured-usage-img.jpg",
                             style={"height": "100%", "width": "100%"},)
                ]),
            ]), md=4),
        ], className="p-4")
    ]),
html.Div([
        html.H2("团队介绍", className="section-title", style=styles["section-title"]),
        html.Div([
            html.Div([
                html.H4("Low Ren Hong", style={"color": "#2c3e50"}),
                html.P("研究方向：药物-靶点交互预测、深度学习", style={"color": "#7f8c8d"}),
                html.P("浙江理工大学计算机科学与技术专业", style={"color": "#7f8c8d"}),
                html.A("GitHub", href="https://github.com/Burning0322/FinalYearProject.git", target="_blank", style={"color": "#3498db", "textDecoration": "none"}),
            ], className="team-member", style=styles["team-member"]),
        ], style={"display": "flex", "justifyContent": "center"})
    ], className="about-section", style=styles["section"]),

    html.Div([
        html.H2("项目背景与初衷", className="section-title", style=styles["section-title"]),
        html.P("""
        药物-靶点相互作用（Drug-Target Interaction, DTI）预测是药物发现领域的关键任务。
        然而，实验方法成本高、周期长。因此，我们开发了本平台以提供基于深度学习的高效预测工具，
        帮助科研人员与企业在前期筛选中快速定位潜在药物靶点，提升研发效率。
        """, style={"color": "#7f8c8d", "lineHeight": "1.6", "textAlign": "center"})
    ], className="about-section", style=styles["section"]),

    html.Div([
        html.H2("技术栈", className="section-title", style=styles["section-title"]),
        html.Ul([
            html.Li("🔷 Dash - 构建交互式 Web 应用", style={"color": "#7f8c8d", "marginBottom": "10px"}),
            html.Li("🔥 PyTorch - 构建 DTI 深度学习模型", style={"color": "#7f8c8d", "marginBottom": "10px"}),
            html.Li("🧪 RDKit - 化学结构解析与指纹提取", style={"color": "#7f8c8d", "marginBottom": "10px"}),
            html.Li("🧬 py3Dmol - 药物/蛋白三维结构可视化", style={"color": "#7f8c8d", "marginBottom": "10px"}),
        ], style={"textAlign": "center", "listStylePosition": "inside"})
    ], className="about-section", style=styles["section"]),

    html.Div([
        html.H2("引用与数据来源", className="section-title", style=styles["section-title"]),
        html.Ul([
            html.Li([
                "Davis 数据集：Davis MI, Hunt JP, Herrgard S, et al. Nature Biotechnology (2011). ",
                html.A("DOI: 10.1038/nbt.1990", href="https://doi.org/10.1038/nbt.1990", target="_blank", style={"color": "#3498db", "textDecoration": "none"})
            ], style={"color": "#7f8c8d", "marginBottom": "10px"}),
            html.Li([
                "KIBA 数据集：Tang J, Szwajda A, Shakyawar S, et al. Scientific Reports (2014). ",
                html.A("DOI: 10.1038/srep06455", href="https://doi.org/10.1038/srep06455", target="_blank", style={"color": "#3498db", "textDecoration": "none"})
            ], style={"color": "#7f8c8d", "marginBottom": "10px"}),
        ], style={"textAlign": "center", "listStylePosition": "inside"})
    ], className="about-section", style=styles["section"]),
])

layout = html.Div([
    navbar,
    about,
    footer
])


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

@callback(
    [Output("portfolio-title", "children"),
     Output("what-is-dti-title", "children"),
     Output("what-is-dti-text", "children"),
     Output("what-is-affinity-title", "children"),
     Output("what-is-affinity-text", "children"),
     Output("featured-usage-title", "children"),
     Output("featured-usage-text", "children"),
     ],
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
        content[language]["featured_usage_text"],
    )
