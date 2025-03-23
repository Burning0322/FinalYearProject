import dash
from dash import html,dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Drug Target Interaction", className="ms-2", style={"fontWeight": "bold", "fontSize": "24px"},id="navbar-title"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="#",id="navbar-home")),
            dbc.NavItem(dbc.NavLink("Resources", href="#",id="navbar-resources")),
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

layout = html.Div([
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
