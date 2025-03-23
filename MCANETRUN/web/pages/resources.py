import dash
from dash import html

dash.register_page(__name__, path="/resources")

layout = html.Div([
    html.H1("Resources"),
    html.P("Here you can find datasets, documentation, and helpful tools related to DTI prediction."),
])
