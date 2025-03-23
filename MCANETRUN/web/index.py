from app import app
from dash import Input, Output


content = {
    "en": {
        "navbar-title":"Drug Target Interaction Predict",
        "navbar-home":"Home",
        "navbar-resources":"Resources",
        "navbar-dti":"Drug Target Interaction Prediction",
        "navbar-about":"About Us",
        "navbar-contact":"Contact",
    },
    "cn": {
        "navbar-title":"药物靶标相互预测作用",
        "navbar-home":"首页",
        "navbar-resources":"资源",
        "navbar-dti":"药物靶标相互预测作用",
        "navbar-about":"关于我们",
        "navbar-contact":"联系我们"
    }
}

@app.callback(
    [
     Output("navbar-title", "children"),
     Output("navbar-home", "children"),
     Output("navbar-resources", "children"),
     Output("navbar-dti", "children"),
     Output("navbar-about", "children"),
     Output("navbar-contact", "children")
     ],
    [Input("language-dropdown", "value")]
)
def update_language(language):
    return (
        content[language]["navbar-title"],
        content[language]["navbar-home"],
        content[language]["navbar-resources"],
        content[language]["navbar-dti"],
        content[language]["navbar-about"],
        content[language]["navbar-contact"],
    )


if __name__ == "__main__":
    app.run(debug=True)
