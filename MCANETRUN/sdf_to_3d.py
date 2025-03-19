import dash
from dash import html
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
import base64

# 加载 SDF 文件
sdf_file = "/Volumes/PASSPORT/FinalYearProject/MCANETRUN/json/Conformer3D_COMPOUND_CID_44138048.sdf"  # 替换为你的 SDF 文件路径
mol_supplier = Chem.SDMolSupplier(sdf_file)
mol = next(mol_supplier)  # 取第一个分子

# 检查分子是否加载成功
if mol is None:
    raise ValueError("无法加载 SDF 文件，请检查文件路径或格式")

# 将分子转换为 SDF 字符串
sdf_string = Chem.MolToMolBlock(mol)

# 创建 Dash 应用
app = dash.Dash(__name__)

# 使用 py3Dmol 创建 3D 视图
viewer = py3Dmol.view(width=800, height=600)
viewer.addModel(sdf_string, 'sdf')
viewer.setStyle({'stick': {'radius': 0.1}, 'sphere': {'scale': 0.3}})  # 球棒模型
viewer.setBackgroundColor('black')  # 黑色背景
viewer.zoomTo()

# 将 py3Dmol 视图转换为 HTML
viewer_html = viewer._make_html()

# Dash 布局
app.layout = html.Div([
    html.H1("3D Molecule Visualization with py3Dmol", style={'color': 'white'}),
    html.Div(
        html.Iframe(
            srcDoc=viewer_html,
            style={'width': '800px', 'height': '600px', 'border': 'none'}
        ),
        style={'backgroundColor': 'black', 'padding': '10px'}
    )
])

# 运行应用
if __name__ == "__main__":
    app.run_server(debug=True)