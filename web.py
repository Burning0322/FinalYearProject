from dash import Dash,html,Input,dcc,Output
from rdkit.Chem import Draw
from rdkit import Chem
import json
from io import BytesIO
import base64

app = Dash(__name__)

import os
filepath_ligands = os.path.abspath('davis/ligands_can.txt')
filepath_proteins = os.path.abspath('davis/proteins.txt')

with open(filepath_ligands,'r') as file_ligands:
    smile_dict = json.load(file_ligands)

for smile in smile_dict.values():
    break

with open(filepath_proteins,'r') as file_proteins:
    proteins_dict = json.load(file_proteins)

for protein in proteins_dict.values():
    break


app.layout = [
    html.H1("DTI Prediction",style={'textAlign':'center'}),
    html.Div([
        html.Label("SMILES : ",style={'margin-right':'50px'}),
        dcc.Textarea(id = 'smiles-input',placeholder=smile,required=True,style={'width':'300px','margin-right':'50px','height':'100px'}),
        html.Label("Protein : ",style={'margin-right':'50px'}),
        dcc.Textarea(id = 'protein-input',placeholder=protein,required=True,style={'width':'300px','margin-right':'50px','height':'100px'}),
        html.Button("Submit", id='submit-button')
    ],style={'textAlign': 'center', 'height': '300px', 'width': 'auto', 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'justify-content': 'center', 'gap': '20px', 'overflow': 'auto'}),
    html.Div([
        html.Img(id='smiles-output'),
        dcc.Textarea(id='protein-output',readOnly=True,style={'width':'300px','height':'100px'})
    ],style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'justify-content': 'center', 'gap': '100px'})
    ]

# def smile2image(smile):
#     try:
#         mol = Chem.MolFromSmiles(smile)
#         if mol:
#             img = Draw.MolToImage(mol, size=(400, 200))
#             return smile2img(img)
#         else:
#             return None
#     except Exception as e:
#         return None

def smile2img(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def smile2image(smile):
    try:
        # 从输入 SMILES 创建初始分子对象
        mol = Chem.MolFromSmiles(smile)
        if mol:
            # 将分子转换为规范化的含手性信息的 SMILES
            canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=True)
            # 从规范化的 SMILES 重新创建分子对象
            mol = Chem.MolFromSmiles(canonical_smile)
            # 生成图像
            img = Draw.MolToImage(mol, size=(400, 200))
            return smile2img(img)
        else:
            return None
    except Exception as e:
        return None

@app.callback(
    [Output('smiles-output','src'),
    Output('protein-output','value')],
    [Input("smiles-input",'value'),
     Input("protein-input",'value')]

)
def update_output(smile,protein):
    image = smile2image(smile)
    print(f"SMILES:{smile},Protein:{protein}")
    return image,protein

if __name__ == '__main__':
    app.run_server(debug=True)