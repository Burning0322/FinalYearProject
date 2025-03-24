import os
from rdkit import Chem
import py3Dmol

sdf_3d_dir = "/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/3D"
output_dir = "assets/"

for filename in os.listdir(sdf_3d_dir):
    if filename.endswith(".sdf"):
        sdf_path = os.path.join(sdf_3d_dir, filename)
        output_html = os.path.join(output_dir, filename.replace(".sdf", "_3d.html"))

        # 如果已生成则跳过
        if os.path.exists(output_html):
            continue

        supplier = Chem.SDMolSupplier(sdf_path)
        mol = next((m for m in supplier if m is not None), None)
        if mol is None:
            print(f"[SKIP] {filename} 无有效分子")
            continue

        sdf_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=200, height=200)
        viewer.addModel(sdf_block, 'sdf')
        viewer.setStyle({'stick': {'radius': 0.1}, 'sphere': {'scale': 0.3}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()

        html = viewer._make_html()
        html = html.replace(
            '<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>',
            '<script src="/assets/3Dmol-min.js"></script>'
        )

        with open(output_html, "w") as f:
            f.write(html)
        print(f"[DONE] Generated: {output_html}")
