import os
import re
from xml.sax.handler import property_interning_dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
import pandas as pd
import time

with open('Davis.txt', 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    parts = line.strip().split(' ', 4)
    if len(parts) == 5:
        compound_id, protein_name, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
        sequence, label = rest.rsplit(' ', 1)
        data.append({
            'compound_id': compound_id,
            'protein_name': protein_name,
            'smiles': smiles,
            'sequence': sequence,
            'label': int(label)
        })

ligands = set(d['smiles'] for d in data)
sort_ligands = sorted(ligands)

print(sort_ligands)
print(len(sort_ligands))


#
# # 设置下载目录
# download_dir_2D = '/Volumes/PASSPORT/FinalYearProject/MCANETRUN/2D/'
# download_dir_3D = '/Volumes/PASSPORT/FinalYearProject/MCANETRUN/3D/'
# for directory in [download_dir_2D, download_dir_3D]:
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
# # 设置 requests 重试策略
# session = requests.Session()
# retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
# session.mount('https://', HTTPAdapter(max_retries=retries))
#
# # 初始化列表
# PubChem_CID = []
# error_cid = []
# molecular_weight = []
# molecular_formula = []
# count = 0
#
# # 设置 ChromeDriver 路径
# local_path = "/Volumes/PASSPORT/FinalYearProject/MCANETRUN/chromedriver-mac-arm64/chromedriver"
# service = Service(local_path)
#
# for query in sort_ligands:
#     browser = None
#     try:
#         print(f"处理化合物: {query}")
#         browser = webdriver.Chrome(service=service)
#
#         # 获取 PubChem 页面
#         browser.get(f'https://pubchem.ncbi.nlm.nih.gov/#query={query}')
#
#         # 等待并点击搜索结果链接
#         result_link = WebDriverWait(browser, 30).until(
#             EC.presence_of_element_located((By.XPATH, "//a[@data-ga-action='result-link']"))
#         )
#         result_link.click()
#
#         # 等待页面跳转到化合物页面
#         WebDriverWait(browser, 20).until(
#             EC.url_contains("pubchem.ncbi.nlm.nih.gov/compound")
#         )
#
#         # 获取 CID
#         css_selector = "div.text-left.sm\\:table-cell.sm\\:p-2.pb-1.pl-2.sm\\:align-middle"
#         cid_element = WebDriverWait(browser, 20).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
#         )
#         cid_text = cid_element.text.strip()
#         print(f"compound_id={query}, PubChem CID={cid_text}")
#         PubChem_CID.append(cid_text)
#
#         # 下载 2D SDF
#         url_2d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid_text}/record/SDF/?record_type=2d"
#         response_2d = session.get(url_2d, timeout=60)
#         response_2d.raise_for_status()
#         with open(os.path.join(download_dir_2D, f"Structure2D_COMPOUND_CID_{cid_text}.sdf"), 'w') as f:
#             f.write(response_2d.text)
#         print(f"download_dir_2D {query} 下载成功")
#
#         # 下载 3D SDF
#         url_3d = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid_text}/record/SDF/?record_type=3d"
#         response_3d = session.get(url_3d, timeout=60)
#         if response_3d.status_code == 200:
#             with open(os.path.join(download_dir_3D, f"Structure3D_COMPOUND_CID_{cid_text}.sdf"), 'w') as f:
#                 f.write(response_3d.text)
#             print(f"download_dir_3D {query} 下载成功")
#         else:
#             print(f"download_dir_3D {query} 无 3D 结构")
#
#         # 获取 Molecular Weight
#         mw_element = WebDriverWait(browser, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//div[@class='break-words space-y-1']"))
#         )
#         mw_text = mw_element.text.strip()
#         print(f"Molecular Weight: {mw_text}")
#         molecular_weight.append(mw_text)  # 追加到列表
#
#         # 获取 Molecular Formula
#         span_elem = WebDriverWait(browser, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//ul[@class='list-none']//span"))
#         )
#         raw_text = span_elem.text
#         print("Raw text:", repr(raw_text))
#         formula = re.sub(r"\s+", "", raw_text)
#         print(f"Final formula: {formula}")
#         molecular_formula.append(formula)
#
#         count += 1
#         time.sleep(2)  # 每次请求后等待 2 秒
#
#     except (TimeoutException, WebDriverException) as e:
#         print(f"处理 {query} 超时: {str(e)}")
#         error_cid.append(query)
#     except requests.exceptions.RequestException as e:
#         print(f"下载 {query} (CID {cid_text if 'cid_text' in locals() else 'unknown'}) 失败: {str(e)}")
#         error_cid.append(query)
#     except Exception as e:
#         print(f"处理 {query} 失败: {str(e)}")
#         error_cid.append(query)
#     finally:
#         if browser:
#             browser.quit()
#
# print(f"成功处理 {count} 个化合物")
#
# # 保存结果到 CSV 文件
# results = {
#     'Query': list(sort_ligands)[:len(PubChem_CID)],  # 确保长度匹配
#     'PubChem CID': PubChem_CID,
#     'Molecular Weight': molecular_weight,
#     'Molecular Formula': molecular_formula
# }
# df = pd.DataFrame(results)
# df.to_csv('compound_properties.csv', index=False)
# print("已将结果保存到 compound_properties.csv")
#
# # 打印错误列表
# if error_cid:
#     print("以下化合物处理失败：")
#     for cid in error_cid:
#         print(f"  - {cid}")
# else:
#     print("所有化合物处理成功！")